# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""Central dispatch wrapper for every connector operation.

Every connector invocation — from pipeline steps, from input
resolution, from the API test-connection endpoint — flows through
``execute_connector_operation()``. Responsibilities:

  1. Enforce the connector's declared ``capabilities``
  2. Enforce the ``is_read_only`` flag (defense in depth over
     capabilities — even if admin accidentally granted write caps,
     the flag blocks all writes)
  3. Resolve the right handler for (type, subtype)
  4. Dispatch with timing + error capture
  5. Emit a ``connector_usage`` row for audit and lineage
  6. Emit Prometheus metrics with bounded cardinality
  7. Emit an ``audit_logs`` entry (enterprise trail)
  8. Return the handler's result or re-raise with context

One wrapper means every handler gets these behaviors for free.
No handler can forget metrics, no handler can bypass read-only.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from aiorch.connectors import (
    Connector,
    ConnectorOperationDenied,
    WRITE_OPERATIONS,
)
from aiorch.connectors.handlers import get_handler

logger = logging.getLogger("aiorch.connectors.dispatch")


# ---------------------------------------------------------------------------
# ${KEY} token resolver
#
# Connector config + secret fields can store references like
# "${SLACK_WEBHOOK_URL}" instead of literal values. At dispatch time we
# walk the merged config + the operation params and substitute any such
# token with the matching value from the run's workspace secrets (preferred)
# or workspace configs (fallback). Unknown keys are left as literals so
# the failure surfaces at the handler rather than being silently blanked.
#
# This lets you store the real credential in one place (the Secrets page),
# reference it by name from any number of connectors, and rotate it in a
# single edit — every connector picks up the new value on the next run.
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _resolve_tokens(value: Any, lookup: dict[str, str]) -> Any:
    """Recursively replace ``${KEY}`` tokens in string leaves.

    dict and list containers are walked; other scalar types are
    returned unchanged. Unknown keys are left as literals so
    misconfigurations fail loudly in the handler.
    """
    if isinstance(value, str):
        if "${" not in value:
            return value
        return _TOKEN_RE.sub(
            lambda m: lookup.get(m.group(1), m.group(0)),
            value,
        )
    if isinstance(value, dict):
        return {k: _resolve_tokens(v, lookup) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_tokens(v, lookup) for v in value]
    return value


def _build_token_lookup(
    workspace_id: str | None, pipeline_name: str | None = None,
) -> dict[str, str]:
    """Assemble the ``${KEY}`` lookup for a workspace + pipeline scope.

    Order of precedence (later entries overwrite earlier — rightmost wins):
      1. Workspace-wide configs     (plaintext, weakest)
      2. Pipeline-scoped configs    (overlay for this specific pipeline)
      3. Workspace-wide secrets     (encrypted)
      4. Pipeline-scoped secrets    (overlay for this specific pipeline)

    Both scopes are queried whenever ``pipeline_name`` is supplied, and
    the pipeline-specific row wins for any key present at both levels.
    Previously this function always called workspace-wide only (codex
    finding #11) — pipeline-scoped values were silently ignored even
    though the UI + API + `workspace_configs.pipeline_name` column were
    all designed to support them.

    Returns an empty dict when workspace_id is None or storage isn't
    initialized (tests, CLI probes). The resolver treats an empty
    lookup as a no-op.
    """
    if not workspace_id:
        return {}

    lookup: dict[str, str] = {}

    # --- Configs (weakest-first: workspace-wide, then pipeline-scoped) ---
    try:
        from aiorch.storage import get_store
        store = get_store()
        rows = store.query_all(
            "SELECT key, value FROM workspace_configs "
            "WHERE workspace_id = ? AND pipeline_name IS NULL",
            (workspace_id,),
        )
        for r in rows:
            lookup[r["key"]] = r["value"]

        if pipeline_name:
            rows = store.query_all(
                "SELECT key, value FROM workspace_configs "
                "WHERE workspace_id = ? AND pipeline_name = ?",
                (workspace_id, pipeline_name),
            )
            for r in rows:
                lookup[r["key"]] = r["value"]
    except Exception as e:
        logger.debug("workspace_configs lookup skipped: %s", e)

    # --- Secrets (win over configs; pipeline-scoped beats workspace-wide) ---
    try:
        from aiorch.connectors._hooks import resolve_secrets
        # Workspace-wide secrets first.
        lookup.update(resolve_secrets(workspace_id, pipeline_name=None))
        # Pipeline-scoped overlay — overwrites workspace-wide keys when
        # the pipeline declares its own value.
        if pipeline_name:
            lookup.update(resolve_secrets(workspace_id, pipeline_name=pipeline_name))
    except Exception as e:
        logger.debug("workspace_secrets lookup skipped: %s", e)

    return lookup


async def execute_connector_operation(
    *,
    connector: Connector,
    operation: str,
    params: dict,
    context: dict | None = None,
    step_name: str | None = None,
) -> Any:
    """The single entrypoint for every connector operation.

    Args:
        connector: Fully resolved connector (secrets decrypted).
        operation: Canonical operation name (query, write, send, ...).
        params: Operation-specific arguments — passed through to the
            handler unchanged, already Jinja2-resolved by the caller.
        context: Pipeline execution context. Used for run_id (lineage),
            audit log attribution, and pass-through to the handler.
        step_name: Step name for lineage. Optional — defaults to
            "<inline>" when called from an input loader, which has
            no enclosing step.

    Returns:
        Whatever the handler's execute() returns. Canonical shape
        per operation type.

    Raises:
        ConnectorOperationDenied: capability or read-only violation.
        Handler-native exceptions on runtime failure (asyncpg.*,
        botocore.*, httpx.*, etc.) — dispatch wrapper adds
        connector_usage + metrics + audit trail but does NOT wrap
        the exception class. Callers can still catch asyncpg.*
        exceptions directly.
    """
    step_name = step_name or "<inline>"
    ctx = context or {}

    # 1. Capability check
    if operation not in connector.capabilities:
        raise ConnectorOperationDenied(
            f"Connector '{connector.name}' does not support operation "
            f"'{operation}'. Declared capabilities: {connector.capabilities}"
        )

    # 2. Read-only enforcement (defense in depth over capabilities)
    if connector.is_read_only and operation in WRITE_OPERATIONS:
        raise ConnectorOperationDenied(
            f"Connector '{connector.name}' is marked read-only. "
            f"Write operation '{operation}' rejected at dispatch."
        )

    # 3. Resolve handler
    handler = get_handler(connector.type, connector.subtype)

    # 3a. Resolve ${KEY} tokens in merged config + params against the
    # run's workspace secrets / configs. The lookup is built once per
    # dispatch so a single op doesn't round-trip storage multiple times.
    # Pipeline name is threaded through the lookup so pipeline-scoped
    # configs + secrets override workspace-wide ones (codex #11 — the
    # prior implementation silently ignored pipeline-scoped values).
    workspace_id = ctx.get("__workspace_id__") or connector.workspace_id
    pipeline_name = None
    from aiorch.constants import RUNTIME_META_KEY
    meta = ctx.get(RUNTIME_META_KEY)
    if isinstance(meta, dict):
        pipeline_name = meta.get("pipeline_name")
    # Fall back to the connector's own pipeline_name when context
    # doesn't carry one (e.g. input-resolution path called outside a
    # step's runtime meta block).
    pipeline_name = pipeline_name or connector.pipeline_name
    token_lookup = _build_token_lookup(workspace_id, pipeline_name)
    merged_cfg = connector.merged_config()
    if token_lookup:
        merged_cfg = _resolve_tokens(merged_cfg, token_lookup)
        params = _resolve_tokens(params, token_lookup)

    # 4. Dispatch with timing + error capture
    start = time.time()
    outcome = "success"
    error_message: str | None = None
    result: Any = None
    rows_affected: int | None = None
    bytes_transferred: int | None = None

    try:
        result = await handler.execute(
            operation=operation,
            params=params,
            connector_config=merged_cfg,
            context=ctx,
        )

        # Extract observability fields from canonical return shapes
        if isinstance(result, dict):
            rows_affected = result.get("rows_affected")
            bytes_transferred = result.get("bytes_transferred")
        elif isinstance(result, list) and all(isinstance(r, dict) for r in result):
            # database query returns list[dict]
            rows_affected = len(result)

    except Exception as exc:
        outcome = "error"
        error_message = f"{type(exc).__name__}: {str(exc)[:480]}"
        logger.warning(
            "Connector op failed: %s.%s on %s — %s",
            connector.type, operation, connector.name[:40], error_message,
        )
        raise

    finally:
        duration_ms = int((time.time() - start) * 1000)

        # 5. connector_usage row (lineage + audit)
        _record_usage(
            run_id=ctx.get("__run_id__"),
            step_name=step_name,
            connector_id=connector.id,
            operation=operation,
            outcome=outcome,
            duration_ms=duration_ms,
            rows_affected=rows_affected,
            bytes_transferred=bytes_transferred,
            error_message=error_message,
        )

        # 6. Prometheus metrics
        _emit_metrics(
            connector=connector,
            operation=operation,
            outcome=outcome,
            duration_ms=duration_ms,
            rows_affected=rows_affected,
            bytes_transferred=bytes_transferred,
            error_message=error_message,
        )

        # 7. audit_logs entry (enterprise trail)
        _emit_audit_log(
            ctx,
            connector=connector,
            operation=operation,
            outcome=outcome,
        )

    return result


# ---------------------------------------------------------------------------
# Observability emitters (each tolerates a missing backend)
# ---------------------------------------------------------------------------


def _record_usage(
    *,
    run_id: Any,
    step_name: str,
    connector_id: str,
    operation: str,
    outcome: str,
    duration_ms: int,
    rows_affected: int | None,
    bytes_transferred: int | None,
    error_message: str | None,
) -> None:
    """Write a connector_usage row. No-op if run_id is missing."""
    if run_id is None:
        return   # not running inside a pipeline run (e.g. test-connection probe)

    try:
        from aiorch.storage import get_store
        get_store().execute_sql(
            "INSERT INTO connector_usage "
            "(run_id, step_name, connector_id, operation, outcome, "
            " duration_ms, rows_affected, bytes_transferred, error_message, "
            " created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (run_id, step_name, connector_id, operation) "
            "DO UPDATE SET outcome = EXCLUDED.outcome, "
            "              duration_ms = EXCLUDED.duration_ms, "
            "              rows_affected = EXCLUDED.rows_affected, "
            "              bytes_transferred = EXCLUDED.bytes_transferred, "
            "              error_message = EXCLUDED.error_message",
            (
                int(run_id), step_name, connector_id, operation, outcome,
                duration_ms, rows_affected, bytes_transferred, error_message,
                time.time(),
            ),
        )
    except Exception as e:
        # Don't let usage logging break the pipeline. Log and move on.
        logger.debug("Failed to write connector_usage row: %s", e)


def _emit_metrics(
    *,
    connector: Connector,
    operation: str,
    outcome: str,
    duration_ms: int,
    rows_affected: int | None,
    bytes_transferred: int | None,
    error_message: str | None,
) -> None:
    """Emit Prometheus metrics. Gated behind metrics.is_enabled()."""
    try:
        from aiorch.metrics import (
            inc_connector_operation,
            observe_connector_duration,
            inc_connector_rows,
            inc_connector_bytes,
            inc_connector_error,
        )
        inc_connector_operation(
            connector_id=connector.id,
            type=connector.type,
            subtype=connector.subtype,
            operation=operation,
            outcome=outcome,
        )
        observe_connector_duration(
            type=connector.type,
            subtype=connector.subtype,
            operation=operation,
            seconds=duration_ms / 1000.0,
        )
        if rows_affected is not None:
            inc_connector_rows(
                type=connector.type,
                subtype=connector.subtype,
                operation=operation,
                count=rows_affected,
            )
        if bytes_transferred is not None:
            inc_connector_bytes(
                type=connector.type,
                subtype=connector.subtype,
                operation=operation,
                count=bytes_transferred,
            )
        if outcome == "error" and error_message:
            # Bucket the error class — first word of the message
            # (which is the exception class name from _record_usage)
            error_class = error_message.split(":")[0] or "unknown"
            inc_connector_error(
                type=connector.type,
                subtype=connector.subtype,
                error_class=error_class[:32],
            )
    except ImportError:
        # Metric helpers not yet implemented — Phase F adds them.
        # The wrapper still works end-to-end without metrics.
        pass
    except Exception as e:
        logger.debug("Failed to emit connector metrics: %s", e)


def _emit_audit_log(ctx: dict, *, connector: Connector, operation: str, outcome: str) -> None:
    """Write an audit_logs row. Best effort — never raises.

    Platform registers a real sink during lifespan (``set_audit_sink(audit)``);
    CLI leaves the default no-op so pipelines run with no Platform deps."""
    try:
        from aiorch.connectors._hooks import audit

        # The server middleware stores the authenticated user in the
        # execution context under __user__ for run-time contexts.
        user = ctx.get("__user__")
        user_id = user.get("user_id") if isinstance(user, dict) else None

        audit(
            org_id=connector.org_id,
            user_id=user_id,
            action=f"connector.{connector.type}.{operation}",
            workspace_id=connector.workspace_id,
            resource_type="connector",
            resource_id=connector.id,
            detail={
                "name": connector.name,
                "subtype": connector.subtype,
                "outcome": outcome,
            },
        )
    except Exception as e:
        # Audit is best-effort. Don't fail the operation if the
        # audit infrastructure isn't available (CLI, tests, etc.).
        logger.debug("Failed to write audit log for connector op: %s", e)


__all__ = ["execute_connector_operation"]
