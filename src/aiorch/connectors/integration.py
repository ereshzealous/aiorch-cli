# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""Pipeline runtime integration for connectors.

This module is the bridge between the dag / action runtime and the
dispatch wrapper. It handles three entry points:

  1. ``type: connector`` inputs  — resolved at context-build time
     before any step runs. The input becomes whatever the connector
     operation returns (list[dict] for a query, str for object read,
     etc.).

  2. ``action: connector`` steps — explicit invocation inside the
     pipeline. Reads ``uses``, ``operation`` and the operation's
     canonical params from ``step.config``.

  3. Auto-resolve fallback for legacy action shortcuts (``slack``,
     ``email``, ``s3``, ``kafka``, ``teams``, ``discord``, ``webhook``).
     If the step's config supplies a ``uses:`` field, the call is
     routed to the managed connector instead of the env-var path.

All three paths converge on ``execute_connector_operation()`` so the
read-only gate, capability check, usage row, metrics and audit trail
run once per invocation — no matter how the pipeline invoked it.
"""

from __future__ import annotations

import logging
from typing import Any

from aiorch.connectors import (
    ConnectorError,
    ConnectorNotFound,
    get_connector_store,
)
from aiorch.connectors.dispatch import execute_connector_operation

logger = logging.getLogger("aiorch.connectors.integration")


# Fields on a connector config dict that are reserved for dispatch
# routing — everything else is forwarded to the handler as params.
_ROUTING_FIELDS = {"type", "uses", "operation", "track"}


def _scope_from_context(context: dict) -> tuple[str | None, str | None, str | None]:
    """Pull (org_id, workspace_id, pipeline_name) from execution context.

    The executor injects these as ``__org_id__`` / ``__workspace_id__``
    keys, and the runtime metadata dict (``_meta``) carries the
    pipeline name. Missing values return None and the store falls
    through to higher scopes.
    """
    org_id = context.get("__org_id__")
    workspace_id = context.get("__workspace_id__")

    pipeline_name: str | None = None
    from aiorch.constants import RUNTIME_META_KEY
    meta = context.get(RUNTIME_META_KEY)
    if isinstance(meta, dict):
        pipeline_name = meta.get("pipeline_name")

    return org_id, workspace_id, pipeline_name


def _extract_params(config: dict) -> dict[str, Any]:
    """Everything in ``config`` that isn't a routing field is a param."""
    return {k: v for k, v in config.items() if k not in _ROUTING_FIELDS}


async def resolve_and_execute(
    config: dict,
    context: dict,
    *,
    step_name: str | None = None,
    default_type: str | None = None,
    default_subtype: str | None = None,
) -> Any:
    """Resolve a connector reference and run the operation.

    Args:
        config: Dict containing ``uses`` (connector name), ``operation``,
            and any operation-specific params. Can also include a
            ``type:`` field (ignored for routing).
        context: Pipeline execution context — used to pull scope
            (org / workspace / pipeline) and attribution.
        step_name: Step name for lineage. Optional — when resolving
            an input, there is no enclosing step and we pass the
            input key instead.
        default_type / default_subtype: If set and ``uses`` is not
            provided, look up the default connector of this type in
            scope. Used by legacy action shortcuts.

    Returns:
        Whatever the underlying handler returned.

    Raises:
        ConnectorNotFound: no connector matched the name or default.
        ConnectorError and subclasses: anything the dispatcher raises.
        ValueError: config is missing ``operation``.
    """
    connector_name = config.get("uses")
    operation = config.get("operation")
    if not operation:
        raise ValueError(
            "Connector invocation requires 'operation' "
            "(e.g. query, read, write, send, publish)."
        )

    org_id, workspace_id, pipeline_name = _scope_from_context(context)
    if not org_id:
        raise ConnectorError(
            "Connector resolution needs an org context. Connectors are a "
            "Platform-only feature — CLI pipelines cannot use "
            "'type: connector' inputs or 'action: connector' steps. "
            "Run via aiorch-executor (Platform mode) instead."
        )

    try:
        store = get_connector_store()
    except Exception as e:
        raise ConnectorError(
            f"Connector store unavailable: {e}. "
            "Connectors require Platform mode (Postgres-backed storage)."
        ) from e

    connector = None
    if connector_name:
        connector = store.resolve_by_name(
            org_id=org_id,
            name=connector_name,
            workspace_id=workspace_id,
            pipeline_name=pipeline_name,
            include_secrets=True,
        )
    elif default_type and default_subtype:
        connector = store.resolve_default(
            org_id=org_id,
            type=default_type,
            subtype=default_subtype,
            workspace_id=workspace_id,
            pipeline_name=pipeline_name,
            include_secrets=True,
        )
        if connector is None:
            raise ConnectorNotFound(
                f"No default {default_type}/{default_subtype} connector "
                f"found in scope. Either set one as default in the UI or "
                f"pass an explicit 'uses:' field."
            )
    else:
        raise ValueError(
            "Connector invocation requires either 'uses' (connector name) "
            "or a default type+subtype lookup."
        )

    params = _extract_params(config)

    return await execute_connector_operation(
        connector=connector,
        operation=operation,
        params=params,
        context=context,
        step_name=step_name or "<inline>",
    )


async def try_legacy_connector_auto_resolve(
    step_config: dict,
    context: dict,
    *,
    step_name: str,
    type_name: str,
    subtype_name: str,
    operation: str,
    param_map: dict[str, str] | None = None,
) -> Any:
    """Route a legacy action step through the connector dispatcher.

    Called by ``_action_slack`` / ``_action_email`` / ... when the step
    explicitly sets ``config.uses: <name>``. Maps legacy field names
    (``message``, ``to``, ``subject``, etc.) to canonical handler
    params via ``param_map``.

    Args:
        step_config: The step's ``config`` dict (already with templates
            resolved — the caller does that once).
        context: Pipeline execution context.
        step_name: Step name for lineage.
        type_name / subtype_name: Expected connector type+subtype.
            Raises ``ConnectorError`` if the resolved connector is a
            different type — catches "I referenced a Postgres connector
            from an ``action: slack`` step" mistakes.
        operation: Canonical operation name to dispatch.
        param_map: Legacy field → handler field renames. Keys not
            in ``param_map`` are forwarded as-is.
    """
    connector_name = step_config.get("uses")
    if not connector_name:
        return None   # caller falls back to legacy env-var path

    # Build the handler params from the legacy fields.
    param_map = param_map or {}
    params: dict[str, Any] = {}
    for legacy_key, value in step_config.items():
        if legacy_key in _ROUTING_FIELDS:
            continue
        canonical_key = param_map.get(legacy_key, legacy_key)
        params[canonical_key] = value

    org_id, workspace_id, pipeline_name = _scope_from_context(context)
    if not org_id:
        raise ConnectorError(
            "Legacy action referenced a connector via 'uses:' but the "
            "run has no org context — connectors are Platform-only."
        )

    store = get_connector_store()
    connector = store.resolve_by_name(
        org_id=org_id,
        name=connector_name,
        workspace_id=workspace_id,
        pipeline_name=pipeline_name,
        include_secrets=True,
    )

    if connector.type != type_name or connector.subtype != subtype_name:
        raise ConnectorError(
            f"Step '{step_name}' action expects {type_name}/{subtype_name} "
            f"but 'uses: {connector_name}' resolved to "
            f"{connector.type}/{connector.subtype}."
        )

    return await execute_connector_operation(
        connector=connector,
        operation=operation,
        params=params,
        context=context,
        step_name=step_name,
    )


__all__ = [
    "resolve_and_execute",
    "try_legacy_connector_auto_resolve",
]
