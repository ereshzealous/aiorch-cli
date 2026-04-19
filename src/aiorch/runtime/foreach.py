# Copyright 2026 Eresh Gorantla
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Foreach loop orchestration — sequential and parallel iteration.

Per-iteration timeout semantics (when ``timeout`` is set):
  - Each iteration gets its own time budget.
  - If an iteration times out, its slot in the result list becomes a
    ``[TIMEOUT]: ...`` sentinel string. Other iterations continue.
  - If EVERY iteration times out, the step fails with a RuntimeError
    that lists each item — running with all timeouts is never useful
    downstream.
  - Non-timeout exceptions in an iteration still fail the whole step
    (fail-fast) — a crash is usually deterministic and continuing
    would just mask the bug.

Downstream steps must handle the ``[TIMEOUT]`` sentinel the same way
they handle ``[MCP Error]`` — check the prefix and branch accordingly.
The agent's error-propagation detection (``agent.py``) already treats
these as failures if every tool call produced one.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

from aiorch.core import template

logger = logging.getLogger("aiorch.foreach")

TIMEOUT_SENTINEL_PREFIX = "[TIMEOUT]:"
SKIPPED_SENTINEL_PREFIX = "[SKIPPED]:"

# Prefixes that mark a value as "upstream failed" — used by the
# ``skip_on_error`` flag to decide whether to skip an iteration. Keep
# aligned with the sentinels produced by ``aiorch.tools`` (MCP errors)
# and this module (timeout sentinels).
_ERROR_SENTINEL_PREFIXES = (
    TIMEOUT_SENTINEL_PREFIX,
    SKIPPED_SENTINEL_PREFIX,
    "[MCP Error]",
    "[MCP Registry Error]",
    "[ERROR]",
)


def _is_error_sentinel(value: Any) -> bool:
    """True when a value is one of the known error-sentinel strings
    emitted by MCP tool failures, iteration timeouts, or upstream skips.
    """
    if not isinstance(value, str):
        return False
    head = value.lstrip()[:32]
    return any(head.startswith(p) for p in _ERROR_SENTINEL_PREFIXES)


def resolve_foreach_items(raw_items: list | str, context: dict[str, Any]) -> list:
    """Resolve foreach items from a list or template string.

    Handles:
      - list → return as-is
      - str "{{var}}" → look up var in context directly (preserves dicts/objects)
      - str → template resolve, then try JSON array, then comma-split
    """
    if isinstance(raw_items, list):
        return raw_items

    # Direct variable reference — look up in context to preserve types (list of dicts, etc.)
    if isinstance(raw_items, str):
        stripped = raw_items.strip()
        if stripped.startswith("{{") and stripped.endswith("}}"):
            var_name = stripped[2:-2].strip()
            if var_name in context:
                value = context[var_name]
                if isinstance(value, list):
                    return value

    resolved = template.resolve(raw_items, context)
    if isinstance(resolved, list):
        return resolved
    if isinstance(resolved, str):
        try:
            parsed = json.loads(resolved)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return [i.strip() for i in resolved.split(",")]
    return [resolved]


async def run_foreach(
    items: list,
    context: dict[str, Any],
    run_one: Callable[[dict[str, Any]], Awaitable[Any]],
    parallel: bool = False,
    timeout: float | None = None,
    step_name: str | None = None,
    skip_on_error: str | None = None,
) -> list[Any]:
    """Execute a callback for each item, sequentially or in parallel.

    Args:
        items: List of items to iterate over.
        context: Base context dict. Each iteration gets ``{**context, "item": item}``.
        run_one: Async callback that receives the per-item context.
        parallel: If True, run all items concurrently via asyncio.gather.
        timeout: Per-iteration timeout in seconds. When set, an iteration
            exceeding this duration is replaced by a ``[TIMEOUT]:`` sentinel
            and other iterations continue. If EVERY iteration times out the
            function raises RuntimeError. Non-timeout exceptions still
            propagate (fail-fast).
        step_name: Used only for log context when a partial timeout fires.
        skip_on_error: Name of a previously-produced list in ``context``.
            Before running iteration ``i``, check if that list's element at
            index ``i`` is an error sentinel (``[TIMEOUT]:``, ``[MCP Error]:``,
            ``[ERROR]:``, or ``[SKIPPED]:``). If so, skip this iteration and
            emit a ``[SKIPPED]:`` sentinel that propagates the upstream
            cause to the next step.

    Returns:
        List of results in item order. Timed-out iterations are represented
        by ``[TIMEOUT]: ...`` sentinel strings. Skipped iterations (via
        ``skip_on_error``) are represented by ``[SKIPPED]: ...`` sentinels.
    """
    # Resolve the skip list up-front so each iteration can cheaply index
    # into it. None/missing list disables skip_on_error for this call.
    upstream = None
    if skip_on_error:
        val = context.get(skip_on_error)
        if isinstance(val, list):
            upstream = val
        else:
            logger.warning(
                "Step %r — skip_on_error=%r but context value is not a list "
                "(type=%s); skip_on_error will be ignored for this step.",
                step_name or "<foreach>", skip_on_error, type(val).__name__,
            )

    skipped_indices: list[int] = []

    async def _run_one(i: int, item: Any) -> Any:
        # skip_on_error: if upstream output at index i is a sentinel, emit
        # a [SKIPPED]: sentinel without running the iteration at all. No
        # LLM, tool, or shell call is made.
        if upstream is not None and i < len(upstream):
            up = upstream[i]
            if _is_error_sentinel(up):
                skipped_indices.append(i)
                # Propagate a short fingerprint of the upstream cause so
                # operators can trace what failed without cross-referencing.
                preview = str(up).replace("\n", " ")[:80]
                return (
                    f"{SKIPPED_SENTINEL_PREFIX} iter {i} — upstream "
                    f"{skip_on_error!r} has error sentinel: {preview}"
                )

        run_ctx = {**context, "item": item}
        if timeout is None:
            return await run_one(run_ctx)
        try:
            return await asyncio.wait_for(run_one(run_ctx), timeout=timeout)
        except asyncio.TimeoutError:
            sentinel = (
                f"{TIMEOUT_SENTINEL_PREFIX} iter {i} exceeded {timeout}s "
                f"(item={item!r})"
            )
            return sentinel

    if parallel:
        # gather with default return_exceptions=False — non-timeout errors
        # still raise and fail the step. Timeouts are caught inside _run_one
        # and returned as sentinels, so they never reach gather as exceptions.
        results = list(await asyncio.gather(
            *[_run_one(i, item) for i, item in enumerate(items)]
        ))
    else:
        results = []
        for i, item in enumerate(items):
            results.append(await _run_one(i, item))

    timeout_count = 0
    if timeout is not None:
        timeout_count = sum(
            1 for r in results
            if isinstance(r, str) and r.startswith(TIMEOUT_SENTINEL_PREFIX)
        )
        if timeout_count and timeout_count == len(results):
            # All iterations timed out — running downstream on all-sentinels
            # produces garbage, so fail the step with actionable detail.
            bullets = "\n  ".join(str(r) for r in results)
            raise RuntimeError(
                f"Foreach step{' ' + repr(step_name) if step_name else ''} "
                f"had all {len(results)} iterations time out at {timeout}s.\n  "
                f"{bullets}\n"
                f"Likely causes: LLM reasoning stall (Gemini often takes "
                f"30-60s after tool calls), slow target service, or timeout "
                f"set too aggressively. Raise the step timeout, try a faster "
                f"model, or inspect registry/executor logs for the run. "
                f"MCP sessions (if any) recover via 5-minute idle pool TTL."
            )
        if timeout_count:
            # Partial timeout — step still succeeds, but operators need to
            # see which iterations fell out so they can diagnose without
            # digging through the output list in the UI.
            logger.warning(
                "Step %r — %d of %d foreach iteration(s) timed out at %ss; "
                "continuing with partial results. Affected iterations will "
                "appear in the output list as '[TIMEOUT]:' sentinels. "
                "Downstream steps should branch on the prefix.",
                step_name or "<foreach>", timeout_count, len(results), timeout,
            )
    if skipped_indices:
        # Step still succeeds — the upstream error is the real problem.
        # But operators benefit from a single WARN line that tells them
        # which indices fell out of this step without scanning the full
        # output list in the UI.
        logger.warning(
            "Step %r — %d of %d foreach iteration(s) skipped via "
            "skip_on_error=%r (upstream sentinel at indices %s). Slots "
            "carry '[SKIPPED]:' sentinels. No LLM/tool calls were made "
            "for those iterations.",
            step_name or "<foreach>", len(skipped_indices), len(results),
            skip_on_error, skipped_indices,
        )

    # Record warnings onto the per-step metadata carried in the shared
    # _meta dict. The executor's on_done hook reads it and passes it to
    # emit_step_done → event payload → UI. Status stays 'success'; the
    # warnings field is a separate signal the UI can render as a yellow
    # badge alongside the green check.
    if (timeout_count or skipped_indices) and step_name:
        _record_foreach_warnings(
            context=context,
            step_name=step_name,
            results=results,
            timeout_count=timeout_count,
            skipped_indices=skipped_indices,
            timeout=timeout,
            skip_on_error=skip_on_error,
        )
    return results


def _record_foreach_warnings(
    *, context: dict, step_name: str, results: list,
    timeout_count: int, skipped_indices: list[int],
    timeout: float | None, skip_on_error: str | None,
) -> None:
    """Write a structured warnings summary onto ``context[_meta][step_name]``.

    Schema (kept stable — the UI contracts on these keys):

    ::

        {
          "timeouts": int,
          "skipped_on_error": int,
          "iterations_total": int,
          "timed_out_indices": list[int],
          "skipped_indices": list[int],
          "message": str,   # one-line human summary for tooltips
        }
    """
    from aiorch.constants import META_KEY

    meta_root = context.get(META_KEY)
    if meta_root is None:
        meta_root = {}
        context[META_KEY] = meta_root
    step_meta = meta_root.setdefault(step_name, {})

    # Identify exactly which iteration indices fell out — lets the UI
    # list "iter 2" instead of just "1 of 3 failed".
    timed_out_indices = [
        i for i, r in enumerate(results)
        if isinstance(r, str) and r.startswith(TIMEOUT_SENTINEL_PREFIX)
    ]

    parts: list[str] = []
    if timeout_count:
        parts.append(
            f"{timeout_count} of {len(results)} iterations timed out"
            f"{f' at {timeout}s' if timeout is not None else ''}"
        )
    if skipped_indices:
        parts.append(
            f"{len(skipped_indices)} of {len(results)} iterations skipped "
            f"via skip_on_error={skip_on_error!r}"
        )
    message = "; ".join(parts) if parts else ""

    step_meta["warnings"] = {
        "timeouts": timeout_count,
        "skipped_on_error": len(skipped_indices),
        "iterations_total": len(results),
        "timed_out_indices": timed_out_indices,
        "skipped_indices": list(skipped_indices),
        "message": message,
    }
