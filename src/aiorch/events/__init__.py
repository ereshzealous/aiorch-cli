# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""Event system — durable truth + swappable signal bus.

Architecture:
  EventStore      = durable truth (Postgres run_events table)
  EventSignalBus  = wake-up transport (LISTEN/NOTIFY now, Redis/NATS later)
  StreamCoordinator = SSE streaming (wires store + bus)

Rule: store events in Postgres. Wake consumers through a swappable signal bus.
      Never make the signal bus the truth.

Usage:
    from aiorch.events import init_events, emit_step_start, emit_run_completed
    init_events()  # After init_storage()

    # Emit events (executor, scheduler, API)
    emit_run_started(run_id)
    emit_step_start(run_id, "fetch")
    emit_step_done(run_id, "fetch", result)
    emit_run_completed(run_id, "success", total_cost)

    # Stream events (API SSE endpoint)
    coordinator = get_coordinator()
    async for sse_line in coordinator.stream(run_id):
        yield sse_line
"""

from __future__ import annotations

import logging
import os
from typing import Any

from aiorch.events.coordinator import StreamCoordinator
from aiorch.events.models import RunEvent
from aiorch.events.ports import EventSignalBus, EventStore

logger = logging.getLogger("aiorch.events")

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_event_store: EventStore | None = None
_signal_bus: EventSignalBus | None = None
_coordinator: StreamCoordinator | None = None


def init_events(dsn: str | None = None) -> None:
    """Initialize the event system. Call after init_storage().

    Args:
        dsn: Postgres DSN for the LISTEN connection. If None, uses
             DATABASE_URL env var. Falls back to InMemorySignalBus
             if no DSN is available (tests).
    """
    global _event_store, _signal_bus, _coordinator

    if _event_store is not None:
        return  # Already initialized

    from aiorch.events.postgres_store import PostgresEventStore
    _event_store = PostgresEventStore()

    resolved_dsn = dsn or os.environ.get("DATABASE_URL")
    if resolved_dsn:
        from aiorch.events.postgres_signal import PostgresNotifyBus
        _signal_bus = PostgresNotifyBus(resolved_dsn)
    else:
        from aiorch.events.memory import InMemorySignalBus
        _signal_bus = InMemorySignalBus()

    _coordinator = StreamCoordinator(_event_store, _signal_bus)
    logger.info("Event system initialized (bus=%s)", type(_signal_bus).__name__)


def init_events_memory() -> None:
    """Initialize with in-memory implementations (for tests)."""
    global _event_store, _signal_bus, _coordinator

    from aiorch.events.memory import InMemoryEventStore, InMemorySignalBus
    _event_store = InMemoryEventStore()
    _signal_bus = InMemorySignalBus()
    _coordinator = StreamCoordinator(_event_store, _signal_bus)


async def start_signal_bus() -> None:
    """Start the LISTEN connection. Called from FastAPI lifespan startup."""
    if _signal_bus:
        await _signal_bus.start()


async def stop_signal_bus() -> None:
    """Stop the LISTEN connection. Called from FastAPI lifespan shutdown."""
    if _signal_bus:
        await _signal_bus.stop()


def reset_events() -> None:
    """Reset singletons (for tests)."""
    global _event_store, _signal_bus, _coordinator
    _event_store = None
    _signal_bus = None
    _coordinator = None


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def get_event_store() -> EventStore:
    if _event_store is None:
        raise RuntimeError("Event system not initialized. Call init_events() first.")
    return _event_store


def get_signal_bus() -> EventSignalBus:
    if _signal_bus is None:
        raise RuntimeError("Event system not initialized. Call init_events() first.")
    return _signal_bus


def get_coordinator() -> StreamCoordinator:
    if _coordinator is None:
        raise RuntimeError("Event system not initialized. Call init_events() first.")
    return _coordinator


# ---------------------------------------------------------------------------
# Convenience emitters — drop-in replacement for server/events.py
# ---------------------------------------------------------------------------

# LRU cache for run_id → workspace_id. Used by emit_event() to
# resolve the workspace channel without a DB round-trip per call.
# Populated on first lookup; entries never change (runs can't change
# workspaces), so no invalidation is needed.
_RUN_WS_CACHE: dict[int, str | None] = {}
_RUN_WS_CACHE_MAX = 1000

# Run-level event types that trigger a workspace channel publish.
# Step-level events stay per-run only — publishing every step event
# to the workspace channel would flood it with noise the Runs list
# page doesn't care about.
_WORKSPACE_CHANNEL_EVENTS = {"run_started", "run_completed"}


def _resolve_run_workspace(run_id: int) -> str | None:
    """Resolve a run's workspace_id with a small in-process LRU cache.

    Returns ``None`` when the run has no workspace (CLI mode) or the
    lookup fails. Bounded to 1000 entries; oldest entry is evicted
    when full (simple FIFO — no LRU library dependency).
    """
    if run_id in _RUN_WS_CACHE:
        return _RUN_WS_CACHE[run_id]
    try:
        from aiorch.storage import get_store
        row = get_store().query_one(
            "SELECT workspace_id FROM runs WHERE id = ?", (run_id,),
        )
    except Exception:
        row = None
    ws_id = row.get("workspace_id") if row else None
    # Simple eviction: drop the oldest entry when full
    if len(_RUN_WS_CACHE) >= _RUN_WS_CACHE_MAX:
        try:
            _RUN_WS_CACHE.pop(next(iter(_RUN_WS_CACHE)))
        except StopIteration:
            pass
    _RUN_WS_CACHE[run_id] = ws_id
    return ws_id


def emit_event(
    run_id: int,
    event_type: str,
    step_name: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    """Persist event to store and signal the bus.

    Four layers fan out from here:
      1. Durable append to ``run_events`` (Postgres) — authoritative.
      2. Local in-process ``SignalBus`` — wakes SSE handlers in the
         same API worker without polling.
      3. Redis pub/sub on ``aiorch:events:run:{id}`` — wakes per-run
         SSE handlers across API workers. Fail-through no-op when
         Redis is unavailable.
      4. Redis pub/sub on ``aiorch:events:ws:{ws_id}:runs`` — wakes
         workspace-wide SSE handlers (Runs list page). Emitted ONLY
         for run-level status events (``run_started``/``run_completed``)
         to keep the workspace channel low-volume. Step-level events
         stay per-run.

    Payloads are minimal signals — "something changed." Subscribers
    must refetch durable state from Postgres. Events are signals, not
    state, so at-most-once delivery is acceptable.

    Non-fatal: event write failure is logged but never crashes the caller.
    Run state (runs/step_runs tables) is the authority — events are supplementary
    for SSE streaming. A missed event degrades live trace, not pipeline execution.
    """
    event_id = 0
    try:
        event_id = get_event_store().append(run_id, event_type, step_name, payload)
        get_signal_bus().publish(run_id, event_id)
    except Exception:
        logger.warning("Failed to emit event %s for run #%d", event_type, run_id, exc_info=True)

    # R4: cross-worker signal via Redis. This is a nudge — subscribers
    # refetch state from Postgres. Payload kept minimal so missed or
    # out-of-order delivery doesn't corrupt UI state.
    try:
        from aiorch.cache import publish, channel_run, channel_workspace_runs
        publish(channel_run(run_id), {
            "run_id": run_id,
            "event_id": event_id,
            "event_type": event_type,
            "step_name": step_name,
        })
        # Workspace-level fan-out for the Runs list page. Only
        # run-level status events — step events stay per-run to
        # avoid flooding the workspace channel.
        if event_type in _WORKSPACE_CHANNEL_EVENTS:
            ws_id = _resolve_run_workspace(run_id)
            if ws_id:
                status = (payload or {}).get("status") if event_type == "run_completed" else None
                publish(channel_workspace_runs(ws_id), {
                    "run_id": run_id,
                    "event_type": event_type,
                    "status": status,
                })
    except Exception:
        # Publishing is best-effort; don't let a Redis blip block the event emission
        pass


def emit_run_started(run_id: int) -> None:
    emit_event(run_id, "run_started")


def emit_step_start(run_id: int, step_name: str) -> None:
    emit_event(run_id, "step_started", step_name=step_name)


def emit_step_done(run_id: int, step_name: str, result: Any = None, *, meta: dict | None = None, dur_ms: int = 0) -> None:
    payload: dict[str, Any] = {"dur_ms": dur_ms}
    if result is not None:
        try:
            payload["result"] = str(result)[:500]
        except Exception:
            pass
    if meta:
        payload["cost"] = meta.get("cost", 0)
        payload["model"] = meta.get("model", "")
        payload["prompt_tokens"] = meta.get("prompt_tokens", 0)
        payload["completion_tokens"] = meta.get("completion_tokens", 0)
        payload["cache_hit"] = meta.get("cache_hit", False)
        # Foreach partial-success signal — carries counts + iteration
        # indices so the UI can render a yellow 'partial' badge next to
        # the green check instead of a bare success.
        warnings = meta.get("warnings")
        if warnings:
            payload["warnings"] = warnings
    emit_event(run_id, "step_done", step_name=step_name, payload=payload)


def emit_step_progress(run_id: int, step_name: str, partial: str, token_count: int = 0) -> None:
    """Emit partial output during LLM streaming — lightweight, not persisted to DB."""
    try:
        # Signal only — don't persist every chunk to run_events (too noisy)
        get_signal_bus().publish(run_id, 0)
    except Exception:
        pass
    # Broadcast via a lightweight in-memory channel instead
    _progress_state[(run_id, step_name)] = partial


# In-memory partial output state — latest chunk per (run_id, step_name)
_progress_state: dict[tuple[int, str], str] = {}


def get_step_progress(run_id: int, step_name: str) -> str | None:
    """Get the latest partial output for a running step."""
    return _progress_state.get((run_id, step_name))


def clear_step_progress(run_id: int, step_name: str) -> None:
    """Clear partial output after step completes."""
    _progress_state.pop((run_id, step_name), None)


def emit_step_failed(run_id: int, step_name: str, error: Any = None) -> None:
    payload = {"error": str(error)[:500]} if error else None
    emit_event(run_id, "step_failed", step_name=step_name, payload=payload)


def emit_step_skipped(run_id: int, step_name: str, reason: str = "") -> None:
    """Emit a step_skipped event when a step's `condition:` evaluates false.

    Distinct from step_done so the Trace page can render a "skipped"
    pill — important for pipelines with mutually-exclusive branches
    where one branch should be visibly skipped, not appear as a
    successful run returning empty.
    """
    payload = {"reason": str(reason)[:500]} if reason else None
    emit_event(run_id, "step_skipped", step_name=step_name, payload=payload)


def emit_run_completed(run_id: int, status: str, total_cost: float = 0) -> None:
    emit_event(run_id, "run_completed", payload={
        "status": status,
        "total_cost": total_cost,
    })


def get_events(run_id: int, after_id: int = 0) -> list[RunEvent]:
    """Get events for a run — returns RunEvent objects."""
    return get_event_store().get_events(run_id, after_id)
