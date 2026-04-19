# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""StreamCoordinator — connects EventStore + EventSignalBus for SSE streaming.

Replaces the old RunStreamer polling loop. Flow:
  1. Replay existing events from EventStore
  2. Wait on EventSignalBus for wake-up signal
  3. Fetch delta from EventStore
  4. Emit SSE-formatted events
  5. Reconciliation poll every 30s as safety net (catches missed signals)
"""

from __future__ import annotations

import json
import time
from typing import AsyncGenerator

from aiorch.events.models import RunEvent
from aiorch.events.ports import EventSignalBus, EventStore

# How often to reconcile even without a signal (safety net)
RECONCILE_INTERVAL = 30  # seconds

# Close stream after this many seconds with no events at all
IDLE_TIMEOUT = 300  # 5 minutes

# Heartbeat interval (seconds of idle before sending heartbeat)
HEARTBEAT_INTERVAL = 15


class StreamCoordinator:
    """SSE stream coordinator backed by EventStore + EventSignalBus.

    The EventStore is the truth. The EventSignalBus is only the wake-up.
    If a signal is missed, the reconciliation poll catches it.
    """

    def __init__(self, store: EventStore, bus: EventSignalBus):
        self._store = store
        self._bus = bus

    async def stream(self, run_id: int, after_id: int = 0) -> AsyncGenerator[str, None]:
        """Yield SSE events for a run — signal-driven with poll fallback.

        Args:
            run_id: The run to stream events for.
            after_id: Resume from this event ID (from Last-Event-ID header).
                      0 = replay all events from the beginning.

        1. Replays existing events from the store (after after_id)
        2. Waits for signals (or reconciliation timeout)
        3. Fetches new events from store on each wake-up
        4. Sends heartbeats during idle periods
        5. Closes after IDLE_TIMEOUT seconds with no events
        """
        last_event_id = after_id
        last_event_time = time.time()

        # Phase 1: Replay existing events
        existing = self._store.get_events(run_id, after_id=after_id)
        for ev in existing:
            last_event_id = ev.id
            last_event_time = time.time()
            yield _format_sse(ev)
            if ev.is_terminal:
                return

        # Phase 2: Live streaming — signal-driven with reconciliation
        # Register a persistent waiter BEFORE entering the loop to close the
        # race window between fetching events and waiting for signals.
        waiter = self._bus.register_waiter(run_id)
        last_heartbeat = time.time()

        try:
            while True:
                # Fetch any new events (waiter is already registered, no lost signals)
                new_events = self._store.get_events(run_id, after_id=last_event_id)

                if new_events:
                    last_event_time = time.time()
                    last_heartbeat = time.time()
                    for ev in new_events:
                        last_event_id = ev.id
                        yield _format_sse(ev)
                        if ev.is_terminal:
                            return
                else:
                    now = time.time()

                    # Emit step progress (in-memory partial LLM output)
                    progress_sent = _emit_progress(run_id)
                    if progress_sent:
                        for sse_line in progress_sent:
                            yield sse_line
                        last_heartbeat = now

                    # Heartbeat
                    if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                        yield _format_heartbeat()
                        last_heartbeat = now

                    # Idle timeout
                    if now - last_event_time >= IDLE_TIMEOUT:
                        return

                # Wait for next signal or reconciliation timeout — shorter for progress
                wait_timeout = min(RECONCILE_INTERVAL, HEARTBEAT_INTERVAL, 1.0)
                await self._bus.wait_on_waiter(waiter, timeout=wait_timeout)
        finally:
            self._bus.unregister_waiter(run_id, waiter)


def _emit_progress(run_id: int) -> list[str]:
    """Check for in-memory step progress and format as SSE events."""
    from aiorch.events import _progress_state

    lines = []
    for (rid, step_name), partial in list(_progress_state.items()):
        if rid != run_id:
            continue
        data = json.dumps({
            "event": "step_progress",
            "step": step_name,
            "partial": partial[-2000:],  # Last 2000 chars to limit payload
            "chars": len(partial),
        })
        lines.append(f"data: {data}\n\n")
    return lines


def _format_sse(ev: RunEvent) -> str:
    """Convert a RunEvent to an SSE line with id: for browser reconnect."""
    event_map = {
        "run_started": "run_started",
        "step_started": "step_start",
        "step_done": "step_done",
        "step_failed": "step_error",
        "run_completed": "run_done",
    }

    result: dict = {
        "event": event_map.get(ev.event_type, ev.event_type),
        "ts": ev.created_at,
    }

    if ev.step_name:
        result["step"] = ev.step_name

    if ev.event_type == "step_done":
        result["status"] = "success"
        # Include step metadata from payload
        if ev.payload:
            for k in ("result", "cost", "model", "prompt_tokens", "completion_tokens", "cache_hit", "dur_ms"):
                if k in ev.payload:
                    result[k] = ev.payload[k]
    elif ev.event_type == "step_failed":
        result["status"] = "failed"
        result["error"] = ev.payload.get("error", "") if ev.payload else ""
    elif ev.event_type == "step_skipped":
        result["status"] = "skipped"
        if ev.payload and ev.payload.get("reason"):
            result["reason"] = ev.payload["reason"]
    elif ev.event_type == "run_completed":
        result["status"] = ev.payload.get("status", "success")
        result["total_cost"] = ev.payload.get("total_cost", 0)

    # Include id: field so browser EventSource can resume from Last-Event-ID
    return f"id: {ev.id}\ndata: {json.dumps(result)}\n\n"


def _format_heartbeat() -> str:
    return f"data: {json.dumps({'event': 'heartbeat', 'ts': time.time()})}\n\n"
