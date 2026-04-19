# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""Event system ports — stable interfaces for truth store and signal bus.

EventStore  = durable truth (Postgres run_events table)
EventSignalBus = wake-up transport (LISTEN/NOTIFY today, Redis/NATS tomorrow)

Rule: store events in Postgres. Wake consumers through a swappable signal bus.
      Never make the signal bus the truth.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from aiorch.events.models import RunEvent


class EventStore(ABC):
    """Durable event truth — append and replay run lifecycle events."""

    @abstractmethod
    def append(
        self,
        run_id: int,
        event_type: str,
        step_name: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> int:
        """Persist an event and return its auto-generated ID."""
        ...

    @abstractmethod
    def get_events(self, run_id: int, after_id: int = 0) -> list[RunEvent]:
        """Return events for a run, ordered by ID, optionally after a cursor."""
        ...

    @abstractmethod
    def get_latest_event_id(self, run_id: int) -> int | None:
        """Return the highest event ID for a run, or None if no events."""
        ...


class EventSignalBus(ABC):
    """Wake-up signal transport — NOT the source of truth.

    publish() sends a notification that new events exist.
    wait_for() blocks until a signal arrives or timeout expires.
    Missed signals are harmless — consumers always reconcile from EventStore.

    For race-free streaming, use register/unregister_waiter + wait_on_waiter
    to keep a waiter alive across fetch-then-wait cycles.
    """

    @abstractmethod
    def publish(self, run_id: int, event_id: int) -> None:
        """Signal that a new event was persisted for run_id."""
        ...

    @abstractmethod
    async def wait_for(self, run_id: int, timeout: float = 1.0) -> bool:
        """Wait for a signal for run_id. Returns True if signaled, False on timeout."""
        ...

    def register_waiter(self, run_id: int) -> Any:
        """Pre-register a persistent waiter for race-free streaming.

        Returns an opaque waiter object. Caller must call unregister_waiter()
        when done. Between register and unregister, no signals are lost.
        """
        return None  # Default: no persistent waiters (falls back to wait_for)

    def unregister_waiter(self, run_id: int, waiter: Any) -> None:
        """Remove a persistent waiter registered via register_waiter()."""

    async def wait_on_waiter(self, waiter: Any, timeout: float = 1.0) -> bool:
        """Wait on a pre-registered waiter. Returns True if signaled."""
        import asyncio
        await asyncio.sleep(timeout)
        return False  # Default: timeout-based polling

    async def start(self) -> None:
        """Start listening (e.g. LISTEN on Postgres). Called at API startup."""

    async def stop(self) -> None:
        """Stop listening and release resources. Called at API shutdown."""
