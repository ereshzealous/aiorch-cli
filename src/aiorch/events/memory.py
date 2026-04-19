# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""In-memory event implementations for testing — no database, no network."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from aiorch.events.models import RunEvent
from aiorch.events.ports import EventSignalBus, EventStore


class InMemoryEventStore(EventStore):
    """Pure in-memory event store using a Python list."""

    def __init__(self):
        self._events: list[RunEvent] = []
        self._next_id = 0

    def append(
        self,
        run_id: int,
        event_type: str,
        step_name: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> int:
        self._next_id += 1
        self._events.append(RunEvent(
            id=self._next_id,
            run_id=run_id,
            event_type=event_type,
            step_name=step_name,
            payload=payload or {},
            created_at=time.time(),
        ))
        return self._next_id

    def get_events(self, run_id: int, after_id: int = 0) -> list[RunEvent]:
        return [e for e in self._events if e.run_id == run_id and e.id > after_id]

    def get_latest_event_id(self, run_id: int) -> int | None:
        matching = [e.id for e in self._events if e.run_id == run_id]
        return max(matching) if matching else None


class InMemorySignalBus(EventSignalBus):
    """Pure in-memory signal bus using asyncio.Event per waiter."""

    def __init__(self):
        self._waiters: dict[int, list[asyncio.Event]] = {}

    def publish(self, run_id: int, event_id: int) -> None:
        for ev in self._waiters.get(run_id, []):
            ev.set()

    def register_waiter(self, run_id: int) -> asyncio.Event:
        event = asyncio.Event()
        self._waiters.setdefault(run_id, []).append(event)
        return event

    def unregister_waiter(self, run_id: int, waiter: asyncio.Event) -> None:
        waiter_list = self._waiters.get(run_id, [])
        try:
            waiter_list.remove(waiter)
        except ValueError:
            pass
        if not waiter_list:
            self._waiters.pop(run_id, None)

    async def wait_on_waiter(self, waiter: asyncio.Event, timeout: float = 1.0) -> bool:
        try:
            await asyncio.wait_for(waiter.wait(), timeout=timeout)
            waiter.clear()
            return True
        except asyncio.TimeoutError:
            return False

    async def wait_for(self, run_id: int, timeout: float = 1.0) -> bool:
        waiter = self.register_waiter(run_id)
        try:
            return await self.wait_on_waiter(waiter, timeout)
        finally:
            self.unregister_waiter(run_id, waiter)
