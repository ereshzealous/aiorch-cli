# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""Run event domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunEvent:
    """A single event in a run's lifecycle.

    Stored durably in the run_events table. SSE streaming replays
    these events — they are the source of truth, not process memory.
    """

    id: int
    run_id: int
    event_type: str  # run_started, step_started, step_done, step_failed, run_completed
    step_name: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0

    @property
    def is_terminal(self) -> bool:
        return self.event_type == "run_completed"
