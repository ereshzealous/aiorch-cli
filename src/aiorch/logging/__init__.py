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

"""Structured step logging — PySpark-style event log for pipeline tracing.

Each step emits a StepEvent with timing, tokens, cost, and I/O previews.
Events are stored in:
  1. PostgreSQL (step_runs table) — queryable history
  2. Configurable log sinks — console, file, or multi-sink

Usage:
    logger = RunLogger(run_id=1, pipeline_name="pr-review")
    logger.step_start("review")
    logger.step_done("review", result, cost=0.003, prompt_tokens=500, completion_tokens=200, model="gpt-4o")
    logger.step_failed("review", error)
    logger.finish("success")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from aiorch.storage import log_step as _module_log_step, finish_run as _module_finish_run, get_store as get_store


LOG_DIR = Path.home() / ".aiorch" / "logs"


class LogLevel:
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"

    _ORDER = {"DEBUG": 0, "INFO": 1, "WARN": 2, "WARNING": 2, "ERROR": 3}

    @classmethod
    def should_print(cls, event_level: str, config_level: str) -> bool:
        """Return True if event_level >= config_level."""
        return cls._ORDER.get(event_level, 0) >= cls._ORDER.get(config_level.upper(), 2)


@dataclass
class StepEvent:
    """A single step execution event — the core tracing unit."""
    run_id: int
    step_name: str
    primitive: str
    status: str                     # started | success | failed | skipped
    timestamp: float                # unix timestamp of this event
    started_at: float = 0           # when the step started
    finished_at: float = 0          # when the step finished
    duration_ms: float = 0          # wall clock time
    model: str = ""                 # LLM model used (if applicable)
    prompt_tokens: int = 0          # input tokens
    completion_tokens: int = 0      # output tokens
    total_tokens: int = 0           # prompt + completion
    cost: float = 0                 # dollar cost
    input_preview: str = ""         # first 500 chars of input
    output_preview: str = ""        # first 500 chars of output
    error: str = ""                 # error message if failed
    cache_hit: bool = False         # whether result came from cache
    retry_attempt: int = 0          # which retry attempt (0 = first try)


class RunLogger:
    """Logger for a single pipeline run. Tracks all step events."""

    def __init__(self, run_id: int, pipeline_name: str, console_level: str = "WARNING", sink=None, store=None, redact: bool = True):
        from aiorch.logging.sinks import LogSink, create_sink
        self.run_id = run_id
        self.pipeline_name = pipeline_name
        self.console_level = console_level.upper()
        self.events: list[StepEvent] = []
        self._step_starts: dict[str, float] = {}
        self._redact = redact

        self._store = store

        if sink is None:
            self._sink: LogSink = create_sink("file")
        elif isinstance(sink, LogSink):
            self._sink = sink
        else:
            self._sink = create_sink(sink)

        if hasattr(self._sink, "set_run"):
            self._sink.set_run(run_id)

        self._write_event({
            "event": "run_start",
            "run_id": self.run_id,
            "pipeline": self.pipeline_name,
            "timestamp": time.time(),
        })

    def step_start(self, step_name: str, primitive: str) -> None:
        """Record step start."""
        now = time.time()
        self._step_starts[step_name] = now

        event = StepEvent(
            run_id=self.run_id,
            step_name=step_name,
            primitive=primitive,
            status="started",
            timestamp=now,
            started_at=now,
        )
        self.events.append(event)
        self._write_event(asdict(event))

    def step_done(
        self,
        step_name: str,
        primitive: str,
        result: Any = None,
        cost: float = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model: str = "",
        cache_hit: bool = False,
        provider_name: str = "",
        status: str = "success",
    ) -> None:
        """Record step completion with all metrics.

        ``status`` defaults to ``"success"`` for backward compatibility.
        Callers pass ``"partial"`` when the step completed with
        foreach warnings (timeouts or skip_on_error firings) so the UI
        can render a distinct amber badge and the Runs list filter can
        surface partial outcomes separately.
        """
        now = time.time()
        started = self._step_starts.get(step_name, now)
        duration_ms = (now - started) * 1000
        output_preview = str(result)[:500] if result else ""

        event = StepEvent(
            run_id=self.run_id,
            step_name=step_name,
            primitive=primitive,
            status=status,
            timestamp=now,
            started_at=started,
            finished_at=now,
            duration_ms=duration_ms,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
            output_preview=output_preview,
            cache_hit=cache_hit,
        )
        self.events.append(event)
        self._write_event(asdict(event))

        # Also log to storage (redact the preview)
        storage_preview = output_preview[:200]
        if self._redact:
            from aiorch.core.redaction import redact
            storage_preview = redact(storage_preview)
        self._log_step(
            self.run_id, step_name, primitive, status,
            started, now, cost=cost, output_preview=storage_preview,
            model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            provider_name=provider_name,
        )

    def step_failed(self, step_name: str, primitive: str, error: Exception) -> None:
        """Record step failure.

        Captures the formatted Python traceback alongside the error
        message so the Trace UI can show the full stack trace without
        the user having to find the executor's stdout. The traceback
        is the only part of the failure that doesn't already flow to
        the UI through the existing step_runs row, so this is where
        the gap is closed.
        """
        from aiorch.runtime.errors import classify_error
        import traceback as _tb_module

        now = time.time()
        started = self._step_starts.get(step_name, now)
        duration_ms = (now - started) * 1000
        error_type = classify_error(error, primitive)

        try:
            traceback_str = "".join(
                _tb_module.format_exception(type(error), error, error.__traceback__)
            )
        except Exception:
            traceback_str = f"<traceback formatting failed>\n{error!r}"

        TB_MAX_BYTES = 65536
        if len(traceback_str) > TB_MAX_BYTES:
            traceback_str = (
                traceback_str[:TB_MAX_BYTES]
                + "\n... [traceback truncated at 64KB]"
            )

        event = StepEvent(
            run_id=self.run_id,
            step_name=step_name,
            primitive=primitive,
            status="failed",
            timestamp=now,
            started_at=started,
            finished_at=now,
            duration_ms=duration_ms,
            error=str(error)[:500],
        )
        self.events.append(event)
        self._write_event(asdict(event))

        storage_error = str(error)[:200]
        storage_traceback = traceback_str
        if self._redact:
            from aiorch.core.redaction import redact
            storage_error = redact(storage_error)
            storage_traceback = redact(storage_traceback)
        self._log_step(
            self.run_id, step_name, primitive, "failed",
            started, now,
            error=storage_error,
            error_type=error_type,
            traceback=storage_traceback,
        )

    def step_skipped(self, step_name: str, primitive: str, reason: str = "condition false") -> None:
        """Record step skip (condition not met).

        Persists a step_runs row with status='skipped' so the Trace
        page can render a "skipped" pill distinct from success/failed.
        Duration is zero — skipped steps don't have a meaningful
        duration. `started_at == finished_at == now`.
        """
        now = time.time()
        event = StepEvent(
            run_id=self.run_id,
            step_name=step_name,
            primitive=primitive,
            status="skipped",
            timestamp=now,
            started_at=now,
            finished_at=now,
            duration_ms=0,
            output_preview=reason,
        )
        self.events.append(event)
        self._write_event(asdict(event))

        storage_reason = str(reason)[:200]
        if self._redact:
            from aiorch.core.redaction import redact
            storage_reason = redact(storage_reason)
        self._log_step(
            self.run_id, step_name, primitive, "skipped",
            now, now, cost=0, output_preview=storage_reason,
        )

    def finish(self, status: str, total_cost: float = 0) -> None:
        """Finalize the run log."""
        self._write_event({
            "event": "run_end",
            "run_id": self.run_id,
            "pipeline": self.pipeline_name,
            "status": status,
            "total_cost": total_cost,
            "total_steps": len([e for e in self.events if e.status in ("success", "partial", "failed")]),
            "timestamp": time.time(),
        })
        self._finish_run(self.run_id, status, total_cost)

    def log(self, step_name: str, level: str, message: str, data: Any = None) -> None:
        """Write a debug/info/warn/error log entry for a step.

        Use this for detailed tracing — prompt text, LLM responses, tool calls, etc.

        Args:
            step_name: Which step this log belongs to.
            level: One of DEBUG, INFO, WARN, ERROR.
            message: Human-readable log message.
            data: Optional structured data (dict, list, string — anything JSON-serializable).
        """
        event = {
            "event": "log",
            "run_id": self.run_id,
            "step_name": step_name,
            "level": level,
            "message": message,
            "timestamp": time.time(),
        }
        if data is not None:
            event["data"] = str(data)[:2000] if not isinstance(data, (dict, list)) else data
        self._write_event(event)

        # Print to console if level meets threshold
        from aiorch.ui.console import print_log
        print_log(step_name, level, message, data, self.console_level)

    def get_trace(self) -> list[dict]:
        """Get all events as dicts for display."""
        return [asdict(e) for e in self.events if e.status != "started"]

    def _log_step(self, *args, **kwargs) -> None:
        """Delegate to injected store or module-level function."""
        if self._store:
            self._store.log_step(*args, **kwargs)
        else:
            _module_log_step(*args, **kwargs)

    def _finish_run(self, *args, **kwargs) -> None:
        """Delegate to injected store or module-level function."""
        if self._store:
            self._store.finish_run(*args, **kwargs)
        else:
            _module_finish_run(*args, **kwargs)

    def _write_event(self, data: dict) -> None:
        """Write event to the configured sink, with redaction applied."""
        if self._redact:
            from aiorch.core.redaction import redact_dict
            data = redact_dict(data)
        self._sink.write(data)


def load_run_log(run_id: int) -> list[dict]:
    """Load all events from a run's log. Uses FileLogSink by default."""
    from aiorch.logging.sinks.file import FileLogSink
    sink = FileLogSink()
    return sink.query_run(run_id)
