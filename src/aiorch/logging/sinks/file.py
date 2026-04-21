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

"""File-based log sink — writes JSONL to disk. Default sink."""

from __future__ import annotations

import json
from pathlib import Path

from aiorch.logging.sinks import QueryableLogSink


DEFAULT_LOG_DIR = Path.home() / ".aiorch" / "logs"


class FileLogSink(QueryableLogSink):
    """Writes events as JSONL to ~/.aiorch/logs/run-<id>.jsonl.

    Each write is crash-safe — opens file, appends, closes.
    Supports query_run() for reading events back.
    """

    def __init__(self, path: str | Path | None = None):
        self._log_dir = Path(path) if path else DEFAULT_LOG_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._current_file: Path | None = None

    def set_run(self, run_id: int) -> Path:
        """Set the current run file. Truncates any stale log from a prior session."""
        self._current_file = self._log_dir / f"run-{run_id}.jsonl"
        if self._current_file.exists():
            self._current_file.unlink()
        return self._current_file

    def write(self, event: dict) -> None:
        if self._current_file is None:
            run_id = event.get("run_id", "unknown")
            self.set_run(run_id)
        with open(self._current_file, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def close(self) -> None:
        self._current_file = None

    def query_run(self, run_id: int) -> list[dict]:
        """Read back all events for a run from its JSONL file."""
        path = self._log_dir / f"run-{run_id}.jsonl"
        if not path.exists():
            return []
        events = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events
