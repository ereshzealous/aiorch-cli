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

"""Stdout log sink — prints structured events to stdout. For CI/CD pipelines."""

from __future__ import annotations

import json
import sys

from aiorch.logging.sinks import LogSink


class StdoutLogSink(LogSink):
    """Prints events as JSON lines to stdout. Useful in CI environments."""

    def __init__(self, fmt: str | None = None):
        self._format = fmt or "json"

    def write(self, event: dict) -> None:
        if self._format == "json":
            sys.stdout.write(json.dumps(event, default=str) + "\n")
            sys.stdout.flush()
        else:
            # Simple text format
            etype = event.get("event", event.get("status", ""))
            step = event.get("step_name", "")
            msg = event.get("message", "")
            sys.stdout.write(f"[{etype}] {step} {msg}\n")
            sys.stdout.flush()

    def close(self) -> None:
        pass
