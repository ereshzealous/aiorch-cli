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

"""Multi-sink — fans out events to multiple sinks at once."""

from __future__ import annotations

from aiorch.logging.sinks import LogSink, QueryableLogSink


class MultiLogSink(LogSink):
    """Writes events to multiple sinks. For production: disk + S3 + Postgres."""

    def __init__(self, sinks: list[LogSink]):
        self._sinks = sinks

    def write(self, event: dict) -> None:
        for sink in self._sinks:
            sink.write(event) 

    def flush(self) -> None:
        for sink in self._sinks:
            sink.flush()

    def close(self) -> None:
        for sink in self._sinks:
            sink.close()

    def query_run(self, run_id: int) -> list[dict]:
        """Delegate to the first QueryableLogSink in the list."""
        for sink in self._sinks:
            if isinstance(sink, QueryableLogSink):
                return sink.query_run(run_id)
        raise NotImplementedError("No sink supports reads")
