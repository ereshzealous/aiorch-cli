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

"""Log sink abstraction — pluggable destinations for pipeline events.

Usage:
    # Default (file) — zero config
    sink = create_sink("file")

    # From config
    sink = create_sink(LogSinkConfig(type="file", path="~/.aiorch/logs"))

    # Multiple sinks
    sink = create_sink([
        LogSinkConfig(type="file"),
        LogSinkConfig(type="stdout"),
    ])
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from pydantic import BaseModel, Field


class LogSink(ABC):
    """Abstract base for write-only log event destinations.

    Implement write() and close() to add a new backend.
    Events are plain dicts — the dict IS the schema.

    For sinks that also support reading events back (e.g., file-based),
    extend QueryableLogSink instead.
    """

    @abstractmethod
    def write(self, event: dict) -> None:
        """Write a single event. Must be crash-safe."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        ...

    def flush(self) -> None:
        """Force flush. Default: no-op."""
        pass


class QueryableLogSink(LogSink):
    """LogSink that also supports reading events back.

    Extend this for backends that can serve reads (file, database).
    Write-only sinks (stdout, S3 append-only) should extend LogSink instead.
    """

    @abstractmethod
    def query_run(self, run_id: int) -> list[dict]:
        """Read back all events for a run."""
        ...


class LogSinkConfig(BaseModel):
    """Configuration for a single log sink. Uses same shape as ConnectorConfig."""
    type: str = "file"
    path: str | None = None
    host: str | None = None
    port: int | None = None
    database: str | None = None
    username: str | None = None
    password: str | None = None
    url: str | None = None
    bucket: str | None = None
    region: str | None = None
    format: str | None = None
    options: dict = Field(default_factory=dict)


# --- Sink registry ---

SinkFactory = Callable[[LogSinkConfig], LogSink]
_SINK_FACTORIES: dict[str, SinkFactory] = {}


def register_sink(name: str, factory: SinkFactory) -> None:
    """Register a log sink backend.

    Args:
        name: Sink type name (e.g., "file", "stdout", "s3").
        factory: Callable that accepts a LogSinkConfig and returns a LogSink.
    """
    _SINK_FACTORIES[name] = factory


def _register_builtin_sinks() -> None:
    """Register the built-in sink backends."""
    def _file_factory(cfg: LogSinkConfig) -> LogSink:
        from aiorch.logging.sinks.file import FileLogSink
        return FileLogSink(path=cfg.path)

    def _stdout_factory(cfg: LogSinkConfig) -> LogSink:
        from aiorch.logging.sinks.stdout import StdoutLogSink
        return StdoutLogSink(fmt=cfg.format)

    register_sink("file", _file_factory)
    register_sink("stdout", _stdout_factory)


_register_builtin_sinks()


def create_sink(config: str | LogSinkConfig | list | None = None) -> LogSink:
    """Factory — create a LogSink from config.

    Args:
        config: "file", LogSinkConfig, list of configs, or None (default=file)

    Returns:
        A LogSink instance.
    """
    if config is None:
        return create_sink(LogSinkConfig(type="file"))

    if isinstance(config, list):
        from aiorch.logging.sinks.multi import MultiLogSink
        return MultiLogSink([create_sink(c) for c in config])

    if isinstance(config, dict):
        return create_sink(LogSinkConfig(**config))

    if isinstance(config, str):
        return create_sink(LogSinkConfig(type=config))

    factory = _SINK_FACTORIES.get(config.type)
    if factory is None:
        raise ValueError(
            f"Unknown sink type: '{config.type}'. "
            f"Registered: {', '.join(_SINK_FACTORIES.keys())}. "
            f"Use register_sink() to add new backends."
        )
    return factory(config)
