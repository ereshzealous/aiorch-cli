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

"""Load aiorch.yaml config and resolve ${ENV_VAR} references."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

CONFIG_NAMES = ["aiorch.yaml", "aiorch.yml", ".aiorch.yaml"]

_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


class LLMConfig(BaseModel):
    model: str = "gpt-4o-mini"
    provider: str = "openai"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class DefaultsConfig(BaseModel):
    timeout: str | None = None
    retry: int = 0


class ConnectorConfig(BaseModel):
    """Config for a logging sink. Only ``type``, ``path``, and ``format`` are
    read by the built-in file/stdout sinks; ``options`` is passed through
    unchanged so custom sinks can consume whatever they need."""
    type: str = "file"
    path: str | None = None
    format: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class LoggingConfig(BaseModel):
    level: str = "WARNING"
    sink: str | ConnectorConfig | list[ConnectorConfig] = "file"


class ServerConfig(BaseModel):
    """Config for ``aiorch serve``. The CLI entrypoint sets
    AIORCH_AUTH_ENABLED=false before importing the server, so single-user
    local use doesn't need a key; keep auth on by default otherwise."""
    host: str = "127.0.0.1"
    port: int = 7842
    secret: str | None = None
    open_browser: bool = True
    auth_enabled: bool = True
    secret_key: str | None = None


class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    env: dict[str, str] = Field(default_factory=dict)
    redaction: Any = Field(default=None)
    policy: Any = Field(default=None)


def _resolve_env(value: Any) -> Any:
    """Replace ${ENV_VAR} with its value from the environment."""
    if not isinstance(value, str):
        return value
    return _ENV_PATTERN.sub(
        lambda m: os.environ.get(m.group(1), ""),
        value,
    )


def _resolve_env_recursive(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _resolve_env_recursive(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_resolve_env_recursive(v) for v in data]
    return _resolve_env(data)


def find_config(start: Path | None = None) -> Path | None:
    """Walk up from start directory looking for a config file."""
    cwd = (start or Path.cwd()).resolve()
    for directory in (cwd, *cwd.parents):
        for name in CONFIG_NAMES:
            p = directory / name
            if p.exists():
                return p
    return None


def load_config(path: str | Path | None = None) -> Config:
    """Load config from a YAML file. Falls back to defaults if not found.

    The ``env:`` block is injected into os.environ *before* ``${VAR}``
    resolution, so aiorch.yaml can reference its own env values."""
    if path is None:
        path = find_config()

    if path is None:
        return Config()

    path = Path(path)
    if not path.exists():
        return Config()

    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        return Config()

    raw_env = raw.get("env", {})
    if isinstance(raw_env, dict):
        for key, value in raw_env.items():
            if key not in os.environ:
                # Resolve ${VAR} against the current environment so shell-exported
                # values can flow through. `env: SLACK_WEBHOOK: ${SLACK_WEBHOOK}`
                # is now a no-op passthrough; writing a literal value still works.
                os.environ[key] = _resolve_env(str(value))

    resolved = _resolve_env_recursive(raw)
    return Config(**resolved)


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Reset singleton (useful for tests)."""
    global _config
    _config = None
