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

# NOTE: load_dotenv() is NOT called here. CLI reads aiorch.yaml only.
# Platform (aiorch serve) calls load_dotenv() explicitly before init.

CONFIG_NAMES = ["aiorch.yaml", "aiorch.yml", ".aiorch.yaml"]

_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


class LLMConfig(BaseModel):
    model: str = "gpt-4o-mini"
    provider: str = "openai"  # "openai" (default) or "litellm" (optional)
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class DefaultsConfig(BaseModel):
    timeout: str | None = None
    retry: int = 0


class ConnectorConfig(BaseModel):
    """Universal connector config — works for any storage or sink backend.

    Each backend reads only the fields it needs. Unknown fields are ignored.

    Examples:
        type: sqlite
        path: ~/.aiorch/history.db

        type: postgres
        host: localhost
        port: 5432
        database: aiorch
        username: ${DB_USER}
        password: ${DB_PASSWORD}

        type: s3
        bucket: my-logs
        region: us-east-1
    """
    type: str = "sqlite"

    # Relational (postgres, mysql)
    host: str | None = None
    port: int | None = None
    database: str | None = None
    username: str | None = None
    password: str | None = None
    url: str | None = None          # shorthand — overrides host/port/etc

    # File-based (sqlite, file sink)
    path: str | None = None

    # Object store (s3)
    bucket: str | None = None
    region: str | None = None

    # Stdout sink
    format: str | None = None

    # Type-specific extras
    options: dict[str, Any] = Field(default_factory=dict)

    def get_url(self) -> str:
        """Build connection URL from fields. Returns url if set, else constructs it."""
        if self.url:
            return self.url
        if self.type == "sqlite":
            return self.path or "~/.aiorch/history.db"

        auth = ""
        if self.username:
            auth = self.username
            if self.password:
                auth += f":{self.password}"
            auth += "@"

        host = self.host or "localhost"
        port = f":{self.port}" if self.port else ""
        db = f"/{self.database}" if self.database else ""

        return f"{self.type}://{auth}{host}{port}{db}"


class LoggingConfig(BaseModel):
    level: str = "WARNING"
    sink: str | ConnectorConfig | list[ConnectorConfig] = "file"


class StorageConfig(BaseModel):
    """Storage backend config — PostgreSQL via DATABASE_URL env var."""
    type: str = "postgres"

    # Connection fields (used when DATABASE_URL is not set)
    host: str | None = None
    port: int | None = None
    database: str | None = None
    username: str | None = None
    password: str | None = None
    url: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    pool_size: int = 10


class ServerConfig(BaseModel):
    """Server configuration for aiorch serve."""
    host: str = "127.0.0.1"
    port: int = 7842
    secret: str | None = None
    open_browser: bool = True
    # Secure-by-default. The CLI entrypoint (aiorch run/validate/etc.)
    # sets AIORCH_AUTH_ENABLED=false explicitly before importing the
    # server module (see cli.py), so single-user CLI use keeps working.
    # Server mode (aiorch serve / uvicorn app:create_app) inherits True
    # unless an operator intentionally flips it — and even then the
    # server-lifespan guard refuses to boot without AIORCH_DEV_MODE.
    # See executor CRITICAL arc / the auth_enabled-footgun memo.
    auth_enabled: bool = True
    secret_key: str | None = None  # For JWT signing + API key encryption


class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    env: dict[str, str] = Field(default_factory=dict)  # Injected into os.environ (webhook URLs, SMTP, etc.)
    redaction: Any = Field(default=None)  # RedactionConfig, loaded lazily to avoid circular imports
    policy: Any = Field(default=None)    # PolicyConfig, loaded lazily to avoid circular imports


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
    cwd = start or Path.cwd()
    for name in CONFIG_NAMES:
        p = cwd / name
        if p.exists():
            return p
    return None


def load_config(path: str | Path | None = None) -> Config:
    """Load config from a YAML file. Falls back to defaults if not found.

    The env: block is injected into os.environ BEFORE ${VAR} resolution,
    so aiorch.yaml can reference its own env values:
        env:
          OPENAI_API_KEY: sk-...
        llm:
          api_key: ${OPENAI_API_KEY}   # resolves from env: block above
    """
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

    # Step 1: Inject env: block into os.environ BEFORE resolving ${}
    raw_env = raw.get("env", {})
    if isinstance(raw_env, dict):
        for key, value in raw_env.items():
            if key not in os.environ:
                os.environ[key] = str(value)

    # Step 2: Now resolve ${} references (env: values are available)
    resolved = _resolve_env_recursive(raw)
    return Config(**resolved)


# Singleton — loaded once, used everywhere
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
