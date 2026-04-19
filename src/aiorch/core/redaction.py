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

"""Log redaction — prevent secrets from leaking into logs, traces, and history.

Provides configurable regex-based redaction with sensible defaults for
common API key formats (OpenAI, Anthropic, GitHub, AWS, Slack).
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Default patterns for common API key formats
# ---------------------------------------------------------------------------

_DEFAULT_PATTERNS: list[str] = [
    r"sk-[a-zA-Z0-9]{20,}",              # OpenAI keys
    r"sk-ant-[a-zA-Z0-9\-]{20,}",        # Anthropic keys
    r"sk-or-v1-[a-zA-Z0-9]{20,}",        # OpenRouter keys
    r"ghp_[a-zA-Z0-9]{36}",              # GitHub PATs
    r"gho_[a-zA-Z0-9]{36}",              # GitHub OAuth tokens
    r"github_pat_[a-zA-Z0-9_]{40,}",     # GitHub fine-grained PATs
    r"xoxb-[a-zA-Z0-9\-]+",              # Slack bot tokens
    r"xoxp-[a-zA-Z0-9\-]+",              # Slack user tokens
    r"AKIA[A-Z0-9]{16}",                 # AWS access key IDs
]

_DEFAULT_ENV_KEYS: list[str] = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "GITHUB_TOKEN",
    "SLACK_WEBHOOK_URL",
    "AWS_SECRET_ACCESS_KEY",
    "SMTP_PASS",
    "SMTP_PASSWORD",
]

REDACTED = "***REDACTED***"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class RedactionConfig(BaseModel):
    """Configuration for log redaction."""
    enabled: bool = True
    patterns: list[str] = Field(default_factory=lambda: list(_DEFAULT_PATTERNS))
    env_keys: list[str] = Field(default_factory=lambda: list(_DEFAULT_ENV_KEYS))


# ---------------------------------------------------------------------------
# Redactor builder
# ---------------------------------------------------------------------------

def build_redactor(config: RedactionConfig | None = None) -> Callable[[str], str]:
    """Build a compiled redaction function from config.

    Returns a callable that replaces secret patterns with REDACTED.
    Pre-compiles all patterns for performance.
    """
    if config is None:
        config = RedactionConfig()

    if not config.enabled:
        return lambda text: text

    # Collect all patterns
    all_patterns: list[str] = list(config.patterns)

    # Add current env var values as literal patterns
    for key in config.env_keys:
        value = os.environ.get(key)
        if value and len(value) >= 8:  # Only redact if value is long enough to be a real secret
            all_patterns.append(re.escape(value))

    if not all_patterns:
        return lambda text: text

    # Compile into a single alternation regex for performance
    combined = re.compile("|".join(f"({p})" for p in all_patterns))

    def _redact(text: str) -> str:
        return combined.sub(REDACTED, text)

    return _redact


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_redactor: Callable[[str], str] | None = None


def _get_redactor() -> Callable[[str], str]:
    """Get or build the global redactor from config."""
    global _redactor
    if _redactor is None:
        try:
            from aiorch.core.config import get_config
            cfg = get_config()
            redaction_cfg = getattr(cfg, "redaction", None)
            _redactor = build_redactor(redaction_cfg)
        except Exception:
            _redactor = build_redactor(RedactionConfig())
    return _redactor


def reset_redactor() -> None:
    """Reset the global redactor (useful for tests)."""
    global _redactor
    _redactor = None


def redact(text: str) -> str:
    """Redact secrets from a string using the global config."""
    if not isinstance(text, str):
        return text
    return _get_redactor()(text)


def redact_dict(data: dict) -> dict:
    """Recursively redact all string values in a dict.

    Returns a new dict — does not mutate the original.
    """
    redactor = _get_redactor()

    def _redact_value(value: Any) -> Any:
        if isinstance(value, str):
            return redactor(value)
        if isinstance(value, dict):
            return {k: _redact_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_redact_value(v) for v in value]
        return value

    return {k: _redact_value(v) for k, v in data.items()}
