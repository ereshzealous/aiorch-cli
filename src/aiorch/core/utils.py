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

"""Shared utilities — duration parsing, input resolution, and common helpers."""

from __future__ import annotations

from typing import Any

from aiorch.core import template
from aiorch.core.loader import stringify


def parse_duration(value: str | None, default: float | None = None) -> float | None:
    """Parse a duration string into seconds.

    Supports: '500ms', '5s', '1m', bare numbers (treated as seconds).

    Args:
        value: Duration string, or None.
        default: Value to return when *value* is None/empty.

    Returns:
        Duration in seconds, or *default* if *value* is falsy.
    """
    if not value:
        return default
    s = value.strip().lower()
    if s.endswith("ms"):
        return float(s[:-2]) / 1000
    if s.endswith("s"):
        return float(s[:-1])
    if s.endswith("m"):
        return float(s[:-1]) * 60
    return float(s)


def resolve_input(
    input_val: str | list[str] | None, context: dict[str, Any]
) -> str | None:
    """Resolve step input to a string.

    Handles three forms:
      - None  → None
      - str   → template-resolve + stringify
      - list  → resolve each, join with blank line

    This is the single shared implementation used by runtime, flow, and agent.
    """
    if input_val is None:
        return None
    if isinstance(input_val, str):
        result = template.resolve(input_val, context)
        return stringify(result) if not isinstance(result, str) else result
    if isinstance(input_val, list):
        parts = []
        for item in input_val:
            result = template.resolve(item, context)
            parts.append(stringify(result) if not isinstance(result, str) else result)
        return "\n\n".join(parts)
    return None
