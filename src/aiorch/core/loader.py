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

"""Smart input loader — detects type and loads accordingly.

Handles: file paths (txt, md, json, yaml, sql, csv, etc.), inline JSON, inline YAML, plain text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


# Extensions that get parsed as structured data
_YAML_EXTS = {".yaml", ".yml"}
_JSON_EXTS = {".json"}

# Everything else is read as plain text
_TEXT_EXTS = {".txt", ".md", ".sql", ".csv", ".tsv", ".log", ".sh", ".py", ".go", ".js", ".ts", ".xml", ".html", ".toml", ".ini", ".cfg", ".env"}


def load_value(raw: str) -> Any:
    """Detect what `raw` is and return the appropriate Python object.

    Resolution order:
        1. File path  → read file, parse if structured (json/yaml), else text
        2. JSON string → parse as dict/list
        3. YAML string → parse as dict/list (only if it looks structured)
        4. Plain text  → return as-is
    """
    stripped = raw.strip()

    # 1. File path?
    path = Path(stripped)
    if path.exists() and path.is_file():
        return _load_file(path)

    # 2. Inline JSON? (starts with { or [)
    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # 3. Inline YAML with structure? (contains ":" suggesting key-value)
    if ":" in stripped and "\n" in stripped:
        try:
            parsed = yaml.safe_load(stripped)
            if isinstance(parsed, (dict, list)):
                return parsed
        except yaml.YAMLError:
            pass

    # 4. Plain text
    return raw


def _load_file(path: Path) -> Any:
    """Read a file and parse based on extension."""
    ext = path.suffix.lower()
    content = path.read_text()

    if ext in _JSON_EXTS:
        return json.loads(content)

    if ext in _YAML_EXTS:
        parsed = yaml.safe_load(content)
        return parsed if parsed is not None else ""

    # Everything else: return raw text
    return content


def stringify(value: Any) -> str:
    """Convert any loaded value back to a string for prompt injection."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2)
    return str(value)
