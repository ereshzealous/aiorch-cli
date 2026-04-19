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

"""Condition evaluation for step execution."""

from __future__ import annotations


def eval_condition(condition: str) -> bool:
    """Evaluate a simple condition string.

    Supported forms:
      - "true", "yes", "1" → True
      - "false", "no", "0", "none", "" → False
      - "x == y" → equality check (strips quotes)
      - "x != y" → inequality check (strips quotes)
      - Anything else → bool(condition)
    """
    condition = condition.strip()
    if condition.lower() in ("true", "yes", "1"):
        return True
    if condition.lower() in ("false", "no", "0", "none", ""):
        return False
    if "==" in condition:
        left, right = condition.split("==", 1)
        return left.strip().strip("'\"") == right.strip().strip("'\"")
    if "!=" in condition:
        left, right = condition.split("!=", 1)
        return left.strip().strip("'\"") != right.strip().strip("'\"")
    return bool(condition)
