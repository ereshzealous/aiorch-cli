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

"""Cron expression validation for aiorch schedule management.

The ``trigger:`` block was removed from the pipeline YAML spec in
2026-04-13 — it was non-functional, and schedules are now managed
exclusively via the ``schedules`` table (UI / POST /api/schedules).
This module retains only ``validate_cron`` for use by the schedule
creation route.
"""

from __future__ import annotations

import re

# Pattern for a single cron field: *, a number, a range (e.g. 1-5), a list
# (e.g. 1,3,5), or step values (e.g. */2).
_CRON_FIELD_RE = re.compile(
    r"^(\*(/\d+)?|\d+(-\d+)?(/\d+)?(,\d+(-\d+)?(/\d+)?)*)$"
)


def validate_cron(expr: str) -> bool:
    """Return ``True`` if *expr* looks like a valid 5-field cron expression.

    This performs a structural check only (five whitespace-separated fields
    where each field is ``*``, a number, a range, a list, or a step value).
    It does **not** validate value ranges (e.g. minute 0-59).
    """
    fields = expr.strip().split()
    if len(fields) != 5:
        return False
    return all(_CRON_FIELD_RE.match(f) for f in fields)
