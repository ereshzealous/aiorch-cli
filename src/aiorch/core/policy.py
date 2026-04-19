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

"""Policy engine — safety controls for shell commands and cost budgets.

Provides shell command allowlisting/blocklisting and per-step/per-run cost caps.
All defaults are permissive (unrestricted) to avoid breaking existing pipelines.
"""

from __future__ import annotations

import shlex
from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PolicyViolationError(Exception):
    """Raised when a step violates a policy constraint."""
    pass


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------

class ShellPolicyMode(str, Enum):
    unrestricted = "unrestricted"
    allowlist = "allowlist"
    blocklist = "blocklist"


class ShellPolicyConfig(BaseModel):
    """Shell command execution policy."""
    mode: ShellPolicyMode = ShellPolicyMode.unrestricted
    commands: list[str] = Field(default_factory=list)


class PolicyConfig(BaseModel):
    """Top-level policy configuration."""
    shell: ShellPolicyConfig = Field(default_factory=ShellPolicyConfig)
    max_cost_per_step: float | None = None
    max_cost_per_run: float | None = None


# ---------------------------------------------------------------------------
# Shell command validation
# ---------------------------------------------------------------------------

def _extract_base_command(command: str) -> str:
    """Extract the base command name from a shell command string.

    Handles: paths (/usr/bin/git → git), env prefixes (VAR=x cmd → cmd),
    and takes only the first command in pipes.
    """
    # Take only the first command in a pipe chain
    first_cmd = command.split("|")[0].strip()

    # Try to split with shlex for proper parsing
    try:
        parts = shlex.split(first_cmd)
    except ValueError:
        # Fallback: simple whitespace split
        parts = first_cmd.split()

    if not parts:
        return ""

    # Skip env variable assignments (VAR=value cmd ...)
    for part in parts:
        if "=" in part and not part.startswith("-"):
            continue
        # Extract basename from path
        cmd = part.rsplit("/", 1)[-1]
        return cmd

    return parts[-1].rsplit("/", 1)[-1] if parts else ""


import re as _re

# Regex to split on shell compound operators: ;  &&  ||  |  $()  backticks
_COMPOUND_SPLIT = _re.compile(r'\s*(?:;|&&|\|\||`[^`]*`|\$\([^)]*\))\s*|\s*\|\s*')


def _extract_all_commands(command: str) -> list[str]:
    """Extract base command names from all parts of a compound shell command."""
    parts = _COMPOUND_SPLIT.split(command)
    commands = []
    for part in parts:
        part = part.strip()
        if part:
            base = _extract_base_command(part)
            if base:
                commands.append(base)
    return commands


def check_shell_command(command: str, policy: ShellPolicyConfig) -> None:
    """Check if a shell command is allowed by the policy.

    Raises PolicyViolationError if the command is denied.
    Does nothing if mode is unrestricted.
    Checks ALL commands in compound expressions (;, &&, ||, pipes, $()).
    """
    if policy.mode == ShellPolicyMode.unrestricted:
        return

    bases = _extract_all_commands(command)
    if not bases:
        bases = [_extract_base_command(command)]

    for base in bases:
        if policy.mode == ShellPolicyMode.allowlist:
            if base not in policy.commands:
                raise PolicyViolationError(
                    f"Command '{base}' is not in the shell allowlist. "
                    f"Allowed: {', '.join(policy.commands)}. "
                    f"Full command: {command[:200]}"
                )

        elif policy.mode == ShellPolicyMode.blocklist:
            if base in policy.commands:
                raise PolicyViolationError(
                    f"Command '{base}' is blocked by policy. "
                    f"Blocked: {', '.join(policy.commands)}. "
                    f"Full command: {command[:200]}"
                )


# ---------------------------------------------------------------------------
# Cost validation
# ---------------------------------------------------------------------------

def check_step_cost(step_name: str, cost: float, max_cost: float | None) -> None:
    """Check if a step's cost exceeds the maximum.

    Raises PolicyViolationError if exceeded. Does nothing if max_cost is None.
    """
    if max_cost is not None and cost > max_cost:
        raise PolicyViolationError(
            f"Step '{step_name}' cost ${cost:.4f} exceeds limit ${max_cost:.4f}"
        )


def check_run_cost(total_cost: float, max_cost: float | None) -> None:
    """Check if the total run cost exceeds the maximum.

    Raises PolicyViolationError if exceeded. Does nothing if max_cost is None.
    """
    if max_cost is not None and total_cost > max_cost:
        raise PolicyViolationError(
            f"Run total cost ${total_cost:.4f} exceeds limit ${max_cost:.4f}"
        )
