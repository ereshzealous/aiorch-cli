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

"""Primitive and action registries — extensible dispatch without if/elif chains.

Usage:
    # Register a new primitive type
    from aiorch.runtime.registry import register_primitive

    register_primitive("transform", handler=my_handler, cost_estimator=my_estimator)

    # Register a new action
    from aiorch.runtime.registry import register_action

    register_action("jira", handler=my_jira_handler)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Awaitable

from aiorch.core.parser import Step


# --- Type aliases ---

StepHandler = Callable[[Step, dict[str, Any]], Awaitable[Any]]
CostEstimator = Callable[[Step], float]
ActionHandler = Callable[[Step, dict[str, Any]], Awaitable[Any]]


# ---------------------------------------------------------------------------
# Primitive registry
# ---------------------------------------------------------------------------

@dataclass
class PrimitiveSpec:
    """Describes a registered primitive type."""
    name: str
    handler: StepHandler
    cost_estimator: CostEstimator | None = None


_PRIMITIVES: dict[str, PrimitiveSpec] = {}


def register_primitive(
    name: str,
    handler: StepHandler,
    cost_estimator: CostEstimator | None = None,
) -> None:
    """Register a primitive type.

    Args:
        name: Primitive name (e.g., "run", "prompt", "transform").
        handler: Async function ``(step, context) -> Any``.
        cost_estimator: Optional function ``(step) -> float`` for pre-run cost estimation.
    """
    _PRIMITIVES[name] = PrimitiveSpec(
        name=name,
        handler=handler,
        cost_estimator=cost_estimator,
    )


def get_primitive(name: str) -> PrimitiveSpec | None:
    """Look up a registered primitive by name."""
    return _PRIMITIVES.get(name)


def get_registered_primitives() -> list[str]:
    """Return all registered primitive names."""
    return list(_PRIMITIVES.keys())


# ---------------------------------------------------------------------------
# Action registry
# ---------------------------------------------------------------------------

_ACTIONS: dict[str, ActionHandler] = {}


def register_action(name: str, handler: ActionHandler) -> None:
    """Register an action handler.

    Args:
        name: Action name (e.g., "slack", "jira", "pagerduty").
        handler: Async function ``(step, context) -> Any``.
    """
    _ACTIONS[name] = handler


def get_action(name: str) -> ActionHandler | None:
    """Look up a registered action by name."""
    return _ACTIONS.get(name)


def get_registered_actions() -> list[str]:
    """Return all registered action names."""
    return list(_ACTIONS.keys())


# ---------------------------------------------------------------------------
# Built-in registrations (called once at import time)
# ---------------------------------------------------------------------------

def _register_builtins() -> None:
    """Register all built-in primitives and actions.

    Uses lazy imports so modules are only loaded when actually dispatched.
    """
    # --- Primitives ---

    async def _run_handler(step: Step, context: dict[str, Any]) -> Any:
        from aiorch.core import template
        from aiorch.core.utils import parse_duration
        from aiorch.runtime.run import execute_run
        from aiorch.runtime.run_env import merge_env
        from aiorch.constants import LOGGER_KEY
        logger = context.get(LOGGER_KEY)
        try:
            # Shell-safe resolver: every {{ expr }} is passed through
            # shlex.quote so attacker-controlled values (webhook body,
            # LLM output, DB rows) cannot inject shell metacharacters.
            # See aiorch.core.template for the rules — notably, Jinja
            # expressions must appear in bare position, not wrapped
            # in user-written shell quotes.
            cmd = template.resolve_for_shell(step.run, context)
        except template.ShellTemplateError:
            # ShellTemplateError carries its own educational message
            # (line/column + fix hint). Re-raise unchanged so the
            # user sees the pointer, not a generic "failed to resolve".
            raise
        except Exception as e:
            raise RuntimeError(
                f"Step '{step.name}': failed to resolve command template\n"
                f"Template: {step.run[:200]}\n"
                f"Available variables: {[k for k in context if not k.startswith('__')]}\n"
                f"Error: {e}"
            )

        if logger: logger.log(step.name, "DEBUG", "Shell command resolved", {"command": cmd[:1000]})

        # Policy check: shell command allowlist/blocklist
        from aiorch.core.policy import check_shell_command
        from aiorch.constants import CONFIG_KEY
        cfg = context.get(CONFIG_KEY)
        if cfg and getattr(cfg, "policy", None):
            from aiorch.core.policy import ShellPolicyConfig
            policy_data = cfg.policy
            if isinstance(policy_data, dict):
                shell_data = policy_data.get("shell", {})
                shell_policy = ShellPolicyConfig(**shell_data) if isinstance(shell_data, dict) else ShellPolicyConfig()
            elif hasattr(policy_data, "shell"):
                shell_policy = policy_data.shell
            else:
                shell_policy = ShellPolicyConfig()
            check_shell_command(cmd, shell_policy)

        timeout = parse_duration(step.timeout)
        # Subprocess env = os.environ + configs + only the secrets named
        # in `step.secrets` (allowlist — unnamed secrets are stripped).
        secrets_allowed = set(step.secrets or [])
        stdout = await execute_run(
            cmd,
            timeout=timeout,
            env=merge_env(context, secrets_allowed=secrets_allowed),
        )
        if logger:
            logger.log(step.name, "DEBUG", "Shell command completed", {
                "stdout_bytes": len(stdout),
                "stdout_preview": stdout[:600],
            })

        # Honor format: json on `run:` steps. Without this, every shell
        # step's output is a raw string in the template context, so
        # `{{report.is_healthy}}` fails with "'str object' has no
        # attribute 'is_healthy'" even when the script printed valid JSON.
        # LLM steps already get this via the executor's output_format
        # plumbing — this aligns shell steps with that contract.
        if step.format and step.format.value == "json":
            import json as _json
            try:
                return _json.loads(stdout)
            except (ValueError, TypeError) as exc:
                raise RuntimeError(
                    f"Step '{step.name}': format: json was set but stdout is not "
                    f"valid JSON. Error: {exc}\n"
                    f"First 200 chars of stdout: {stdout[:200]}"
                )
        return stdout

    async def _prompt_handler(step: Step, context: dict[str, Any]) -> Any:
        from aiorch.runtime import _dispatch_prompt
        return await _dispatch_prompt(step, context, None)

    async def _flow_handler(step: Step, context: dict[str, Any]) -> Any:
        from aiorch.runtime.flow import execute_flow
        return await execute_flow(step, context)

    async def _python_handler(step: Step, context: dict[str, Any]) -> Any:
        from aiorch.runtime.python import python_handler
        return await python_handler(step, context)

    def _run_cost(step: Step) -> float:
        return 0.0

    def _flow_cost(step: Step) -> float:
        return 0.0

    register_primitive("run", _run_handler, cost_estimator=_run_cost)
    register_primitive("prompt", _prompt_handler)
    register_primitive("flow", _flow_handler, cost_estimator=_flow_cost)
    register_primitive("python", _python_handler)
    # CLI primitives are: prompt, python, run, flow, foreach, condition.
    # `agent`, `action`, and connector primitives (email/s3/kafka/teams/
    # discord) are Platform-only. Pipelines using them fail at
    # DAG-build with "Unknown primitive".


_register_builtins()
