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

"""Post-step output processing — unwrap results, accumulate cost, save to file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aiorch.constants import COST_KEY, META_KEY
from aiorch.core import template
from aiorch.core.loader import stringify
from aiorch.runtime.prompt import PromptResult


def unwrap_result(
    result: Any, context: dict[str, Any], step_name: str, max_cost: float | None = None,
) -> Any:
    """Unwrap PromptResult, accumulating cost and metadata in context.

    If result is a PromptResult, extracts cost/token info into the
    shared accumulators and returns the content. Otherwise returns as-is.
    Checks per-step and per-run cost limits if configured.
    """
    if isinstance(result, PromptResult):
        if COST_KEY not in context:
            context[COST_KEY] = {}
        context[COST_KEY][step_name] = result.cost

        if META_KEY not in context:
            context[META_KEY] = {}
        from aiorch.runtime.llm import PROVIDER_NAME_KEY
        context[META_KEY][step_name] = {
            "model": result.model,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "cost": result.cost,
            "provider_name": context.get(PROVIDER_NAME_KEY, ""),
            "model_source": context.pop("__model_source__", "explicit"),
        }

        # Policy: check per-step cost limit
        from aiorch.core.policy import check_step_cost, check_run_cost
        from aiorch.constants import CONFIG_KEY

        cfg = context.get(CONFIG_KEY)
        global_max_step = None
        global_max_run = None
        if cfg:
            policy = getattr(cfg, "policy", None)
            if policy:
                if isinstance(policy, dict):
                    global_max_step = policy.get("max_cost_per_step")
                    global_max_run = policy.get("max_cost_per_run")
                else:
                    global_max_step = getattr(policy, "max_cost_per_step", None)
                    global_max_run = getattr(policy, "max_cost_per_run", None)

        effective_max = max_cost if max_cost is not None else global_max_step
        check_step_cost(step_name, result.cost, effective_max)

        # Check total run cost
        total = sum(context[COST_KEY].values())
        check_run_cost(total, global_max_run)

        return result.content
    return result


def save_to_file(result: Any, save_path_template: str, context: dict[str, Any]) -> None:
    """Save step output to a file if configured.

    The rendered path is confined to an allowed root via
    ``core.paths.safe_path`` so a template-injected value can't be
    used to write outside the executor's sandbox. Default root is
    process CWD; operators add more via ``AIORCH_SAFE_ROOTS``.
    ``/dev/stdout`` / ``/dev/stderr`` / ``/dev/null`` are accepted
    as symbolic destinations for CLI pipelines that print to the
    terminal.

    Args:
        result: The step result to save.
        save_path_template: Jinja2 template for the file path.
        context: Template context for resolving the path.
    """
    from aiorch.core.paths import safe_path

    rendered = template.resolve(save_path_template, context)
    save_path = safe_path(
        rendered,
        purpose="save: target",
        default_root=Path.cwd(),
        allow_symbolic=True,
    )

    # Symbolic destinations have no parent dir to create.
    if str(save_path) not in ("/dev/stdout", "/dev/stderr", "/dev/null"):
        save_path.parent.mkdir(parents=True, exist_ok=True)

    content = result
    if isinstance(result, PromptResult):
        content = result.content
    save_path.write_text(stringify(content))
