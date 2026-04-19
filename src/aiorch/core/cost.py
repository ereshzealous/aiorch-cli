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

"""Cost estimation module for aiorch pipelines.

Estimates the cost of running a pipeline BEFORE executing it, based on
step types, prompt lengths, and model pricing.
"""

from __future__ import annotations

from aiorch.core.parser import Agentfile, Step

# Default pricing per 1K tokens.
_DEFAULT_INPUT_COST_PER_1K = 0.01
_DEFAULT_OUTPUT_COST_PER_1K = 0.03

# Assumed number of output tokens for a single prompt step.
_ASSUMED_OUTPUT_TOKENS = 500


def _get_model_pricing(model: str | None) -> tuple[float, float]:
    """Return (input_cost_per_1k, output_cost_per_1k) for *model*.

    Uses the built-in MODEL_PRICING table from runtime.llm.
    """
    if not model:
        return _DEFAULT_INPUT_COST_PER_1K, _DEFAULT_OUTPUT_COST_PER_1K

    try:
        from aiorch.runtime.llm import MODEL_PRICING

        if model in MODEL_PRICING:
            input_per_m, output_per_m = MODEL_PRICING[model]
            return input_per_m / 1000, output_per_m / 1000

        # Try partial match
        for name, (ic, oc) in MODEL_PRICING.items():
            if model.startswith(name) or name.startswith(model):
                return ic / 1000, oc / 1000

    except Exception:  # noqa: BLE001
        pass

    return _DEFAULT_INPUT_COST_PER_1K, _DEFAULT_OUTPUT_COST_PER_1K


def _estimate_prompt_cost(step: Step) -> float:
    """Estimate the dollar cost of a single prompt step."""
    prompt_text = step.prompt or ""
    input_tokens = len(prompt_text) / 4
    output_tokens = _ASSUMED_OUTPUT_TOKENS

    input_cost_per_1k, output_cost_per_1k = _get_model_pricing(step.model)

    cost = (input_tokens / 1000) * input_cost_per_1k + (output_tokens / 1000) * output_cost_per_1k
    return cost


def estimate_step_cost(step: Step) -> tuple[str, float]:
    """Return ``(step_name, estimated_cost)`` for a single step.

    Uses the registry's cost_estimator if available, otherwise falls back
    to built-in heuristics for prompt and agent steps.
    """
    from aiorch.runtime.registry import get_primitive

    ptype = step.primitive_type
    spec = get_primitive(ptype)

    # Use registered cost estimator if available
    if spec and spec.cost_estimator:
        return step.name, spec.cost_estimator(step)

    # Built-in heuristics for LLM-based primitives
    if ptype == "prompt":
        return step.name, _estimate_prompt_cost(step)

    if ptype == "agent":
        prompt_text = ""
        if isinstance(step.agent, str):
            prompt_text = step.agent
        elif isinstance(step.agent, dict):
            prompt_text = step.agent.get("prompt", "")
        base_cost = _estimate_prompt_cost(
            Step(name=step.name, prompt=prompt_text, model=step.model),
        )
        return step.name, base_cost * step.max_iterations * 0.5

    # Unknown primitive — assume free
    return step.name, 0.0


def estimate_pipeline_cost(af: Agentfile) -> list[tuple[str, float]]:
    """Return a list of ``(step_name, estimated_cost)`` for every step."""
    return [estimate_step_cost(step) for step in af.steps.values()]


def format_cost_table(estimates: list[tuple[str, float]]) -> str:
    """Format cost estimates as a human-readable table string."""
    if not estimates:
        return "No steps to estimate."

    # Determine column widths.
    name_width = max(len(name) for name, _ in estimates)
    name_width = max(name_width, len("Step"))

    lines: list[str] = []
    header = f"{'Step':<{name_width}}  {'Est. Cost':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    total = 0.0
    for name, cost in estimates:
        lines.append(f"{name:<{name_width}}  ${cost:>9.4f}")
        total += cost

    lines.append("-" * len(header))
    lines.append(f"{'Total':<{name_width}}  ${total:>9.4f}")

    return "\n".join(lines)
