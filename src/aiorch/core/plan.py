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

"""Execution plan builder — deterministic pre-run analysis without LLM calls.

Generates a complete execution plan showing DAG layers, step types, conditions,
cache settings, estimated costs, and input sources. Used by `aiorch plan`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aiorch.core.dag import build_graph, get_execution_order
from aiorch.core.parser import Agentfile, Step


@dataclass
class StepPlan:
    """Plan for a single step — what will happen when it runs."""
    name: str
    primitive_type: str
    layer: int
    depends: list[str]
    condition: str | None = None
    cache: bool = False
    model: str | None = None
    estimated_cost: float = 0.0
    input_source: str | None = None
    foreach: str | None = None
    output_var: str | None = None
    has_validation: bool = False


@dataclass
class ExecutionPlan:
    """Complete execution plan for a pipeline."""
    pipeline_name: str
    total_steps: int
    total_layers: int
    estimated_total_cost: float
    steps: list[StepPlan] = field(default_factory=list)
    layers: list[list[str]] = field(default_factory=list)


def build_plan(af: Agentfile) -> ExecutionPlan:
    """Build a complete execution plan from a parsed Agentfile.

    Pure function — no side effects, no LLM calls, no cost.
    Uses the DAG builder for layer computation and cost estimator for pricing.
    """
    graph = build_graph(af)
    layers = get_execution_order(graph)

    # Build step-to-layer mapping
    step_layer: dict[str, int] = {}
    for layer_idx, layer_names in enumerate(layers):
        for name in layer_names:
            step_layer[name] = layer_idx

    # Estimate costs
    step_costs: dict[str, float] = {}
    try:
        from aiorch.core.cost import estimate_pipeline_cost
        for name, cost in estimate_pipeline_cost(af):
            step_costs[name] = cost
    except Exception:
        pass

    # Build step plans
    step_plans: list[StepPlan] = []
    for name, step in af.steps.items():
        input_source = _describe_input(step)
        foreach_desc = _describe_foreach(step)

        step_plans.append(StepPlan(
            name=name,
            primitive_type=step.primitive_type,
            layer=step_layer.get(name, 0),
            depends=step.depends,
            condition=step.condition,
            cache=step.cache,
            model=step.model,
            estimated_cost=step_costs.get(name, 0.0),
            input_source=input_source,
            foreach=foreach_desc,
            output_var=step.output,
            has_validation=bool(step.output_schema or step.assertions),
        ))

    # Sort by layer for display
    step_plans.sort(key=lambda s: (s.layer, s.name))

    total_cost = sum(s.estimated_cost for s in step_plans)

    return ExecutionPlan(
        pipeline_name=af.name,
        total_steps=len(step_plans),
        total_layers=len(layers),
        estimated_total_cost=total_cost,
        steps=step_plans,
        layers=[list(layer) for layer in layers],
    )


def _describe_input(step: Step) -> str | None:
    """Describe where a step gets its input."""
    if step.input is None:
        return None
    if isinstance(step.input, list):
        return " + ".join(str(i)[:50] for i in step.input)
    return str(step.input)[:80]


def _describe_foreach(step: Step) -> str | None:
    """Describe a step's foreach configuration."""
    if step.foreach is None:
        return None
    if isinstance(step.foreach, list):
        return f"[{len(step.foreach)} items]"
    return str(step.foreach)[:50]
