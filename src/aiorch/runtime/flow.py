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

"""Flow primitive runtime — calls a sub-pipeline and returns its output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aiorch.core.parser import Step, parse_file
from aiorch.core import template
from aiorch.core.dag import execute as dag_execute
from aiorch.core.utils import resolve_input as _resolve_input_shared


async def execute_flow(step: Step, context: dict[str, Any]) -> Any:
    """Execute a flow step by loading and running a sub-pipeline.

    Resolves the flow path (with template variables), parses the
    sub-pipeline, builds a child context from step.vars and step.input,
    then executes the sub-pipeline via the DAG executor.

    Args:
        step: The flow step to execute. ``step.flow`` must be set.
        context: The parent pipeline context.

    Returns:
        The full child context dict if the sub-pipeline produces multiple
        outputs, or the single output value if there is exactly one.

    Raises:
        FileNotFoundError: If the resolved flow path does not exist.
    """
    from aiorch.constants import LOGGER_KEY
    run_logger = context.get(LOGGER_KEY)

    raw_path = template.resolve(step.flow, context)

    from aiorch.constants import SOURCE_DIR_KEY
    from aiorch.core.paths import safe_path

    source_dir = Path(context.get(SOURCE_DIR_KEY, "."))
    resolved_path = safe_path(
        raw_path,
        purpose="flow: sub-pipeline",
        default_root=source_dir,
        allow_symbolic=False,
    )

    sub_af = parse_file(resolved_path)

    child_context: dict[str, Any] = {}
    if step.vars:
        child_context.update(template.resolve_dict(step.vars, context))
    if step.input is not None:
        resolved_input = _resolve_input_shared(step.input, context) or ""
        child_context["input"] = resolved_input

    if run_logger:
        run_logger.log(step.name, "DEBUG", "Sub-pipeline starting", {
            "flow": str(resolved_path),
            "child_steps": len(sub_af.steps),
            "child_context_keys": sorted(k for k in child_context if not k.startswith("__")),
        })

    # Lazy import to avoid circular dependency:
    from aiorch.runtime import execute_step

    result_context = await dag_execute(sub_af, runner=execute_step, context=child_context)

    outputs = {
        s.output: result_context[s.output]
        for s in sub_af.steps.values()
        if s.output and s.output in result_context
    }

    if run_logger:
        run_logger.log(step.name, "DEBUG", "Sub-pipeline finished", {
            "outputs_produced": list(outputs.keys()),
        })

    if len(outputs) == 1:
        return next(iter(outputs.values()))

    return result_context


def _resolve_input_for_flow(
    input_val: str | list[str] | None, context: dict[str, Any]
) -> str:
    """Resolve step.input into a single string for the child context.

    Delegates to shared utility. Kept for backward compatibility.
    """
    return _resolve_input_shared(input_val, context) or ""
