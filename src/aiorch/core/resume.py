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

"""Reusable resume logic — shared between CLI and server."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from aiorch.core.parser import parse_file, Agentfile
from aiorch.core.dag import execute
from aiorch.runtime import execute_step, COST_KEY, META_KEY, LOGGER_KEY, CONFIG_KEY
from aiorch.storage import (
    get_run,
    get_run_steps,
    get_step_outputs,
    start_run,
    save_step_output,
)


async def resume_run(
    run_id: int,
    *,
    new_run_id: int | None = None,
    af: Agentfile | None = None,
    on_step_start: Callable[[str], None] | None = None,
    on_step_done: Callable[[str, Any], None] | None = None,
    on_step_error: Callable[[str, Exception], None] | None = None,
    extra_context: dict[str, Any] | None = None,
) -> int:
    """Resume a failed run from its last checkpoint.

    Args:
        run_id: The ID of the failed run to resume.
        new_run_id: Pre-created run ID (server mode). If None, creates one (CLI mode).
        af: Pre-parsed pipeline (server mode). If None, loads from filesystem (CLI mode).
        on_step_start: Optional callback when a step begins.
        on_step_done: Optional callback when a step completes.
        on_step_error: Optional callback when a step fails.
        extra_context: Optional additional context (e.g., logger, config, provider keys).

    Returns:
        The run_id for the resumed run.
    """
    # 1. Load run record
    run_record = get_run(run_id)
    if not run_record:
        raise ValueError(f"Run #{run_id} not found")
    if run_record["status"] == "success":
        raise ValueError(f"Run #{run_id} already succeeded. Nothing to resume.")

    # 2. Load pipeline — from pre-parsed (server) or filesystem (CLI)
    if af is None:
        pipeline_file = run_record.get("file")
        if not pipeline_file:
            raise ValueError(f"Run #{run_id} has no pipeline file recorded")
        pipeline_path = Path(pipeline_file)
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_file}")
        af = parse_file(pipeline_path)

    # 3. Load successful step outputs as restored context
    restored = get_step_outputs(run_id)

    # 4. Identify failed steps
    steps = get_run_steps(run_id)
    failed_steps = [s["step_name"] for s in steps if s["status"] == "failed"]

    if not failed_steps and not restored:
        raise ValueError(f"Run #{run_id} has no checkpointed data to resume from")

    resume_from = failed_steps[0] if failed_steps else None

    # 5. Create run record only if not pre-created (CLI mode)
    if new_run_id is None:
        new_run_id = start_run(af.name, run_record.get("file", ""))

    # 6. Build context
    context: dict[str, Any] = {}
    shared_meta: dict[str, dict] = {}
    shared_costs: dict[str, float] = {}
    context[META_KEY] = shared_meta
    context[COST_KEY] = shared_costs

    if extra_context:
        context.update(extra_context)

    # 7. Checkpoint callback
    def _checkpoint(name: str, result: Any) -> None:
        try:
            output_json = json.dumps(result, default=str)
            save_step_output(new_run_id, name, output_json)
        except Exception:
            pass

    # 8. Execute with resume
    await execute(
        af,
        runner=execute_step,
        context=context,
        on_step_start=on_step_start,
        on_step_done=on_step_done,
        on_step_error=on_step_error,
        from_step=resume_from,
        checkpoint=_checkpoint,
        restored_outputs=restored,
    )

    return new_run_id
