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

"""DAG resolution and async parallel execution of steps."""

from __future__ import annotations

import asyncio
from graphlib import CycleError, TopologicalSorter
from typing import Any, Callable, Awaitable

from aiorch.core.parser import Agentfile, Step
from aiorch.constants import COST_KEY, RUNTIME_META_KEY


class DAGError(Exception):
    pass


def build_graph(af: Agentfile) -> dict[str, set[str]]:
    """Build a dependency graph from a pipeline.

    Returns a dict mapping step_name -> set of step names it depends on.
    """
    graph: dict[str, set[str]] = {}
    step_names = set(af.steps.keys())

    for name, step in af.steps.items():
        deps: set[str] = set()

        # Explicit depends
        for dep in step.depends:
            if dep not in step_names:
                raise DAGError(f"Step '{name}' depends on unknown step '{dep}'")
            deps.add(dep)

        # `parallel: step_a` means copy step_a's dependencies to this step
        # so they run in the same layer
        if isinstance(step.parallel, str) and step.parallel in step_names:
            # Copy the deps from the referenced step
            ref_step = af.steps[step.parallel]
            for dep in ref_step.depends:
                if dep in step_names:
                    deps.add(dep)

        graph[name] = deps

    return graph


def get_execution_order(graph: dict[str, set[str]]) -> list[list[str]]:
    """Return steps grouped into layers that can run in parallel.

    Each layer is a list of step names. All steps in a layer can run
    concurrently. Layers must execute sequentially.
    """
    sorter = TopologicalSorter(graph)
    try:
        sorter.prepare()
    except CycleError as e:
        raise DAGError(f"Cycle detected in step dependencies: {e}")

    layers: list[list[str]] = []
    while sorter.is_active():
        ready = list(sorter.get_ready())
        layers.append(ready)
        for node in ready:
            sorter.done(node)

    return layers


def get_steps_from(graph: dict[str, set[str]], start: str) -> set[str]:
    """Return `start` and all its downstream dependents (steps that depend on it, transitively)."""
    if start not in graph:
        raise DAGError(f"Step '{start}' not found in graph")

    # Build a reverse adjacency map: step -> set of steps that depend on it
    reverse: dict[str, set[str]] = {name: set() for name in graph}
    for name, deps in graph.items():
        for dep in deps:
            reverse[dep].add(name)

    # BFS from start through the reverse (downstream) edges
    result: set[str] = set()
    queue = [start]
    while queue:
        current = queue.pop()
        if current in result:
            continue
        result.add(current)
        for downstream in reverse[current]:
            if downstream not in result:
                queue.append(downstream)

    return result


StepRunner = Callable[[Step, dict[str, Any]], Awaitable[Any]]


async def execute(
    af: Agentfile,
    runner: StepRunner,
    context: dict[str, Any] | None = None,
    on_step_start: Callable[[str], None] | None = None,
    on_step_done: Callable[[str, Any], None] | None = None,
    on_step_error: Callable[[str, Exception], None] | None = None,
    on_step_skipped: Callable[[str, str], None] | None = None,
    step_filter: str | None = None,
    from_step: str | None = None,
    checkpoint: Callable[[str, Any], None] | None = None,
    restored_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute all steps in a pipeline respecting dependencies.

    Args:
        af: Parsed Aiorch.
        runner: Async function that executes a single step.
        context: Initial context (env vars, inputs). Mutated with step outputs.
        on_step_start: Callback when a step begins.
        on_step_done: Callback when a step completes.
        on_step_error: Callback when a step fails.
        step_filter: If set, only run this single step.
        from_step: If set, skip all steps before this one and run from here downstream.
        checkpoint: Callback to persist step output after success.
        restored_outputs: Previously checkpointed outputs to restore (skips re-execution).

    Returns:
        Final context dict with all step outputs.
    """
    from aiorch.inputs import load_input, LazyHttpInput

    ctx = {}

    # Inject env vars first
    ctx.update(af.env)

    # Pre-seed private scope keys (__org_id__, __workspace_id__, __run_id__,
    # __user__, and the runtime meta dict) so that type: connector inputs
    # and other scope-aware loaders can resolve before the normal input
    # loop runs. Connector lookup walks org → workspace → pipeline, so
    # the scope must be populated before we iterate af.input.
    from aiorch.constants import RUNTIME_META_KEY
    if context:
        for _k in (
            "__org_id__", "__workspace_id__", "__run_id__",
            "__user__", RUNTIME_META_KEY,
        ):
            if _k in context:
                ctx[_k] = context[_k]

    # Inject top-level input defaults (typed input loading).
    # Skip defaults for keys the caller overrode (context) so we don't
    # eagerly load a declared fallback file that doesn't exist on disk —
    # e.g., `type: file, path: document.txt` as the declared default
    # when the user supplies a different path at run time. If the user
    # did NOT override and the default file is missing, set the key to
    # None rather than crashing the whole pipeline at context build time;
    # downstream steps will fail with a clearer error if they actually
    # need the value.
    overridden_keys = set(context.keys()) if context else set()
    if af.input:
        for k, v in af.input.items():
            if k in overridden_keys:
                continue

            # http inputs are resolved lazily — wrap in LazyHttpInput
            # so the fetch only fires if a step actually references
            # the value. Dead http inputs never touch the network.
            if isinstance(v, dict) and v.get("type") == "http":
                ctx[k] = LazyHttpInput(v)
                continue

            # Connector inputs (Postgres / S3 / Kafka / SMTP / webhook)
            # are a commercial aiorch Platform feature — they need the
            # connector registry, workspace secrets, and audit logging
            # that only Platform provides.
            if isinstance(v, dict) and v.get("type") == "connector":
                raise RuntimeError(
                    f"Input '{k}' declares type: connector — connectors are "
                    "available only in the commercial aiorch Platform. "
                    "The CLI supports: text/string, integer, number, boolean, "
                    "list, file (local disk), http (GET), env, and stdin."
                )

            try:
                ctx[k] = load_input(v)
            except FileNotFoundError:
                ctx[k] = None

    # CLI / API overrides — use schema type info to load correctly.
    # Scalars pass through unchanged; artifact/http/env wrap as typed
    # dicts so the loader receives the right shape even if the caller
    # supplied a bare string or an artifact_id.
    if context:
        input_schema = af.input or {}
        for k, v in context.items():
            schema_entry = input_schema.get(k)
            if not (schema_entry and isinstance(schema_entry, dict)):
                # No schema entry for this key — pass-through
                ctx[k] = load_input(v)
                continue

            declared_type = schema_entry.get("type", "string")

            # Scalars + pre-loaded file content (incl. non-string values
            # like int / bool / list / parsed dict from @json). `file`
            # carries already-parsed content that parse_kv_inputs handed
            # us, so no further loader is needed here.
            if declared_type in ("string", "text", "integer", "number", "boolean", "list", "file"):
                ctx[k] = v
                continue

            # Artifact: caller can supply either a typed dict
            # {type: artifact, artifact_id: "..."} OR just an
            # artifact_id string. Either way we build the loader
            # config from the schema's format field.
            #
            # IMPORTANT: the schema's `format:` is the source of
            # truth and OVERRIDES any format the caller supplied.
            # The CLI's parse_kv_inputs guesses format from file
            # extension when uploading via @file syntax, but those
            # guesses are inferences — the pipeline author's
            # declared format is the authoritative answer. Without
            # this override, a `format: binary` declaration was
            # silently ignored when the CLI uploaded a .png as
            # text (because parse_kv_inputs has no binary path),
            # and the run failed at the python step with a confusing
            # "expected bytes, got str" error.
            if declared_type == "artifact":
                # Artifact-typed inputs (platform-managed content store)
                # are a commercial aiorch Platform feature. CLI users
                # should pass local files via `type: file` or pipe
                # content via `type: stdin`.
                raise RuntimeError(
                    f"Input '{k}' declares type: artifact — artifact storage "
                    "is available only in the commercial aiorch Platform. "
                    "Use `type: file` with a local path in the CLI."
                )

            # Connector-typed inputs are Platform-only — same as the
            # untyped-connector branch above.
            if declared_type == "connector":
                raise RuntimeError(
                    f"Input '{k}' declares type: connector — connectors are "
                    "available only in the commercial aiorch Platform."
                )

            # http: lazy fetch via LazyHttpInput. Defer the network
            # call until a step actually references the value —
            # dead http inputs never touch the network, and pipeline
            # start isn't blocked on a slow remote API.
            if declared_type == "http":
                # Merge runtime overrides (url, headers) with the
                # schema declaration (method, body, format defaults).
                cfg: dict = dict(schema_entry)
                cfg["type"] = "http"
                if isinstance(v, dict) and v.get("type") == "http":
                    cfg.update({kk: vv for kk, vv in v.items() if kk != "type"})
                elif isinstance(v, str):
                    cfg["url"] = v
                ctx[k] = LazyHttpInput(cfg)
                continue

            # Unknown type — fall through to generic loader
            ctx[k] = load_input(v)

    # Propagate source directory so flow steps resolve relative paths correctly
    from aiorch.constants import SOURCE_DIR_KEY
    if af.source_path:
        ctx[SOURCE_DIR_KEY] = str(af.source_path.parent)

    # Pre-inject restored outputs into context so downstream steps can reference them
    if restored_outputs:
        for step_name, output in restored_outputs.items():
            step = af.steps.get(step_name)
            if step and step.output:
                ctx[step.output] = output

    if step_filter:
        # Run a single step
        if step_filter not in af.steps:
            raise DAGError(f"Step '{step_filter}' not found")
        step = af.steps[step_filter]
        if on_step_start:
            on_step_start(step_filter)
        try:
            result = await runner(step, ctx)
            if step.output:
                ctx[step.output] = result
            if on_step_done:
                on_step_done(step_filter, result)
        except Exception as e:
            if on_step_error:
                on_step_error(step_filter, e)
            raise
        return ctx

    # Full DAG execution
    graph = build_graph(af)
    layers = get_execution_order(graph)

    # Compute the set of steps to run when --from is used
    from_steps: set[str] | None = None
    if from_step:
        if from_step not in af.steps:
            raise DAGError(f"Step '{from_step}' not found")
        from_steps = get_steps_from(graph, from_step)

    for layer in layers:
        # Filter out steps whose condition is not met
        runnable = []
        for name in layer:
            step = af.steps[name]
            # Skip on_failure trigger steps during normal execution
            if step.trigger == "on_failure":
                continue
            # Skip steps not in the from_step downstream set
            if from_steps is not None and name not in from_steps:
                continue
            runnable.append(name)

        # Run all steps in this layer concurrently
        async def _run_step(name: str) -> None:
            step = af.steps[name]

            # Restore from checkpoint — skip execution, inject output
            if restored_outputs and name in restored_outputs:
                restored = restored_outputs[name]
                if step.output:
                    ctx[step.output] = restored
                if on_step_done:
                    on_step_done(name, restored)
                return

            if on_step_start:
                on_step_start(name)

            attempts = max(1, step.retry + 1) if step.retry else 1
            last_error: Exception | None = None

            from aiorch.constants import LOGGER_KEY
            _dbg_logger = ctx.get(LOGGER_KEY) if isinstance(ctx, dict) else None

            for attempt in range(attempts):
                if _dbg_logger and attempts > 1:
                    _dbg_logger.log(name, "DEBUG", f"Attempt {attempt + 1}/{attempts}")
                try:
                    result = await runner(step, ctx)
                    if step.output:
                        ctx[step.output] = result
                    # on_done first (creates storage row), then checkpoint (updates it)
                    if on_step_done:
                        on_step_done(name, result)
                    if checkpoint:
                        checkpoint(name, result)
                    return
                except Exception as e:
                    # StepSkipped is control flow, not an error — the
                    # step's condition evaluated false. Emit a distinct
                    # skip event and return cleanly. Don't retry, don't
                    # run on_failure, don't count against attempts.
                    # Import locally to avoid a circular import.
                    from aiorch.runtime import StepSkipped
                    if isinstance(e, StepSkipped):
                        if on_step_skipped:
                            on_step_skipped(name, e.resolved)
                        return
                    last_error = e
                    if _dbg_logger and attempt < attempts - 1:
                        _dbg_logger.log(
                            name, "WARN",
                            f"Attempt {attempt + 1} failed, retrying: {type(e).__name__}: {str(e)[:200]}",
                        )
                    if attempt < attempts - 1:
                        delay = _parse_retry_delay(step.retry_delay)
                        await asyncio.sleep(delay)

            # All retries exhausted
            if on_step_error:
                on_step_error(name, last_error)

            # Run on_failure handler if configured
            if step.on_failure and step.on_failure in af.steps:
                failure_step = af.steps[step.on_failure]
                await runner(failure_step, ctx)

            raise last_error

        tasks = [_run_step(name) for name in runnable]
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # All tasks ran (successful ones are checkpointed).
            # Re-raise the first exception so the pipeline fails correctly.
            for r in results:
                if isinstance(r, Exception):
                    raise r

    # Update _meta.total_cost so templates in later steps/outputs can reference it
    costs = ctx.get(COST_KEY, {})
    if isinstance(costs, dict) and costs:
        meta = ctx.get(RUNTIME_META_KEY)
        if isinstance(meta, dict):
            meta["total_cost"] = round(sum(costs.values()), 6)

    return ctx


def collect_outputs(af, ctx: dict) -> dict:
    """Collect declared pipeline outputs from the execution context.

    If af.outputs is defined, resolve the referenced variables.
    Otherwise, collect all step output variables.
    """
    if af.outputs:
        result = {}
        for key, ref in af.outputs.items():
            if isinstance(ref, str) and ref in ctx:
                result[key] = ctx[ref]
            elif isinstance(ref, list):
                result[key] = [ctx.get(r) for r in ref if isinstance(r, str)]
            else:
                result[key] = ref
        return result

    # No outputs declared — return all step output variables
    step_outputs = {}
    for step_name, step in af.steps.items():
        if step.output and step.output in ctx:
            step_outputs[step.output] = ctx[step.output]
    return step_outputs


def _parse_retry_delay(delay_str: str | None) -> float:
    """Parse retry_delay string like '5s', '1m', '500ms'. Default: 1 second."""
    from aiorch.core.utils import parse_duration
    return parse_duration(delay_str, default=1.0)
