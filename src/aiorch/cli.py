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

"""CLI entry point — Click commands for aiorch."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
from pathlib import Path

import click

from aiorch import __version__
from aiorch.core.parser import parse_file
from aiorch.core.dag import build_graph, get_execution_order, execute
from aiorch.ui.display import (
    console,
    print_dag,
    print_step_list,
    print_validation_ok,
    print_validation_error,
    step_done,
    step_failed,
    print_run_summary,
    print_history,
    print_run_details,
    print_cost_breakdown,
    print_step_explanation,
    print_doctor_results,
    print_cost_estimate,
)
from aiorch.storage import start_run, get_runs, get_run, get_run_steps


AGENTFILE_NAMES = ["Aiorch", "Aiorch.yaml", "Aiorch.yml", "aiorch.yaml", "aiorch.yml"]


def _find_pipeline(path: str | None) -> Path:
    """Find the pipeline to use."""
    if path:
        p = Path(path)
        if not p.exists():
            raise click.ClickException(f"File not found: {path}")
        return p

    for name in AGENTFILE_NAMES:
        p = Path(name)
        if p.exists():
            return p

    raise click.ClickException(
        "No pipeline found. Create one with `aiorch init` or specify a path."
    )


@click.group()
@click.version_option(version=__version__, prog_name="aiorch")
def main():
    """Aiorch — Makefile for AI workflows."""
    pass


@main.command()
@click.argument("file", required=False)
@click.option("--step", help="Run a single step only")
@click.option("--from", "from_step", help="Resume from this step (skip earlier steps)")
@click.option("--dry", is_flag=True, help="Dry run — show plan without executing")
@click.option("--verbose", "-v", is_flag=True, help="Show step inputs and outputs")
@click.option("--input", "input_arg", help="Input file (YAML/JSON) or inline JSON string")
@click.option("-i", "--set", "input_overrides", multiple=True, metavar="KEY=VALUE",
              help="Set an input: -i topic=AI or -i data=@file.csv (repeatable)")
@click.option("--watch", is_flag=True, help="Re-run on file change")
@click.option("--format", "output_format", type=click.Choice(["text", "json"]), default="text",
              help="Output format (text or json)")
@click.option("--model", default=None, help="Override LLM model for all prompt/agent steps")
@click.option("--max-cost", default=None, type=float, help="Abort if cost exceeds this value in dollars")
def run(file: str | None, step: str | None, from_step: str | None, dry: bool, verbose: bool,
        input_arg: str | None, input_overrides: tuple, watch: bool,
        output_format: str, model: str | None, max_cost: float | None):
    """Run a pipeline."""
    from aiorch.runtime import execute_step, COST_KEY, META_KEY, LOGGER_KEY
    from aiorch.logging import RunLogger
    from aiorch.core.config import get_config
    from aiorch.logging.sinks import create_sink
    from aiorch.inputs import parse_input_arg, load_inputs
    from aiorch.storage import init_storage

    init_storage()  # MemoryStore fallback when no DATABASE_URL

    import json as _json

    json_mode = output_format == "json"

    try:
        path = _find_pipeline(file)
        af = parse_file(path)
    except (click.ClickException, Exception) as e:
        if json_mode:
            click.echo(_json.dumps({
                "status": "failed", "pipeline": file or "",
                "error": str(e), "steps": [],
            }, indent=2))
            raise SystemExit(2)
        raise

    if dry:
        if not json_mode:
            console.print(f"\n  [dim]Dry run:[/dim] [bold]{af.name}[/bold]\n")
        graph = build_graph(af)
        layers = get_execution_order(graph)
        if json_mode:
            click.echo(_json.dumps({
                "status": "dry_run", "pipeline": af.name,
                "dag_layers": [list(layer) for layer in layers],
                "step_count": len(af.steps),
            }, indent=2))
        else:
            for i, layer in enumerate(layers):
                for name in layer:
                    s = af.steps[name]
                    console.print(f"  {i + 1}. [bold]{name}[/bold] [dim]({s.primitive_type})[/dim]")
            console.print()
        return

    # Parse --input overrides
    overrides = {}
    if input_arg:
        raw_inputs = parse_input_arg(input_arg)
        overrides = load_inputs(raw_inputs)

    # Parse -i key=value overrides (more specific, wins over --input)
    if input_overrides:
        from aiorch.inputs import parse_kv_inputs
        try:
            kv_overrides = parse_kv_inputs(input_overrides)
        except ValueError as e:
            raise click.ClickException(str(e))
        overrides = {**overrides, **kv_overrides}

    # --model override: inject into context for runtime resolution
    if model:
        overrides["__model_override__"] = model

    # Resolve logging config: pipeline overrides global
    cfg = get_config()
    log_level = cfg.logging.level
    sink_cfg = cfg.logging.sink
    if af.logging:
        if "level" in af.logging:
            log_level = af.logging["level"]
        if "sink" in af.logging:
            sink_cfg = af.logging["sink"]

    sink = create_sink(sink_cfg)

    run_id = start_run(af.name, str(path))
    start_time = time.time()
    step_count = 0
    total_cost = 0.0

    logger = RunLogger(run_id, af.name, console_level=log_level, sink=sink)

    # Shared mutable dicts for cost/meta accumulation
    shared_meta: dict[str, dict] = {}
    shared_costs: dict[str, float] = {}
    overrides[LOGGER_KEY] = logger
    overrides[META_KEY] = shared_meta
    overrides[COST_KEY] = shared_costs

    # User-visible runtime metadata ({{_meta.run_id}}, {{_meta.trigger_type}}, etc.)
    from aiorch.constants import RUNTIME_META_KEY
    overrides[RUNTIME_META_KEY] = {
        "run_id": run_id,
        "pipeline_name": af.name,
        "trigger_type": "manual",
        "triggered_at": time.time(),
    }

    # Per-step tracking for JSON output
    step_results: list[dict] = []

    def on_start(name: str):
        nonlocal step_count
        step_count += 1
        ptype = af.steps[name].primitive_type if name in af.steps else "unknown"
        logger.step_start(name, ptype)
        if not json_mode:
            console.print(f"  [dim]▶[/dim] {name}")

    def on_done(name: str, result):
        nonlocal total_cost
        ptype = af.steps[name].primitive_type if name in af.steps else "unknown"
        meta = shared_meta.get(name, {})
        logger.step_done(
            name, ptype, result=result,
            cost=meta.get("cost", 0),
            prompt_tokens=meta.get("prompt_tokens", 0),
            completion_tokens=meta.get("completion_tokens", 0),
            model=meta.get("model", ""),
        )
        step_cost = meta.get("cost", 0)

        if json_mode:
            step_results.append({
                "name": name, "status": "success", "primitive": ptype,
                "duration_ms": round((time.time() - start_time) * 1000),
                "cost": step_cost, "error": None,
            })
        else:
            t = time.time() - start_time
            step_done(name, t * 1000, cost=step_cost)
            if result and ptype == "run":
                console.print(f"    {str(result).rstrip()}")
            elif verbose and result:
                preview = str(result)[:200]
                console.print(f"    [dim]{preview}[/dim]")

        # --max-cost guard
        if max_cost is not None:
            current = sum(shared_costs.values())
            if current > max_cost:
                raise RuntimeError(
                    f"Cost limit ${max_cost:.4f} exceeded "
                    f"(${current:.4f} spent). Run aborted."
                )

    def on_error(name: str, err: Exception):
        ptype = af.steps[name].primitive_type if name in af.steps else "unknown"
        logger.step_failed(name, ptype, err)
        if json_mode:
            step_results.append({
                "name": name, "status": "failed", "primitive": ptype,
                "duration_ms": round((time.time() - start_time) * 1000),
                "cost": 0, "error": str(err),
            })
        else:
            step_failed(name, str(err))

    if not json_mode:
        console.print(f"\n  [bold]{af.name}[/bold]\n")

    if from_step and not json_mode:
        console.print(f"  [yellow]Resuming from '{from_step}' — earlier steps skipped. Use --input to supply missing values.[/yellow]\n")

    def _checkpoint(name: str, result):
        """Persist step output for potential resume."""
        import json as _json
        try:
            output_json = _json.dumps(result, default=str)
            from aiorch.storage import save_step_output
            save_step_output(run_id, name, output_json)
        except Exception:
            pass  # Best-effort; don't fail the run for checkpoint errors

    try:
        asyncio.run(
            execute(
                af,
                runner=execute_step,
                context=overrides if overrides else None,
                on_step_start=on_start,
                on_step_done=on_done,
                on_step_error=on_error,
                step_filter=step,
                from_step=from_step,
                checkpoint=_checkpoint,
            )
        )
        # Total cost from shared accumulator
        total_cost = sum(shared_costs.values())

        duration = time.time() - start_time
        logger.finish("success", total_cost)

        if json_mode:
            click.echo(_json.dumps({
                "status": "success", "pipeline": af.name,
                "duration_seconds": round(duration, 2),
                "total_cost": total_cost,
                "steps": step_results, "error": None,
            }, indent=2))
        else:
            if shared_costs:
                print_cost_breakdown(shared_costs)
            print_run_summary(af.name, step_count, duration, total_cost)

    except Exception as e:
        duration = time.time() - start_time
        total_cost = sum(shared_costs.values())
        logger.finish("failed", total_cost)

        if json_mode:
            click.echo(_json.dumps({
                "status": "failed", "pipeline": af.name,
                "duration_seconds": round(duration, 2),
                "total_cost": total_cost,
                "steps": step_results, "error": str(e),
            }, indent=2))
            raise SystemExit(1)

        if not watch:
            raise click.ClickException(str(e))
        console.print(f"  [red]Error:[/red] {e}")

    if watch:
        try:
            from watchfiles import watch as watch_files
        except ImportError:
            raise click.ClickException("--watch requires watchfiles: pip install aiorch[watch]")

        console.print(f"  [dim]Watching {path} for changes...[/dim]\n")
        for changes in watch_files(path):
            console.print("\n  [dim]File changed, re-running...[/dim]\n")
            try:
                af = parse_file(path)

                run_id = start_run(af.name, str(path))
                start_time = time.time()
                step_count = 0
                total_cost = 0.0
                shared_meta.clear()
                shared_costs.clear()

                # New logger for this run
                logger = RunLogger(run_id, af.name, console_level=log_level, sink=sink)
                overrides[LOGGER_KEY] = logger

                console.print(f"\n  [bold]{af.name}[/bold]\n")

                asyncio.run(
                    execute(
                        af,
                        runner=execute_step,
                        context=overrides if overrides else None,
                        on_step_start=on_start,
                        on_step_done=on_done,
                        on_step_error=on_error,
                        step_filter=step,
                        from_step=from_step,
                    )
                )
                total_cost = sum(shared_costs.values())
                duration = time.time() - start_time
                logger.finish("success", total_cost)
                if shared_costs:
                    print_cost_breakdown(shared_costs)
                print_run_summary(af.name, step_count, duration, total_cost)

            except Exception as e:
                try:
                    logger.finish("failed", 0)
                except Exception:
                    pass
                console.print(f"  [red]Error:[/red] {e}")

            console.print(f"  [dim]Watching {path} for changes...[/dim]\n")


@main.command()
@click.argument("run_id", type=int)
@click.option("--verbose", "-v", is_flag=True, help="Show step inputs and outputs")
def resume(run_id: int, verbose: bool):
    """Resume a failed run from its last checkpoint."""
    from aiorch.runtime import execute_step, COST_KEY, META_KEY, LOGGER_KEY, CONFIG_KEY
    from aiorch.logging import RunLogger
    from aiorch.core.config import get_config
    from aiorch.logging.sinks import create_sink
    from aiorch.storage import init_storage, get_run, get_run_steps, get_step_outputs

    init_storage()

    # 1. Load run record
    run_record = get_run(run_id)
    if not run_record:
        raise click.ClickException(f"Run #{run_id} not found")
    if run_record["status"] == "success":
        raise click.ClickException(f"Run #{run_id} already succeeded. Nothing to resume.")

    # 2. Load the pipeline file
    pipeline_file = run_record.get("file")
    if not pipeline_file:
        raise click.ClickException(f"Run #{run_id} has no pipeline file recorded")
    pipeline_path = Path(pipeline_file)
    if not pipeline_path.exists():
        raise click.ClickException(f"Pipeline file not found: {pipeline_file}")
    af = parse_file(pipeline_path)

    # 3. Load successful step outputs as restored context
    restored = get_step_outputs(run_id)

    # 4. Identify failed steps
    steps = get_run_steps(run_id)
    failed_steps = [s["step_name"] for s in steps if s["status"] == "failed"]

    if not failed_steps and not restored:
        raise click.ClickException(f"Run #{run_id} has no checkpointed data to resume from")

    resume_from = failed_steps[0] if failed_steps else None

    console.print(f"\n  [bold]Resuming run #{run_id}[/bold]")
    console.print(f"  Pipeline: {af.name}")
    if resume_from:
        console.print(f"  Resuming from: [yellow]{resume_from}[/yellow]")
    console.print(f"  Restored {len(restored)} step output(s) from checkpoint\n")

    # 5. Create new run record
    new_run_id = start_run(af.name, str(pipeline_path))
    start_time = time.time()
    step_count = 0
    total_cost = 0.0

    cfg = get_config()
    sink = create_sink(cfg.logging.sink)
    logger = RunLogger(new_run_id, af.name, console_level=cfg.logging.level, sink=sink)

    shared_meta: dict[str, dict] = {}
    shared_costs: dict[str, float] = {}
    overrides: dict = {}
    overrides[LOGGER_KEY] = logger
    overrides[META_KEY] = shared_meta
    overrides[COST_KEY] = shared_costs
    overrides[CONFIG_KEY] = cfg

    def on_start(name: str):
        nonlocal step_count
        step_count += 1
        ptype = af.steps[name].primitive_type if name in af.steps else "unknown"
        logger.step_start(name, ptype)
        console.print(f"  [dim]▶[/dim] {name}")

    def on_done(name: str, result):
        ptype = af.steps[name].primitive_type if name in af.steps else "unknown"
        meta = shared_meta.get(name, {})
        # Don't log restored steps as new completions
        if name in restored:
            console.print(f"  [green]✓[/green] {name}  [dim](restored)[/dim]")
            return
        logger.step_done(
            name, ptype, result=result,
            cost=meta.get("cost", 0),
            prompt_tokens=meta.get("prompt_tokens", 0),
            completion_tokens=meta.get("completion_tokens", 0),
            model=meta.get("model", ""),
        )
        t = time.time() - start_time
        cost = meta.get("cost", 0)
        step_done(name, t * 1000, cost=cost)
        if result and ptype == "run":
            console.print(f"    {str(result).rstrip()}")
        elif verbose and result:
            preview = str(result)[:200]
            console.print(f"    [dim]{preview}[/dim]")

    def on_error(name: str, err: Exception):
        ptype = af.steps[name].primitive_type if name in af.steps else "unknown"
        logger.step_failed(name, ptype, err)
        step_failed(name, str(err))

    def _checkpoint(name: str, result):
        import json as _json
        try:
            output_json = _json.dumps(result, default=str)
            from aiorch.storage import save_step_output
            save_step_output(new_run_id, name, output_json)
        except Exception:
            pass

    try:
        asyncio.run(
            execute(
                af,
                runner=execute_step,
                context=overrides,
                on_step_start=on_start,
                on_step_done=on_done,
                on_step_error=on_error,
                from_step=resume_from,
                checkpoint=_checkpoint,
                restored_outputs=restored,
            )
        )
        total_cost = sum(shared_costs.values())
        duration = time.time() - start_time
        logger.finish("success", total_cost)
        if shared_costs:
            print_cost_breakdown(shared_costs)
        print_run_summary(af.name, step_count, duration, total_cost)

    except Exception as e:
        duration = time.time() - start_time
        logger.finish("failed", total_cost)
        raise click.ClickException(str(e))


@main.command()
@click.argument("file", required=False)
@click.option("--format", "output_format", type=click.Choice(["text", "json"]), default="text",
              help="Output format (text or json)")
def validate(file: str | None, output_format: str):
    """Validate a pipeline's syntax and structure."""
    import json as _json

    path = _find_pipeline(file)
    try:
        af = parse_file(path)
        graph = build_graph(af)
        layers = get_execution_order(graph)

        # `trigger:` was removed from the YAML spec; schedules are
        # managed via the schedules table only. No validate_trigger.
        warnings: list[str] = []

        if output_format == "json":
            click.echo(_json.dumps({
                "valid": True,
                "errors": [],
                "warnings": warnings,
                "step_count": len(af.steps),
                "dag_layers": [list(layer) for layer in layers],
            }, indent=2))
            raise SystemExit(0)

        print_validation_ok(af)
        for w in warnings:
            console.print(f"    [yellow]⚠[/yellow] {w}")
        if warnings:
            console.print()

    except SystemExit:
        raise
    except Exception as e:
        if output_format == "json":
            click.echo(_json.dumps({
                "valid": False,
                "errors": [str(e)],
                "warnings": [],
                "step_count": 0,
                "dag_layers": [],
            }, indent=2))
            raise SystemExit(2)
        print_validation_error(str(e))
        raise SystemExit(1)


@main.command()
@click.argument("file", required=False)
def visualize(file: str | None):
    """Show the step DAG in the terminal."""
    path = _find_pipeline(file)
    af = parse_file(path)
    print_dag(af)


@main.command(name="list")
@click.argument("file", required=False)
def list_steps(file: str | None):
    """List all steps in a pipeline."""
    path = _find_pipeline(file)
    af = parse_file(path)
    print_step_list(af)


@main.command()
@click.argument("step_name")
@click.argument("file", required=False)
def explain(step_name: str, file: str | None):
    """Explain what a step does."""
    path = _find_pipeline(file)
    af = parse_file(path)
    if step_name not in af.steps:
        raise click.ClickException(f"Step '{step_name}' not found")
    print_step_explanation(af.steps[step_name], af)


@main.command()
@click.argument("file", required=False)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--format", "output_format", type=click.Choice(["text", "json"]), default="text",
              help="Output format (text or json)")
def plan(file: str | None, output_json: bool, output_format: str):
    """Show execution plan without running — DAG layers, costs, conditions."""
    path = _find_pipeline(file)
    af = parse_file(path)
    from aiorch.core.plan import build_plan
    execution_plan = build_plan(af)

    if output_json or output_format == "json":
        import json as _json
        from dataclasses import asdict
        click.echo(_json.dumps(asdict(execution_plan), indent=2))
    else:
        from aiorch.ui.display import print_execution_plan
        print_execution_plan(execution_plan)


@main.command()
@click.argument("file", required=False)
@click.option("--format", "output_format", type=click.Choice(["text", "json"]), default="text",
              help="Output format (text or json)")
def cost(file: str | None, output_format: str):
    """Estimate the cost of running a pipeline."""
    import json as _json

    path = _find_pipeline(file)
    af = parse_file(path)
    from aiorch.core.cost import estimate_pipeline_cost
    estimates = estimate_pipeline_cost(af)

    if output_format == "json":
        click.echo(_json.dumps({
            "pipeline": af.name,
            "estimated_cost": sum(e[1] for e in estimates),
            "per_step": [{"step": name, "cost": c} for name, c in estimates],
        }, indent=2))
        return

    print_cost_estimate(af.name, estimates)


@main.command()
@click.option("--template", "-t", help="Template name (use --list to see options)")
@click.option("--list", "list_templates_flag", is_flag=True, help="List available templates")
def init(template: str | None, list_templates_flag: bool):
    """Create a starter pipeline from a template."""
    from aiorch.templates import list_templates, get_template

    if list_templates_flag:
        templates = list_templates()
        console.print("\n  [bold]Available templates:[/bold]\n")
        for t in templates:
            console.print(f"    [bold]{t.name:<20}[/bold] [dim]{t.description}[/dim]")
        console.print(f"\n  Usage: aiorch init --template {templates[0].name}\n")
        return

    filename = "Aiorch.yaml"
    if Path(filename).exists():
        raise click.ClickException(f"{filename} already exists")

    template_name = template or "default"
    content = get_template(template_name)
    if content is None:
        available = ", ".join(t.name for t in list_templates())
        raise click.ClickException(f"Unknown template '{template_name}'. Available: {available}")

    Path(filename).write_text(content)
    console.print(f"\n  [green]✓[/green] Created {filename} (template: {template_name})\n")


@main.command()
@click.argument("run_id", required=False, type=int)
@click.option("--format", "output_format", type=click.Choice(["text", "json"]), default="text",
              help="Output format (text or json)")
@click.option("--limit", default=20, type=int, help="Max runs to show (default: 20)")
def history(run_id: int | None, output_format: str, limit: int):
    """Show past runs, or step details for a specific run."""
    import json as _json
    from aiorch.storage import init_storage
    init_storage()

    if output_format == "json":
        if run_id is not None:
            run_record = get_run(run_id)
            if not run_record:
                click.echo(_json.dumps({"error": f"Run #{run_id} not found"}))
                raise SystemExit(1)
            steps = get_run_steps(run_id)
            click.echo(_json.dumps(
                {"run": dict(run_record), "steps": [dict(s) for s in steps]},
                indent=2, default=str))
        else:
            runs = get_runs(limit=limit)
            click.echo(_json.dumps([dict(r) for r in runs], indent=2, default=str))
        return

    if run_id is not None:
        run_record = get_run(run_id)
        if not run_record:
            raise click.ClickException(f"Run #{run_id} not found")
        steps = get_run_steps(run_id)
        print_run_details(run_record, steps)
    else:
        runs = get_runs(limit=limit)
        print_history(runs)


@main.command()
def dashboard():
    """Open the terminal dashboard — runs, costs, cache stats."""
    from aiorch.storage import init_storage
    init_storage()
    from aiorch.ui.dashboard import render_dashboard
    render_dashboard()


@main.command()
@click.argument("run_id", type=int)
def trace(run_id: int):
    """Show detailed step trace for a run — timing, tokens, cost, model."""
    from aiorch.logging import load_run_log
    events = load_run_log(run_id)

    if not events:
        raise click.ClickException(f"No trace log found for run #{run_id}. Logs at ~/.aiorch/logs/")

    from rich.table import Table
    console.print(f"\n  [bold]Trace — Run #{run_id}[/bold]\n")

    table = Table(show_lines=False, expand=False)
    table.add_column("Step", style="bold", no_wrap=True)
    table.add_column("Type", style="cyan")
    table.add_column("Status")
    table.add_column("Duration", justify="right")
    table.add_column("Model", style="dim")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", style="yellow", justify="right")

    for e in events:
        if e.get("event") in ("run_start", "run_end"):
            continue
        if e.get("status") == "started":
            continue

        status = e.get("status", "")
        status_style = "green" if status == "success" else "red" if status == "failed" else "dim"
        dur = f"{e.get('duration_ms', 0):.0f}ms" if e.get("duration_ms") else "-"
        model = e.get("model", "") or "-"
        pt = e.get("prompt_tokens", 0)
        ct = e.get("completion_tokens", 0)
        tokens = f"{pt}→{ct}" if (pt or ct) else "-"
        cost = f"${e.get('cost', 0):.4f}" if e.get("cost", 0) > 0 else "-"

        table.add_row(
            e.get("step_name", ""),
            e.get("primitive", ""),
            f"[{status_style}]{status}[/{status_style}]",
            dur, model, tokens, cost,
        )

    console.print(table)

    for e in events:
        if e.get("event") == "run_end":
            console.print(f"\n  Total: {e.get('total_steps', 0)} steps, ${e.get('total_cost', 0):.4f}")
            console.print(f"  Log: ~/.aiorch/logs/run-{run_id}.jsonl\n")


@main.command()
@click.option("--format", "output_format", type=click.Choice(["text", "json"]), default="text",
              help="Output format (text or json)")
def doctor(output_format: str):
    """Check setup — API keys, tools, dependencies."""
    checks = []

    # Python version
    py_ok = sys.version_info >= (3, 11)
    checks.append(("Python >= 3.11", py_ok, f"Python {sys.version_info.major}.{sys.version_info.minor}"))

    # openai SDK
    try:
        import openai
        ver = getattr(openai, "__version__", "installed")
        checks.append(("openai SDK installed", True, f"v{ver}"))
    except ImportError:
        checks.append(("openai SDK installed", False, "pip install openai"))

    # litellm (optional)
    try:
        import litellm
        ver = getattr(litellm, "__version__", "installed")
        checks.append(("litellm installed (optional)", True, f"v{ver}"))
    except ImportError:
        checks.append(("litellm installed", None, "pip install aiorch[server]"))

    # API keys
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))

    from aiorch.core.config import get_config
    cfg = get_config()
    has_config_key = bool(cfg.llm.api_key)

    any_key = has_openai or has_anthropic or has_openrouter or has_config_key
    key_detail = []
    if has_openai:
        key_detail.append("OPENAI_API_KEY")
    if has_anthropic:
        key_detail.append("ANTHROPIC_API_KEY")
    if has_openrouter:
        key_detail.append("OPENROUTER_API_KEY")
    if has_config_key:
        key_detail.append("aiorch.yaml")
    checks.append(("LLM API key configured", any_key, ", ".join(key_detail) if key_detail else "Set OPENAI_API_KEY or configure aiorch.yaml"))

    # Config file
    from aiorch.core.config import find_config
    config_path = find_config()
    checks.append(("Config file found", config_path is not None, str(config_path) if config_path else "Optional — create aiorch.yaml"))

    # Database connectivity
    db_url = os.environ.get("DATABASE_URL", "")
    checks.append(("DATABASE_URL configured", bool(db_url), db_url[:40] + "..." if db_url else "Not set — required for server mode"))

    # Optional tools
    for tool_name in ["gh", "kubectl", "aws", "psql", "curl"]:
        found = shutil.which(tool_name) is not None
        checks.append((f"{tool_name} CLI", found, "available" if found else "optional — install for integrations"))

    if output_format == "json":
        import json as _json
        all_ok = all(ok is True for _, ok, _ in checks if ok is not None)
        click.echo(_json.dumps({
            "all_ok": all_ok,
            "checks": [
                {"name": name, "ok": ok, "detail": detail}
                for name, ok, detail in checks
            ],
        }, indent=2))
        raise SystemExit(0 if all_ok else 1)

    print_doctor_results(checks)


@main.command()
@click.option("--host", default=None, help="Bind host (default: from config or 127.0.0.1)")
@click.option("--port", default=None, type=int, help="Bind port (default: from config or 7842)")
@click.option("--daemon", is_flag=True, help="Run in background")
@click.option("--reload", is_flag=True, help="Auto-reload on code changes (dev mode)")
@click.option("--pipeline-dir", default=None, type=click.Path(exists=True), help="Directory to discover pipeline YAML files from (default: CWD)")
@click.option("--dev", is_flag=True, help="Dev mode — disables auth for local development. Requires DATABASE_URL.")
def serve(host, port, daemon, reload, pipeline_dir, dev):
    """Start the aiorch server."""
    try:
        import uvicorn
    except ImportError:
        raise click.ClickException(
            "Server dependencies not installed. Run:\n  pip install aiorch[server]"
        )

    # Platform mode: load .env for DATABASE_URL, secrets, etc.
    from dotenv import load_dotenv
    load_dotenv()

    from aiorch.core.config import get_config

    if dev:
        if not os.environ.get("DATABASE_URL"):
            raise click.ClickException(
                "DATABASE_URL is required even in single-user mode.\n"
                "Example: export DATABASE_URL=postgresql://localhost/aiorch\n"
                "Then run: aiorch serve --dev"
            )
        os.environ["AIORCH_AUTH_ENABLED"] = "false"
        os.environ["AIORCH_SINGLE_USER"] = "true"
        console.print("  [dim]Dev mode — auth disabled[/dim]")
    from aiorch.storage import init_storage
    from aiorch.core.paths import set_pipeline_dir

    cfg = get_config()
    host = host or cfg.server.host
    port = port or cfg.server.port

    if pipeline_dir:
        set_pipeline_dir(pipeline_dir)

    init_storage()

    if daemon:
        import subprocess
        pid_file = Path.home() / ".aiorch" / "daemon.pid"
        pid_file.parent.mkdir(parents=True, exist_ok=True)

        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "aiorch.server.app:create_app", "--factory",
             "--host", host, "--port", str(port)],
            start_new_session=True,
        )
        pid_file.write_text(str(proc.pid))
        console.print(f"  [green]✓[/green] Server started (PID {proc.pid})")
        console.print(f"  http://{host}:{port}")
        return

    console.print(f"\n  [bold]aiorch server[/bold] v{__version__}")
    console.print(f"  http://{host}:{port}\n")

    if cfg.server.open_browser:
        import webbrowser
        import threading

        def _open():
            import time as _t
            _t.sleep(1.5)
            webbrowser.open(f"http://{host}:{port}")

        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(
        "aiorch.server.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


@main.command()
def status():
    """Show server and scheduler status."""
    from aiorch.storage import init_storage, get_store, get_runs
    init_storage()

    pid_file = Path.home() / ".aiorch" / "daemon.pid"
    if pid_file.exists():
        pid = pid_file.read_text().strip()
        console.print(f"  Server: [green]running[/green] (PID {pid})")
    else:
        console.print("  Server: [dim]stopped[/dim]")

    # Storage
    store = get_store()
    store_type = type(store).__name__
    console.print(f"  Storage: {store_type}")

    # Recent runs
    runs = get_runs(limit=5)
    if runs:
        console.print(f"\n  Last {len(runs)} runs:")
        for r in runs:
            status_style = "green" if r["status"] == "success" else "red" if r["status"] == "failed" else "yellow"
            cost = f"${r.get('total_cost', 0):.4f}" if r.get("total_cost") else "-"
            console.print(f"    #{r['id']} {r['name']} [{status_style}]{r['status']}[/{status_style}] {cost}")
    else:
        console.print("\n  No runs yet.")
    console.print()


@main.command()
def stop():
    """Stop the aiorch daemon."""
    pid_file = Path.home() / ".aiorch" / "daemon.pid"
    if not pid_file.exists():
        raise click.ClickException("No daemon PID file found. Is the server running?")

    pid = int(pid_file.read_text().strip())
    try:
        os.kill(pid, 15)  # SIGTERM
        console.print(f"  [green]✓[/green] Sent SIGTERM to PID {pid}")
    except ProcessLookupError:
        console.print(f"  [yellow]⚠[/yellow] Process {pid} not found (already stopped?)")
    finally:
        pid_file.unlink(missing_ok=True)


# -- Templates --

