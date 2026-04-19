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

"""Rich-based terminal display for pipeline visualization and progress."""

from __future__ import annotations


from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aiorch.core.dag import build_graph, get_execution_order
from aiorch.core.parser import Agentfile, Step


console = Console()


def print_dag(af: Agentfile) -> None:
    """Print the step DAG as a tree view."""
    graph = build_graph(af)
    layers = get_execution_order(graph)

    console.print()
    console.print(Panel(f"[bold]{af.name}[/bold]", style="blue", expand=False))
    console.print()

    for i, layer in enumerate(layers):
        if len(layer) == 1:
            step = af.steps[layer[0]]
            icon = _primitive_icon(step.primitive_type)
            console.print(f"  {icon} [bold]{layer[0]}[/bold]  [dim]({step.primitive_type})[/dim]")
        else:
            console.print("  [dim]├── parallel ──┤[/dim]")
            for name in layer:
                step = af.steps[name]
                icon = _primitive_icon(step.primitive_type)
                console.print(f"  │  {icon} [bold]{name}[/bold]  [dim]({step.primitive_type})[/dim]")
            console.print("  [dim]├──────────────┤[/dim]")

        if i < len(layers) - 1:
            console.print("  [dim]│[/dim]")

    console.print()


def print_step_list(af: Agentfile) -> None:
    """Print a table of all steps."""
    table = Table(title=af.name, show_lines=False)
    table.add_column("Step", style="bold")
    table.add_column("Type", style="cyan")
    table.add_column("Model", style="yellow")
    table.add_column("Depends", style="dim")
    table.add_column("Output", style="green")

    for name, step in af.steps.items():
        table.add_row(
            name,
            step.primitive_type,
            step.model or "-",
            ", ".join(step.depends) if step.depends else "-",
            step.output or "-",
        )

    console.print()
    console.print(table)
    console.print()


def print_step_explanation(step: Step, af: Agentfile) -> None:
    """Print a detailed explanation of what a step does."""
    console.print()
    console.print(f"  {_primitive_icon(step.primitive_type)} [bold]{step.name}[/bold]  [dim]({step.primitive_type})[/dim]")
    console.print()

    ptype = step.primitive_type
    if ptype == "run":
        console.print(f"  [dim]Command:[/dim]  {step.run}")
    elif ptype == "prompt":
        preview = step.prompt[:200] + "..." if len(step.prompt) > 200 else step.prompt
        console.print(f"  [dim]Prompt:[/dim]   {preview}")
    elif ptype == "agent":
        console.print(f"  [dim]Agent:[/dim]    {step.agent}")
        if step.goal:
            console.print(f"  [dim]Goal:[/dim]     {step.goal}")
        if step.tools:
            console.print(f"  [dim]Tools:[/dim]    {', '.join(str(t) for t in step.tools)}")
        console.print(f"  [dim]Max iter:[/dim] {step.max_iterations}")
    elif ptype == "action":
        console.print(f"  [dim]Action:[/dim]   {step.action}")
        if step.config:
            for k, v in list(step.config.items())[:3]:
                console.print(f"  [dim]{k}:[/dim]  {v}")
    elif ptype == "flow":
        console.print(f"  [dim]Flow:[/dim]     {step.flow}")

    if step.model:
        console.print(f"  [dim]Model:[/dim]    {step.model}")
    if step.input:
        console.print(f"  [dim]Input:[/dim]    {step.input}")
    if step.depends:
        console.print(f"  [dim]Depends:[/dim]  {', '.join(step.depends)}")
    if step.output:
        console.print(f"  [dim]Output:[/dim]   {step.output}")
    if step.condition:
        console.print(f"  [dim]Condition:[/dim] {step.condition}")
    if step.foreach:
        console.print(f"  [dim]Foreach:[/dim]  {step.foreach}")
    if step.retry:
        console.print(f"  [dim]Retry:[/dim]    {step.retry}")
    if step.cache:
        console.print("  [dim]Cache:[/dim]    enabled")
    if step.save:
        console.print(f"  [dim]Save:[/dim]     {step.save}")

    console.print()


def print_validation_ok(af: Agentfile) -> None:
    console.print(f"\n  [green]✓[/green] [bold]{af.name}[/bold] is valid")
    console.print(f"    {len(af.steps)} steps\n")


def print_validation_error(error: str) -> None:
    console.print(f"\n  [red]✗[/red] Validation failed: {error}\n")


def step_started(name: str) -> None:
    console.print(f"  [dim]▶[/dim] [bold]{name}[/bold]", end="")


def step_done(name: str, duration_ms: float, cost: float = 0) -> None:
    if cost > 0:
        console.print(f"  [green]✓[/green] {name}  [dim]{duration_ms:.0f}ms[/dim]  [yellow]${cost:.4f}[/yellow]")
    else:
        console.print(f"  [green]✓[/green] {name}  [dim]{duration_ms:.0f}ms[/dim]")


def step_failed(name: str, error: str) -> None:
    console.print(f"  [red]✗[/red] {name}  [red]{error}[/red]")


def print_run_summary(name: str, total_steps: int, duration_s: float, total_cost: float) -> None:
    console.print()
    console.print(f"  [bold]{name}[/bold]")
    console.print(f"  {total_steps} steps, {duration_s:.1f}s, ${total_cost:.4f}")
    console.print()


def print_execution_plan(plan) -> None:
    """Print execution plan as a Rich table."""
    table = Table(title=f"Execution Plan — {plan.pipeline_name}", show_lines=False)
    table.add_column("Step", style="bold")
    table.add_column("Type", style="cyan")
    table.add_column("Layer", style="dim", justify="right")
    table.add_column("Depends", style="dim")
    table.add_column("Model", style="yellow")
    table.add_column("Cache", style="dim", justify="center")
    table.add_column("Condition", style="dim")
    table.add_column("Est. Cost", style="yellow", justify="right")

    for step in plan.steps:
        icon = _primitive_icon(step.primitive_type)
        cache_str = "yes" if step.cache else "-"
        model_str = step.model or "-"
        depends_str = ", ".join(step.depends) if step.depends else "-"
        cond_str = step.condition[:30] + "..." if step.condition and len(step.condition) > 30 else (step.condition or "-")
        cost_str = f"${step.estimated_cost:.4f}" if step.estimated_cost > 0 else "-"

        extras = []
        if step.foreach:
            extras.append(f"foreach:{step.foreach}")
        if step.has_validation:
            extras.append("validated")
        extra_str = f" [{', '.join(extras)}]" if extras else ""

        table.add_row(
            f"{icon} {step.name}{extra_str}",
            step.primitive_type,
            str(step.layer + 1),
            depends_str,
            model_str,
            cache_str,
            cond_str,
            cost_str,
        )

    table.add_section()
    table.add_row(
        f"[bold]{plan.total_steps} steps, {plan.total_layers} layers[/bold]",
        "", "", "", "", "", "",
        f"[bold]${plan.estimated_total_cost:.4f}[/bold]",
    )

    console.print()
    console.print(table)
    console.print("\n  [dim]Estimates are approximate. No LLM calls made.[/dim]\n")


def print_cost_estimate(name: str, estimates: list[tuple[str, float]]) -> None:
    """Print cost estimation table."""
    table = Table(title=f"Cost Estimate — {name}", show_lines=False)
    table.add_column("Step", style="bold")
    table.add_column("Est. Cost", style="yellow", justify="right")

    total = 0.0
    for step_name, cost in estimates:
        table.add_row(step_name, f"${cost:.4f}")
        total += cost

    table.add_section()
    table.add_row("[bold]Total[/bold]", f"[bold]${total:.4f}[/bold]")

    console.print()
    console.print(table)
    console.print("\n  [dim]Estimates are approximate. Actual cost depends on input size and model output.[/dim]\n")


def print_doctor_results(checks: list[tuple[str, bool, str]]) -> None:
    """Print doctor check results."""
    console.print("\n  [bold]Aiorch Doctor[/bold]\n")

    for name, ok, detail in checks:
        icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
        detail_style = "dim" if ok else "yellow"
        console.print(f"  {icon} {name}  [{detail_style}]{detail}[/{detail_style}]")

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    console.print(f"\n  {passed}/{total} checks passed\n")


def print_history(runs: list[dict]) -> None:
    if not runs:
        console.print("\n  No runs yet.\n")
        return

    table = Table(title="Run History", show_lines=False)
    table.add_column("ID", style="dim")
    table.add_column("Name", style="bold")
    table.add_column("Status")
    table.add_column("Cost", style="yellow")
    table.add_column("Duration", style="dim")

    for run in runs:
        status_style = "green" if run["status"] == "success" else "red"
        elapsed = ""
        if run.get("finished_at") and run.get("started_at"):
            elapsed = f"{run['finished_at'] - run['started_at']:.1f}s"

        table.add_row(
            str(run["id"]),
            run["name"],
            f"[{status_style}]{run['status']}[/{status_style}]",
            f"${run.get('total_cost', 0):.4f}",
            elapsed,
        )

    console.print()
    console.print(table)
    console.print()


def print_run_details(run: dict, steps: list[dict]) -> None:
    """Print detailed step-by-step info for a specific run."""
    console.print()

    status_style = "green" if run["status"] == "success" else "red"
    duration = ""
    if run.get("finished_at") and run.get("started_at"):
        duration = f"{run['finished_at'] - run['started_at']:.1f}s"

    console.print(f"  [bold]{run['name']}[/bold]  [dim](run #{run['id']})[/dim]")
    console.print(f"  Status: [{status_style}]{run['status']}[/{status_style}]"
                  f"  Cost: [yellow]${run.get('total_cost', 0):.4f}[/yellow]"
                  f"  Duration: [dim]{duration}[/dim]")
    console.print()

    if not steps:
        console.print("  [dim]No steps recorded.[/dim]")
        console.print()
        return

    table = Table(show_lines=False)
    table.add_column("Step", style="bold")
    table.add_column("Type", style="cyan")
    table.add_column("Status")
    table.add_column("Duration", justify="right", style="dim")
    table.add_column("Cost", justify="right", style="yellow")

    for s in steps:
        s_status_style = "green" if s["status"] == "success" else "red"
        dur = f"{s.get('duration_ms', 0):.0f}ms" if s.get("duration_ms") else "-"
        cost_str = f"${s.get('cost', 0):.4f}" if s.get("cost", 0) > 0 else "-"
        table.add_row(
            s["step_name"],
            s.get("primitive", "-"),
            f"[{s_status_style}]{s['status']}[/{s_status_style}]",
            dur,
            cost_str,
        )

    console.print(table)
    console.print()


def print_cost_breakdown(step_costs: dict[str, float]) -> None:
    """Print a mini table of per-step costs."""
    if not step_costs or all(c == 0 for c in step_costs.values()):
        return

    console.print()
    console.print("  [bold]Cost breakdown[/bold]")
    total = 0.0
    for name, cost in step_costs.items():
        if cost > 0:
            console.print(f"    {name}  [yellow]${cost:.4f}[/yellow]")
            total += cost
    console.print(f"    [dim]Total[/dim]  [yellow]${total:.4f}[/yellow]")


def _primitive_icon(ptype: str) -> str:
    icons = {
        "run": "[yellow]$[/yellow]",
        "prompt": "[magenta]>[/magenta]",
        "agent": "[cyan]@[/cyan]",
        "action": "[blue]![/blue]",
        "flow": "[green]~[/green]",
    }
    return icons.get(ptype, "?")
