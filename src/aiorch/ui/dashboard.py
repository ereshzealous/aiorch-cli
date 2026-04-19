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

"""Terminal dashboard — rich TUI for run history, costs, and cache stats."""

from __future__ import annotations

import time

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aiorch.storage import get_runs, get_run_steps, get_dashboard_stats, cache_stats


console = Console()


def _time_ago(timestamp: float) -> str:
    """Format a timestamp as a human-readable 'X ago' string."""
    diff = time.time() - timestamp
    if diff < 60:
        return f"{int(diff)}s ago"
    if diff < 3600:
        return f"{int(diff / 60)}m ago"
    if diff < 86400:
        return f"{int(diff / 3600)}h ago"
    return f"{int(diff / 86400)}d ago"


def _cost_bar(cost: float, budget: float = 5.0, width: int = 20) -> str:
    """Render a cost bar like: ████████░░░░ $0.42 / $5.00"""
    if budget <= 0:
        return f"${cost:.2f}"
    ratio = min(cost / budget, 1.0)
    filled = int(ratio * width)
    empty = width - filled
    bar = "█" * filled + "░" * empty
    return f"{bar} ${cost:.2f} / ${budget:.2f}"


def render_dashboard() -> None:
    """Render the full terminal dashboard."""
    stats = get_dashboard_stats()
    cache = cache_stats()
    recent = get_runs(limit=10)

    console.clear()
    console.print()

    # Header
    console.print(Panel(
        "[bold]Aiorch Dashboard[/bold]",
        style="blue",
        expand=False,
    ))
    console.print()

    # --- Row 1: Stats cards ---
    cards = []

    cards.append(Panel(
        f"[bold]{stats['today_runs']}[/bold] runs\n"
        f"[yellow]${stats['today_cost']:.4f}[/yellow]",
        title="[dim]Today[/dim]",
        expand=True,
        width=24,
    ))

    cards.append(Panel(
        f"[bold]{stats['week_runs']}[/bold] runs\n"
        f"[yellow]${stats['week_cost']:.4f}[/yellow]",
        title="[dim]This Week[/dim]",
        expand=True,
        width=24,
    ))

    cards.append(Panel(
        f"[bold]{stats['total_runs']}[/bold] runs\n"
        f"[yellow]${stats['total_cost']:.4f}[/yellow]",
        title="[dim]All Time[/dim]",
        expand=True,
        width=24,
    ))

    rate = stats["success_rate"]
    rate_color = "green" if rate >= 90 else "yellow" if rate >= 70 else "red"
    cards.append(Panel(
        f"[{rate_color}]{rate:.0f}%[/{rate_color}]\n"
        f"[dim]{stats['total_runs']} total[/dim]",
        title="[dim]Success Rate[/dim]",
        expand=True,
        width=24,
    ))

    console.print(Columns(cards, equal=True, expand=True))
    console.print()

    # --- Row 2: Recent runs + Top pipelines ---

    # Recent runs table
    runs_table = Table(title="Recent Runs", show_lines=False, expand=True)
    runs_table.add_column("ID", style="dim", width=4)
    runs_table.add_column("Pipeline", style="bold")
    runs_table.add_column("Status", width=8)
    runs_table.add_column("Steps", justify="right", width=6)
    runs_table.add_column("Cost", style="yellow", justify="right", width=8)
    runs_table.add_column("Duration", style="dim", justify="right", width=8)
    runs_table.add_column("When", style="dim", width=8)

    for run in recent:
        status_style = "green" if run["status"] == "success" else "red"
        duration = ""
        if run.get("finished_at") and run.get("started_at"):
            duration = f"{run['finished_at'] - run['started_at']:.1f}s"
        when = _time_ago(run["started_at"]) if run.get("started_at") else ""

        # Get step count for this run
        # TODO: replace with batch COUNT query before Postgres support (N+1 issue)
        steps = get_run_steps(run["id"])
        step_count = str(len(steps)) if steps else "-"

        cost_str = f"${run.get('total_cost', 0):.4f}"

        runs_table.add_row(
            str(run["id"]),
            run["name"],
            f"[{status_style}]{run['status']}[/{status_style}]",
            step_count,
            cost_str,
            duration,
            when,
        )

    # Top pipelines table
    top_table = Table(title="Top Pipelines (7 days)", show_lines=False, expand=True)
    top_table.add_column("Pipeline", style="bold")
    top_table.add_column("Runs", justify="right", width=5)
    top_table.add_column("Avg Cost", style="yellow", justify="right", width=10)
    top_table.add_column("Avg Time", style="dim", justify="right", width=10)

    if stats["top_pipelines"]:
        for p in stats["top_pipelines"]:
            avg_dur = f"{p['avg_duration']:.1f}s" if p.get("avg_duration") else "-"
            top_table.add_row(
                p["name"],
                str(p["runs"]),
                f"${p['avg_cost']:.4f}" if p.get("avg_cost") else "-",
                avg_dur,
            )
    else:
        top_table.add_row("[dim]No runs this week[/dim]", "", "", "")

    console.print(Columns([runs_table, top_table], expand=True))
    console.print()

    # --- Row 3: Cost bar + Cache stats + Recent failures ---

    # Cost bar
    cost_panel = Panel(
        f"  [dim]7-day spend:[/dim]  {_cost_bar(stats['week_cost'])}\n"
        f"  [dim]All time:[/dim]    [yellow]${stats['total_cost']:.4f}[/yellow]",
        title="[dim]Cost[/dim]",
        expand=True,
    )

    # Cache stats
    cache_panel = Panel(
        f"  [dim]Entries:[/dim]  [bold]{cache['entries']}[/bold]\n"
        f"  [dim]Hits:[/dim]     [bold]{cache['total_hits']}[/bold]\n"
        f"  [dim]Saved:[/dim]    [yellow]${cache['saved_cost']:.4f}[/yellow]",
        title="[dim]Cache[/dim]",
        expand=True,
    )

    # Recent failures
    if stats["recent_failures"]:
        fail_lines = []
        for f in stats["recent_failures"]:
            when = _time_ago(f["started_at"]) if f.get("started_at") else ""
            fail_lines.append(f"  [red]✗[/red] {f['name']}  [dim]{when}[/dim]")
        fail_text = "\n".join(fail_lines)
    else:
        fail_text = "  [green]No recent failures[/green]"

    fail_panel = Panel(
        fail_text,
        title="[dim]Recent Failures[/dim]",
        expand=True,
    )

    console.print(Columns([cost_panel, cache_panel, fail_panel], expand=True))
    console.print()
