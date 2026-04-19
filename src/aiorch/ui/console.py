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

"""Rich console renderer for pipeline execution — spinners, colors, inline logs."""

from __future__ import annotations

from typing import Any

from rich.console import Console

from aiorch.logging import LogLevel


console = Console()

# Spinner frames
_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# Step status icons
_ICONS = {
    "success": "[green]✓[/green]",
    "failed": "[red]✗[/red]",
    "running": "[yellow]●[/yellow]",
    "waiting": "[dim]○[/dim]",
    "skipped": "[dim]⊘[/dim]",
}

# Log level styles
_LOG_STYLES = {
    "DEBUG": "dim",
    "INFO": "blue",
    "WARN": "yellow",
    "WARNING": "yellow",
    "ERROR": "red",
}


def print_header(pipeline_name: str) -> None:
    console.print(f"\n  [bold]{pipeline_name}[/bold]\n")


def print_step_start(name: str) -> None:
    console.print(f"  [yellow]●[/yellow] [bold]{name}[/bold]  [dim]running...[/dim]")


def print_step_done(
    name: str,
    primitive: str,
    duration_ms: float,
    cost: float = 0,
    model: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    output_preview: str = "",
) -> None:
    """Print a completed step with all metadata."""
    parts = []

    # Icon + name
    parts.append(f"  [green]✓[/green] [bold]{name}[/bold]")

    # Type
    type_colors = {"run": "yellow", "prompt": "magenta", "agent": "cyan", "action": "blue", "flow": "green"}
    tc = type_colors.get(primitive, "dim")
    parts.append(f"  [{tc}]{primitive}[/{tc}]")

    # Duration
    if duration_ms >= 1000:
        parts.append(f"  [dim]{duration_ms / 1000:.1f}s[/dim]")
    else:
        parts.append(f"  [dim]{duration_ms:.0f}ms[/dim]")

    # Cost
    if cost > 0:
        parts.append(f"  [yellow]${cost:.4f}[/yellow]")

    # Model + tokens
    if model:
        # Shorten model name
        short_model = model.split("/")[-1] if "/" in model else model
        parts.append(f"  [dim]{short_model}[/dim]")
    if prompt_tokens or completion_tokens:
        parts.append(f"  [dim]{prompt_tokens}→{completion_tokens}[/dim]")

    console.print("".join(parts))

    # Output preview (truncated)
    if output_preview:
        preview = output_preview[:120].replace("\n", " ")
        console.print(f"    [dim]{preview}[/dim]")


def print_step_failed(name: str, error: str) -> None:
    console.print(f"  [red]✗[/red] [bold]{name}[/bold]")
    console.print(f"    [red]│[/red] [red]ERROR[/red]  {error[:200]}")


def print_step_skipped(name: str) -> None:
    console.print(f"  [dim]⊘ {name}  skipped (condition false)[/dim]")


def print_log(step_name: str, level: str, message: str, data: Any = None, console_level: str = "WARNING") -> None:
    """Print an inline log entry under the current step."""
    if not LogLevel.should_print(level, console_level):
        return

    style = _LOG_STYLES.get(level, "dim")
    # Each level has a distinct prefix icon
    icons = {"DEBUG": "·", "INFO": "→", "WARN": "⚠", "WARNING": "⚠", "ERROR": "✗"}
    icon = icons.get(level, "·")

    console.print(f"    [{style}]{icon} {level:<5}[/{style}]  {message}")

    if data and isinstance(data, dict):
        for k, v in data.items():
            val = str(v)[:120] if v else "-"
            console.print(f"    [{style}]       {k}:[/{style}] [dim]{val}[/dim]")


def print_summary(
    pipeline_name: str,
    step_count: int,
    duration_s: float,
    total_cost: float,
    cache_hits: int = 0,
    cache_saved: float = 0,
) -> None:
    """Print the run summary line."""
    console.print()
    console.print(f"  [dim]{'─' * 60}[/dim]")

    parts = [f"  [bold]{step_count} steps[/bold]"]
    parts.append(f"[dim]·[/dim] {duration_s:.1f}s")
    if total_cost > 0:
        parts.append(f"[dim]·[/dim] [yellow]${total_cost:.4f}[/yellow]")
    if cache_hits > 0:
        parts.append(f"[dim]·[/dim] [dim]cache: {cache_hits} hits (${cache_saved:.4f} saved)[/dim]")

    console.print(" ".join(parts))
    console.print()


def print_saved_files(files: list[str]) -> None:
    for f in files:
        console.print(f"  [dim]saved →[/dim] {f}")
