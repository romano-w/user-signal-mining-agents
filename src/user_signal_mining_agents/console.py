"""Shared Rich console and logging helpers for beautiful CLI output."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

# Custom theme for the project
_THEME = Theme({
    "info": "cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "dim": "dim",
    "label": "bold magenta",
    "step": "bold cyan",
    "cached": "dim green",
    "key": "bold white",
})

console = Console(theme=_THEME, highlight=False)


def header(title: str, subtitle: str = "") -> None:
    """Print a prominent section header."""
    content = f"[bold]{title}[/bold]"
    if subtitle:
        content += f"\n[dim]{subtitle}[/dim]"
    console.print(Panel(content, border_style="cyan", padding=(0, 2)))


def step(agent: str, message: str) -> None:
    """Print a step-level log line with agent tag."""
    console.print(f"  [step][{agent}][/step] {message}")


def success(agent: str, message: str) -> None:
    """Print a success message."""
    console.print(f"  [success][{agent}] {message}[/success]")


def cached(agent: str, message: str) -> None:
    """Print a cache-hit message."""
    console.print(f"  [cached][{agent}] {message}[/cached]")


def warning(message: str) -> None:
    """Print a warning."""
    console.print(f"  [warning]{message}[/warning]")


def error(message: str) -> None:
    """Print an error."""
    console.print(f"  [error]{message}[/error]")


def llm_tokens(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    """Print LLM token usage."""
    console.print(
        f"  [dim]  LLM[/dim] [key]{model}[/key] "
        f"[dim]{prompt_tokens:,} prompt / {completion_tokens:,} completion tokens[/dim]"
    )


def llm_rate_limited(wait: float, attempt: int, max_retries: int) -> None:
    """Print a rate-limit retry message."""
    console.print(
        f"  [warning]  LLM rate limited, rotating key & waiting {wait:.0f}s "
        f"(attempt {attempt}/{max_retries})[/warning]"
    )


def prompt_table(prompt_id: str, index: int, total: int) -> None:
    """Print numbered prompt header."""
    console.print()
    console.rule(f"[bold cyan]{index}/{total}[/bold cyan]  {prompt_id}", style="cyan")


def results_table(scores: dict[str, tuple[float, float]], baseline_avg: float, pipeline_avg: float) -> None:
    """Print a comparison table of baseline vs pipeline scores."""
    table = Table(title="Aggregate Scores", border_style="cyan", show_lines=False)
    table.add_column("Dimension", style="key")
    table.add_column("Baseline", justify="center", style="info")
    table.add_column("Pipeline", justify="center", style="info")
    table.add_column("Delta", justify="center")

    for dim, (b, p) in scores.items():
        delta = p - b
        delta_style = "green" if delta > 0 else "red" if delta < 0 else "dim"
        delta_str = f"[{delta_style}]{delta:+.2f}[/{delta_style}]"
        table.add_row(dim, f"{b:.2f}", f"{p:.2f}", delta_str)

    # Overall row
    delta = pipeline_avg - baseline_avg
    delta_style = "green" if delta > 0 else "red" if delta < 0 else "dim"
    delta_str = f"[{delta_style}]{delta:+.2f}[/{delta_style}]"
    table.add_row(
        "[bold]Overall[/bold]",
        f"[bold]{baseline_avg:.2f}[/bold]",
        f"[bold]{pipeline_avg:.2f}[/bold]",
        f"[bold]{delta_str}[/bold]",
        end_section=True,
    )

    console.print(table)
