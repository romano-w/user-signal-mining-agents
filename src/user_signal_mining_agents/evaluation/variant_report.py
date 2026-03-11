"""Markdown report generator for variant evaluations."""

from __future__ import annotations

from pathlib import Path

from .variant_runner import RUBRIC_DIMS, VariantEvaluationSummary


def generate_variant_report(summary: VariantEvaluationSummary, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "variant_evaluation_report.md"

    lines: list[str] = [
        "# Variant Evaluation Report\n",
        f"**Control variant:** `{summary.control_variant}`",
        f"**Prompts evaluated ({len(summary.prompt_ids)}):** {', '.join(summary.prompt_ids)}\n",
        "## Aggregate Ranking\n",
        "| Variant | Control Overall | Variant Overall | Delta |",
        "|---|:---:|:---:|:---:|",
    ]

    for aggregate in summary.aggregates:
        sign = "+" if aggregate.delta_overall >= 0 else ""
        lines.append(
            f"| {aggregate.variant} | {aggregate.control_overall:.2f} | "
            f"{aggregate.variant_overall:.2f} | {sign}{aggregate.delta_overall:.2f} |"
        )

    lines.append("")

    for aggregate in summary.aggregates:
        lines.append(f"## Variant: `{aggregate.variant}`")
        lines.append(aggregate.description)
        lines.append("")
        lines.append("| Dimension | Control Avg | Variant Avg | Delta |")
        lines.append("|---|:---:|:---:|:---:|")

        for dim in RUBRIC_DIMS:
            c = aggregate.control_scores.get(dim, 0.0)
            v = aggregate.variant_scores.get(dim, 0.0)
            d = v - c
            sign = "+" if d >= 0 else ""
            lines.append(f"| {dim.replace('_', ' ').title()} | {c:.2f} | {v:.2f} | {sign}{d:.2f} |")

        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
