"""Report generator: markdown comparison report from evaluation results."""

from __future__ import annotations

from pathlib import Path

from ..schemas import EvaluationSummary, JudgeScores


RUBRIC_DIMS = [
    "relevance",
    "actionability",
    "evidence_grounding",
    "contradiction_handling",
    "non_redundancy",
]


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def generate_report(summary: EvaluationSummary, output_dir: Path) -> Path:
    """Generate a markdown comparison report and write it to output_dir."""

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "evaluation_report.md"

    lines: list[str] = [
        "# Evaluation Report — Baseline vs Pipeline\n",
        f"**Prompts evaluated:** {len(summary.pairs)}\n",
    ]

    # ── Aggregate table ──
    lines.append("## Aggregate Scores\n")
    lines.append("| Dimension | Baseline Avg | Pipeline Avg | Δ |")
    lines.append("|-----------|:------------:|:------------:|:-:|")

    for dim in RUBRIC_DIMS:
        b_vals = [getattr(p.baseline_scores.scores, dim) for p in summary.pairs]
        p_vals = [getattr(p.pipeline_scores.scores, dim) for p in summary.pairs]
        b_avg = _avg(b_vals)
        p_avg = _avg(p_vals)
        delta = p_avg - b_avg
        sign = "+" if delta > 0 else ""
        lines.append(f"| {dim.replace('_', ' ').title()} | {b_avg:.2f} | {p_avg:.2f} | {sign}{delta:.2f} |")

    # Overall
    b_overall = _avg([
        getattr(p.baseline_scores.scores, d) for p in summary.pairs for d in RUBRIC_DIMS
    ])
    p_overall = _avg([
        getattr(p.pipeline_scores.scores, d) for p in summary.pairs for d in RUBRIC_DIMS
    ])
    delta_overall = p_overall - b_overall
    sign_overall = "+" if delta_overall > 0 else ""
    lines.append(f"| **Overall** | **{b_overall:.2f}** | **{p_overall:.2f}** | **{sign_overall}{delta_overall:.2f}** |")
    lines.append("")

    # ── Per-prompt breakdown ──
    lines.append("## Per-Prompt Breakdown\n")
    for pair in summary.pairs:
        pid = pair.prompt.id
        lines.append(f"### `{pid}`\n")
        lines.append(f"> {pair.prompt.statement}\n")

        lines.append("| Dimension | Baseline | Pipeline |")
        lines.append("|-----------|:--------:|:--------:|")
        for dim in RUBRIC_DIMS:
            b = getattr(pair.baseline_scores.scores, dim)
            p = getattr(pair.pipeline_scores.scores, dim)
            lines.append(f"| {dim.replace('_', ' ').title()} | {b:.1f} | {p:.1f} |")
        lines.append("")

        lines.append(f"**Baseline rationale:** {pair.baseline_scores.scores.rationale}\n")
        lines.append(f"**Pipeline rationale:** {pair.pipeline_scores.scores.rationale}\n")
        lines.append("---\n")

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    return report_path
