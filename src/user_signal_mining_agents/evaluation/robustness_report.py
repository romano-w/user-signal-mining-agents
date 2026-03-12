"""Markdown report generator for robustness suite evaluations."""

from __future__ import annotations

from pathlib import Path

from .robustness_runner import RobustnessSuiteSummary


def _worst_delta(dimension_deltas: dict[str, float]) -> float:
    if not dimension_deltas:
        return 0.0
    return min(dimension_deltas.values())


def generate_robustness_report(summary: RobustnessSuiteSummary, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "robustness_report.md"

    lines: list[str] = [
        "# Robustness Evaluation Report\n",
        f"**Suite:** `{summary.suite_id}`",
        f"**Description:** {summary.suite_description}",
        f"**Prompts evaluated ({len(summary.prompt_ids)}):** {', '.join(summary.prompt_ids)}",
        (
            f"**Gate status:** {'PASS' if summary.gate_passed else 'FAIL'} "
            f"({summary.passed_cases}/{summary.total_cases}, {summary.pass_rate:.2%})"
        ),
        (
            "**Thresholds:** "
            f"max_overall_drop={summary.thresholds.max_overall_drop:.2f}, "
            f"max_dimension_drop={summary.thresholds.max_dimension_drop:.2f}, "
            f"min_case_pass_rate={summary.thresholds.min_case_pass_rate:.2%}\n"
        ),
        "## Case Outcomes\n",
        "| Prompt | Case | Family | Delta Overall | Worst Delta | Status |",
        "|---|---|---|:---:|:---:|:---:|",
    ]

    for outcome in summary.outcomes:
        lines.append(
            "| "
            f"{outcome.prompt_id} | {outcome.case_id} | {outcome.family} | "
            f"{outcome.delta_overall:+.2f} | {_worst_delta(outcome.dimension_deltas):+.2f} | "
            f"{'PASS' if outcome.passed else 'FAIL'} |"
        )

    lines.append("")

    if not summary.gate_passed:
        lines.append("## Gate Failures\n")
        for reason in summary.gate_failure_reasons:
            lines.append(f"- {reason}")
        lines.append("")

    failing = [outcome for outcome in summary.outcomes if not outcome.passed]
    if failing:
        lines.append("## Failing Cases\n")
        for outcome in failing:
            lines.append(f"### `{outcome.prompt_id}:{outcome.case_id}`")
            lines.append(f"- **Expected behavior:** {outcome.expected_behavior}")
            lines.append(f"- **Perturbed statement:** {outcome.perturbed_statement}")
            lines.append(
                f"- **Control vs perturbed overall:** "
                f"{outcome.control_scores.overall_preference:.2f} -> {outcome.perturbed_scores.overall_preference:.2f}"
            )
            lines.append(f"- **Failures:** {'; '.join(outcome.failure_reasons)}")
            lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

