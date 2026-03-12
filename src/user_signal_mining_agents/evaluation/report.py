"""Report generator: markdown comparison report from evaluation results."""

from __future__ import annotations

from pathlib import Path

from ..schemas import EvaluationSummary, PromptEvaluationPair


RUBRIC_DIMS = [
    "relevance",
    "actionability",
    "evidence_grounding",
    "contradiction_handling",
    "non_redundancy",
]


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _avg_by_dim(pairs: list[PromptEvaluationPair], *, variant: str) -> dict[str, float]:
    scores = {}
    for dim in RUBRIC_DIMS:
        if variant == "baseline":
            values = [getattr(pair.baseline_scores.scores, dim) for pair in pairs]
        else:
            values = [getattr(pair.pipeline_scores.scores, dim) for pair in pairs]
        scores[dim] = _avg(values)
    return scores


def _overall_from_dim_scores(dim_scores: dict[str, float]) -> float:
    return _avg(list(dim_scores.values()))


def _group_pairs_by_domain(summary: EvaluationSummary) -> dict[str, list[PromptEvaluationPair]]:
    grouped: dict[str, list[PromptEvaluationPair]] = {}
    for pair in summary.pairs:
        grouped.setdefault(pair.prompt.domain, []).append(pair)
    return grouped


def generate_report(summary: EvaluationSummary, output_dir: Path) -> Path:
    """Generate a markdown comparison report and write it to output_dir."""

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "evaluation_report.md"

    lines: list[str] = [
        "# Evaluation Report - Baseline vs Pipeline\n",
        f"**Prompts evaluated:** {len(summary.pairs)}\n",
    ]

    # Aggregate table
    lines.append("## Aggregate Scores\n")
    lines.append("| Dimension | Baseline Avg | Pipeline Avg | Delta |")
    lines.append("|-----------|:------------:|:------------:|:-:|")

    for dim in RUBRIC_DIMS:
        b_vals = [getattr(p.baseline_scores.scores, dim) for p in summary.pairs]
        p_vals = [getattr(p.pipeline_scores.scores, dim) for p in summary.pairs]
        b_avg = _avg(b_vals)
        p_avg = _avg(p_vals)
        delta = p_avg - b_avg
        sign = "+" if delta > 0 else ""
        lines.append(f"| {dim.replace('_', ' ').title()} | {b_avg:.2f} | {p_avg:.2f} | {sign}{delta:.2f} |")

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

    # Domain quality breakdown
    domain_groups = _group_pairs_by_domain(summary)
    domain_order = list(domain_groups.keys())

    lines.append("## Domain Quality Breakdown\n")
    for domain in domain_order:
        domain_pairs = domain_groups[domain]
        b_scores = _avg_by_dim(domain_pairs, variant="baseline")
        p_scores = _avg_by_dim(domain_pairs, variant="pipeline")
        b_domain_overall = _overall_from_dim_scores(b_scores)
        p_domain_overall = _overall_from_dim_scores(p_scores)
        domain_delta = p_domain_overall - b_domain_overall
        sign_domain = "+" if domain_delta > 0 else ""

        lines.append(f"### `{domain}` ({len(domain_pairs)} prompt(s))")
        lines.append("| Dimension | Baseline Avg | Pipeline Avg | Delta |")
        lines.append("|-----------|:------------:|:------------:|:-:|")
        for dim in RUBRIC_DIMS:
            b_val = b_scores[dim]
            p_val = p_scores[dim]
            delta = p_val - b_val
            sign = "+" if delta > 0 else ""
            lines.append(f"| {dim.replace('_', ' ').title()} | {b_val:.2f} | {p_val:.2f} | {sign}{delta:.2f} |")
        lines.append(
            f"| **Overall** | **{b_domain_overall:.2f}** | **{p_domain_overall:.2f}** | **{sign_domain}{domain_delta:.2f}** |"
        )
        lines.append("")

    # Domain transfer deltas
    lines.append("## Domain Transfer Deltas\n")
    if len(domain_order) <= 1:
        lines.append("Only one domain evaluated; transfer deltas require at least two domains.\n")
    else:
        reference_domain = domain_order[0]
        ref_pairs = domain_groups[reference_domain]
        ref_b = _overall_from_dim_scores(_avg_by_dim(ref_pairs, variant="baseline"))
        ref_p = _overall_from_dim_scores(_avg_by_dim(ref_pairs, variant="pipeline"))
        ref_gain = ref_p - ref_b

        lines.append(f"Reference domain: `{reference_domain}`\n")
        lines.append(
            "| Domain | Baseline Overall | Pipeline Overall | In-Domain Delta | "
            "Pipeline Transfer Delta vs Reference | Improvement Transfer Delta vs Reference |"
        )
        lines.append("|--------|:----------------:|:----------------:|:-----------:|:--------------------------------:|:-----------------------------------:|")

        for domain in domain_order:
            domain_pairs = domain_groups[domain]
            b_overall_domain = _overall_from_dim_scores(_avg_by_dim(domain_pairs, variant="baseline"))
            p_overall_domain = _overall_from_dim_scores(_avg_by_dim(domain_pairs, variant="pipeline"))
            gain = p_overall_domain - b_overall_domain
            pipeline_transfer = p_overall_domain - ref_p
            gain_transfer = gain - ref_gain

            gain_sign = "+" if gain > 0 else ""
            pipeline_transfer_sign = "+" if pipeline_transfer > 0 else ""
            gain_transfer_sign = "+" if gain_transfer > 0 else ""
            lines.append(
                f"| {domain} | {b_overall_domain:.2f} | {p_overall_domain:.2f} | {gain_sign}{gain:.2f} | "
                f"{pipeline_transfer_sign}{pipeline_transfer:.2f} | {gain_transfer_sign}{gain_transfer:.2f} |"
            )
        lines.append("")

    # Per-prompt breakdown
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

