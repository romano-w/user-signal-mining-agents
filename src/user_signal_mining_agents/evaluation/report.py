"""Report generator: markdown comparison report from evaluation results."""

from __future__ import annotations

from pathlib import Path

from ..schemas import (
    EvaluationSummary,
    JudgePanelResult,
    MetricWithCI,
    PromptEvaluationPair,
    SignificanceResult,
)


RUBRIC_DIMS = [
    "relevance",
    "contradiction",
    "coverage",
    "distinctiveness",
]


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _metric_label(metric: str) -> str:
    if metric == "overall_preference":
        return "Overall Preference"
    return metric.replace("_", " ").title()


def _find_ci(panel: JudgePanelResult, metric: str) -> MetricWithCI | None:
    for entry in panel.metrics_with_ci:
        if entry.metric == metric:
            return entry
    return None


def _find_significance(panel: JudgePanelResult, metric: str) -> SignificanceResult | None:
    for entry in panel.significance:
        if entry.metric == metric:
            return entry
    return None


def _format_ci(entry: MetricWithCI | None) -> str:
    if entry is None:
        return "n/a"
    return f"{entry.mean:.2f} [{entry.ci95_lower:.2f}, {entry.ci95_upper:.2f}]"


def _avg_by_dim(pairs: list[PromptEvaluationPair], *, variant: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for dim in RUBRIC_DIMS:
        if variant == "baseline":
            values = [getattr(pair.baseline_scores.scores, dim) for pair in pairs]
        else:
            values = [getattr(pair.pipeline_scores.scores, dim) for pair in pairs]
        scores[dim] = _avg(values)
    return scores


def _avg_overall_preference(pairs: list[PromptEvaluationPair], *, variant: str) -> float:
    if variant == "baseline":
        return _avg([pair.baseline_scores.scores.overall_preference for pair in pairs])
    return _avg([pair.pipeline_scores.scores.overall_preference for pair in pairs])


def _group_pairs_by_domain(summary: EvaluationSummary) -> dict[str, list[PromptEvaluationPair]]:
    grouped: dict[str, list[PromptEvaluationPair]] = {}
    for pair in summary.pairs:
        grouped.setdefault(pair.prompt.domain, []).append(pair)
    return grouped


def generate_report(summary: EvaluationSummary, output_dir: Path) -> Path:
    """Generate a markdown comparison report and write it to output_dir."""

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "evaluation_report.md"

    panel_pairs = [pair for pair in summary.pairs if pair.baseline_panel and pair.pipeline_panel]

    lines: list[str] = [
        "# Evaluation Report - Baseline vs Pipeline\n",
        f"**Prompts evaluated:** {len(summary.pairs)}\n",
    ]

    if panel_pairs:
        panel_sizes = sorted({pair.baseline_panel.panel_size for pair in panel_pairs})
        if len(panel_sizes) == 1:
            lines.append(f"**Judge panel mode:** enabled ({panel_sizes[0]} judges per prompt)\n")
        else:
            labels = ", ".join(str(size) for size in panel_sizes)
            lines.append(f"**Judge panel mode:** enabled (sizes: {labels})\n")
        lines.append("**Confidence context:** 95% confidence intervals and paired significance are included per prompt.\n")
    else:
        lines.append("**Judge panel mode:** disabled (single judge).\n")

    lines.append("## Aggregate Scores\n")
    lines.append("| Dimension | Baseline Avg | Pipeline Avg | Delta |")
    lines.append("|-----------|:------------:|:------------:|:-:|")

    for dim in [*RUBRIC_DIMS, "overall_preference"]:
        b_vals = [getattr(p.baseline_scores.scores, dim) for p in summary.pairs]
        p_vals = [getattr(p.pipeline_scores.scores, dim) for p in summary.pairs]
        b_avg = _avg(b_vals)
        p_avg = _avg(p_vals)
        delta = p_avg - b_avg
        sign = "+" if delta > 0 else ""
        lines.append(f"| {_metric_label(dim)} | {b_avg:.2f} | {p_avg:.2f} | {sign}{delta:.2f} |")
    lines.append("")

    domain_groups = _group_pairs_by_domain(summary)
    domain_order = list(domain_groups.keys())

    lines.append("## Domain Quality Breakdown\n")
    for domain in domain_order:
        domain_pairs = domain_groups[domain]
        b_scores = _avg_by_dim(domain_pairs, variant="baseline")
        p_scores = _avg_by_dim(domain_pairs, variant="pipeline")
        b_domain_overall = _avg_overall_preference(domain_pairs, variant="baseline")
        p_domain_overall = _avg_overall_preference(domain_pairs, variant="pipeline")
        domain_delta = p_domain_overall - b_domain_overall
        sign_domain = "+" if domain_delta > 0 else ""

        lines.append(f"### `{domain}` ({len(domain_pairs)} prompt(s))")
        lines.append("| Dimension | Baseline Avg | Pipeline Avg | Delta |")
        lines.append("|-----------|:------------:|:------------:|:-:|")
        for dim in [*RUBRIC_DIMS, "overall_preference"]:
            if dim == "overall_preference":
                b_val = b_domain_overall
                p_val = p_domain_overall
            else:
                b_val = b_scores[dim]
                p_val = p_scores[dim]
            delta = p_val - b_val
            sign = "+" if delta > 0 else ""
            lines.append(f"| {_metric_label(dim)} | {b_val:.2f} | {p_val:.2f} | {sign}{delta:.2f} |")
        lines.append("")

    lines.append("## Domain Transfer Deltas\n")
    if len(domain_order) <= 1:
        lines.append("Only one domain evaluated; transfer deltas require at least two domains.\n")
    else:
        reference_domain = domain_order[0]
        ref_pairs = domain_groups[reference_domain]
        ref_b = _avg_overall_preference(ref_pairs, variant="baseline")
        ref_p = _avg_overall_preference(ref_pairs, variant="pipeline")
        ref_gain = ref_p - ref_b

        lines.append(f"Reference domain: `{reference_domain}`\n")
        lines.append(
            "| Domain | Baseline Overall | Pipeline Overall | In-Domain Delta | "
            "Pipeline Transfer Delta vs Reference | Improvement Transfer Delta vs Reference |"
        )
        lines.append("|--------|:----------------:|:----------------:|:-----------:|:--------------------------------:|:-----------------------------------:|")

        for domain in domain_order:
            domain_pairs = domain_groups[domain]
            b_overall_domain = _avg_overall_preference(domain_pairs, variant="baseline")
            p_overall_domain = _avg_overall_preference(domain_pairs, variant="pipeline")
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

    lines.append("## Per-Prompt Breakdown\n")
    for pair in summary.pairs:
        pid = pair.prompt.id
        lines.append(f"### `{pid}`\n")
        lines.append(f"> {pair.prompt.statement}\n")

        lines.append("| Dimension | Baseline | Pipeline |")
        lines.append("|-----------|:--------:|:--------:|")
        for dim in [*RUBRIC_DIMS, "overall_preference"]:
            b = getattr(pair.baseline_scores.scores, dim)
            p = getattr(pair.pipeline_scores.scores, dim)
            lines.append(f"| {_metric_label(dim)} | {b:.1f} | {p:.1f} |")
        lines.append("")

        if pair.baseline_panel and pair.pipeline_panel:
            panel_size = pair.pipeline_panel.panel_size
            lines.append(f"#### Panel Confidence ({panel_size} judges)\n")
            lines.append("| Metric | Baseline Mean [95% CI] | Pipeline Mean [95% CI] | p-value (Pipeline vs Baseline) |")
            lines.append("|--------|------------------------|------------------------|:------------------------------:|")

            for metric in [*RUBRIC_DIMS, "overall_preference"]:
                baseline_ci = _find_ci(pair.baseline_panel, metric)
                pipeline_ci = _find_ci(pair.pipeline_panel, metric)
                sig = _find_significance(pair.pipeline_panel, metric)
                p_value = f"{sig.p_value:.4f}" if sig is not None else "n/a"
                lines.append(
                    f"| {_metric_label(metric)} | {_format_ci(baseline_ci)} | {_format_ci(pipeline_ci)} | {p_value} |"
                )
            lines.append("")

        lines.append(f"**Baseline rationale:** {pair.baseline_scores.scores.rationale}\n")
        lines.append(f"**Pipeline rationale:** {pair.pipeline_scores.scores.rationale}\n")
        lines.append("---\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

