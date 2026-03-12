"""LLM Judge: score two system outputs on the 5-dimension rubric."""

from __future__ import annotations

import hashlib
import math
import random
from statistics import NormalDist

from .. import console as con
from ..config import Settings, get_settings
from ..llm_client import call_llm_json
from ..schemas import (
    FocusPoint,
    FounderPrompt,
    JudgePanelResult,
    JudgeResult,
    JudgeScores,
    MetricWithCI,
    SignificanceResult,
    SynthesisResult,
)

RUBRIC_DIMS = (
    "relevance",
    "contradiction",
    "coverage",
    "distinctiveness",
)
PANEL_METRICS = (*RUBRIC_DIMS, "overall_preference")


def _load_prompt_template(settings: Settings) -> str:
    path = settings.prompts_dir / "judge.md"
    return path.read_text(encoding="utf-8")


def _format_focus_points(label: str, points: list[FocusPoint]) -> str:
    lines = [f"=== {label} ==="]
    for i, fp in enumerate(points, start=1):
        lines.append(f"  [{i}] {fp.label}")
        lines.append(f"      Why: {fp.why_it_matters}")
        lines.append(f"      Evidence: {', '.join(fp.supporting_snippets)}")
        lines.append(f"      Counter: {fp.counter_signal}")
        lines.append(f"      Next Q: {fp.next_validation_question}")
    return "\n".join(lines)


def _metric_value(scores: JudgeScores, metric: str) -> float:
    if metric == "overall_preference":
        return scores.overall_preference
    return float(getattr(scores, metric))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _build_aggregate_scores(per_judge_scores: list[JudgeScores]) -> JudgeScores:
    panel_size = len(per_judge_scores)
    return JudgeScores(
        relevance=_mean([s.relevance for s in per_judge_scores]),
        contradiction=_mean([s.contradiction for s in per_judge_scores]),
        coverage=_mean([s.coverage for s in per_judge_scores]),
        distinctiveness=_mean([s.distinctiveness for s in per_judge_scores]),
        overall_preference=_mean([s.overall_preference for s in per_judge_scores]),
        rationale=f"Panel aggregate across {panel_size} judge(s).",
    )


def _compute_metric_ci(metric: str, values: list[float]) -> MetricWithCI:
    n = len(values)
    mean_value = _mean(values)

    if n < 2:
        ci_lower = mean_value
        ci_upper = mean_value
    else:
        variance = sum((value - mean_value) ** 2 for value in values) / (n - 1)
        std = math.sqrt(variance)
        margin = 1.96 * (std / math.sqrt(n))
        ci_lower = mean_value - margin
        ci_upper = mean_value + margin

    return MetricWithCI(
        metric=metric,
        mean=mean_value,
        ci95_lower=ci_lower,
        ci95_upper=ci_upper,
        sample_size=n,
    )


def _compute_significance(
    metric: str,
    left_values: list[float],
    right_values: list[float],
    *,
    left_variant: str,
    right_variant: str,
    alpha: float = 0.05,
) -> SignificanceResult:
    n = min(len(left_values), len(right_values))
    paired_diffs = [l - r for l, r in zip(left_values[:n], right_values[:n], strict=False)]

    if n < 2:
        return SignificanceResult(
            metric=metric,
            p_value=1.0,
            is_significant=False,
            effect_size=None,
            notes=f"Paired significance requires at least 2 judges (observed n={n}).",
        )

    diff_mean = _mean(paired_diffs)
    variance = sum((value - diff_mean) ** 2 for value in paired_diffs) / (n - 1)
    std = math.sqrt(variance)

    if std == 0:
        p_value = 0.0 if diff_mean != 0 else 1.0
        effect_size = None if diff_mean != 0 else 0.0
        return SignificanceResult(
            metric=metric,
            p_value=p_value,
            is_significant=p_value < alpha,
            effect_size=effect_size,
            notes=(
                "All paired deltas were identical "
                f"({left_variant} - {right_variant} = {diff_mean:.3f})."
            ),
        )

    standard_error = std / math.sqrt(n)
    z_score = diff_mean / standard_error
    p_value = 2 * (1 - NormalDist().cdf(abs(z_score)))
    p_value = max(0.0, min(1.0, p_value))

    return SignificanceResult(
        metric=metric,
        p_value=p_value,
        is_significant=p_value < alpha,
        effect_size=diff_mean / std,
        notes=f"Paired comparison {left_variant} vs {right_variant} (n={n}).",
    )


def _compute_metrics_with_ci(per_judge_scores: list[JudgeScores]) -> list[MetricWithCI]:
    rows: list[MetricWithCI] = []
    for metric in PANEL_METRICS:
        metric_values = [_metric_value(scores, metric) for scores in per_judge_scores]
        rows.append(_compute_metric_ci(metric, metric_values))
    return rows


def _compute_panel_significance(
    left_per_judge: list[JudgeScores],
    right_per_judge: list[JudgeScores],
    *,
    left_variant: str,
    right_variant: str,
) -> tuple[list[SignificanceResult], list[SignificanceResult]]:
    left_results: list[SignificanceResult] = []
    right_results: list[SignificanceResult] = []

    for metric in PANEL_METRICS:
        left_values = [_metric_value(scores, metric) for scores in left_per_judge]
        right_values = [_metric_value(scores, metric) for scores in right_per_judge]

        left_results.append(
            _compute_significance(
                metric,
                left_values,
                right_values,
                left_variant=left_variant,
                right_variant=right_variant,
            )
        )
        right_results.append(
            _compute_significance(
                metric,
                right_values,
                left_values,
                left_variant=right_variant,
                right_variant=left_variant,
            )
        )

    return left_results, right_results


def _deterministic_left_first(
    prompt_id: str,
    left_variant: str,
    right_variant: str,
    judge_index: int,
) -> bool:
    seed_material = f"{prompt_id}|{left_variant}|{right_variant}|{judge_index}"
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    return digest[0] % 2 == 0


def _judge_once(
    prompt: FounderPrompt,
    left_result: SynthesisResult,
    right_result: SynthesisResult,
    *,
    left_variant: str,
    right_variant: str,
    settings: Settings,
    system_prompt: str,
    left_first: bool | None = None,
    judge_index: int | None = None,
    panel_size: int | None = None,
) -> tuple[JudgeResult, JudgeResult]:
    if left_first is None:
        left_first = random.random() < 0.5

    suffix = ""
    if judge_index is not None and panel_size is not None:
        suffix = f" [judge {judge_index + 1}/{panel_size}]"

    if left_first:
        a_result, b_result = left_result, right_result
        con.step(
            "judge",
            f"Scoring prompt {prompt.id!r}{suffix} (A={left_variant}, B={right_variant})...",
        )
    else:
        a_result, b_result = right_result, left_result
        con.step(
            "judge",
            f"Scoring prompt {prompt.id!r}{suffix} (A={right_variant}, B={left_variant})...",
        )

    a_block = _format_focus_points("System A", a_result.focus_points)
    b_block = _format_focus_points("System B", b_result.focus_points)

    user_prompt = (
        f"Founder statement:\n{prompt.statement}\n\n"
        f"{a_block}\n\n"
        f"{b_block}\n\n"
        "Score both systems. Return JSON with keys \"system_a\" and \"system_b\", "
        "each containing: relevance, contradiction, coverage, distinctiveness, "
        "overall_preference (all 1-5), and rationale."
    )

    raw = call_llm_json(system_prompt=system_prompt, user_prompt=user_prompt, settings=settings)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict from judge LLM, got {type(raw).__name__}")

    a_scores = JudgeScores.model_validate(raw["system_a"])
    b_scores = JudgeScores.model_validate(raw["system_b"])

    if left_first:
        left_scores, right_scores = a_scores, b_scores
    else:
        left_scores, right_scores = b_scores, a_scores

    left_judge = JudgeResult(
        prompt_id=prompt.id,
        system_variant=left_variant,
        scores=left_scores,
    )
    right_judge = JudgeResult(
        prompt_id=prompt.id,
        system_variant=right_variant,
        scores=right_scores,
    )
    return left_judge, right_judge


def judge_named_pair(
    prompt: FounderPrompt,
    left_result: SynthesisResult,
    right_result: SynthesisResult,
    *,
    left_variant: str,
    right_variant: str,
    settings: Settings | None = None,
) -> tuple[JudgeResult, JudgeResult]:
    """Score two arbitrary system outputs for one founder prompt.

    Randomizes A/B position to remove positional bias, then maps scores back.
    """

    s = settings or get_settings()
    system_prompt = _load_prompt_template(s)
    return _judge_once(
        prompt,
        left_result,
        right_result,
        left_variant=left_variant,
        right_variant=right_variant,
        settings=s,
        system_prompt=system_prompt,
    )


def judge_panel_named_pair(
    prompt: FounderPrompt,
    left_result: SynthesisResult,
    right_result: SynthesisResult,
    *,
    left_variant: str,
    right_variant: str,
    panel_size: int,
    settings: Settings | None = None,
) -> tuple[JudgePanelResult, JudgePanelResult]:
    """Run a deterministic multi-judge panel for two system outputs."""

    if panel_size < 1:
        raise ValueError(f"panel_size must be >= 1, got {panel_size}")

    s = settings or get_settings()
    system_prompt = _load_prompt_template(s)

    left_per_judge: list[JudgeScores] = []
    right_per_judge: list[JudgeScores] = []

    for judge_index in range(panel_size):
        left_first = _deterministic_left_first(
            prompt.id,
            left_variant,
            right_variant,
            judge_index,
        )
        left_judge, right_judge = _judge_once(
            prompt,
            left_result,
            right_result,
            left_variant=left_variant,
            right_variant=right_variant,
            settings=s,
            system_prompt=system_prompt,
            left_first=left_first,
            judge_index=judge_index,
            panel_size=panel_size,
        )
        left_per_judge.append(left_judge.scores)
        right_per_judge.append(right_judge.scores)

    left_significance, right_significance = _compute_panel_significance(
        left_per_judge,
        right_per_judge,
        left_variant=left_variant,
        right_variant=right_variant,
    )

    left_panel = JudgePanelResult(
        prompt_id=prompt.id,
        system_variant=left_variant,
        panel_size=panel_size,
        per_judge_scores=left_per_judge,
        aggregate_scores=_build_aggregate_scores(left_per_judge),
        metrics_with_ci=_compute_metrics_with_ci(left_per_judge),
        significance=left_significance,
    )
    right_panel = JudgePanelResult(
        prompt_id=prompt.id,
        system_variant=right_variant,
        panel_size=panel_size,
        per_judge_scores=right_per_judge,
        aggregate_scores=_build_aggregate_scores(right_per_judge),
        metrics_with_ci=_compute_metrics_with_ci(right_per_judge),
        significance=right_significance,
    )
    return left_panel, right_panel


def judge_pair(
    prompt: FounderPrompt,
    baseline_result: SynthesisResult,
    pipeline_result: SynthesisResult,
    settings: Settings | None = None,
) -> tuple[JudgeResult, JudgeResult]:
    """Score baseline vs pipeline for one founder prompt."""

    return judge_named_pair(
        prompt,
        baseline_result,
        pipeline_result,
        left_variant="baseline",
        right_variant="pipeline",
        settings=settings,
    )


def judge_panel_pair(
    prompt: FounderPrompt,
    baseline_result: SynthesisResult,
    pipeline_result: SynthesisResult,
    *,
    panel_size: int,
    settings: Settings | None = None,
) -> tuple[JudgePanelResult, JudgePanelResult]:
    """Panel-based baseline vs pipeline scorer."""

    return judge_panel_named_pair(
        prompt,
        baseline_result,
        pipeline_result,
        left_variant="baseline",
        right_variant="pipeline",
        panel_size=panel_size,
        settings=settings,
    )

