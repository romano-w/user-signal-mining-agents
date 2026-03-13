from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..schemas import JudgeResult


RUBRIC_DIMS = [
    "relevance",
    "groundedness",
    "distinctiveness",
]


@dataclass(frozen=True)
class MetricDelta:
    metric: str
    baseline_avg: float
    pipeline_avg: float
    delta: float


@dataclass(frozen=True)
class RegressionViolation:
    metric: str
    baseline_avg: float
    pipeline_avg: float
    delta: float
    max_drop: float


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_judge_pairs(runs_dir: Path) -> list[tuple[JudgeResult, JudgeResult]]:
    pairs: list[tuple[JudgeResult, JudgeResult]] = []
    if not runs_dir.exists():
        return pairs

    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        baseline_path = run_dir / "judge_baseline.json"
        pipeline_path = run_dir / "judge_pipeline.json"
        if not baseline_path.exists() or not pipeline_path.exists():
            continue

        baseline = JudgeResult.model_validate_json(baseline_path.read_text(encoding="utf-8"))
        pipeline = JudgeResult.model_validate_json(pipeline_path.read_text(encoding="utf-8"))
        pairs.append((baseline, pipeline))

    return pairs


def summarize_metric_deltas(pairs: list[tuple[JudgeResult, JudgeResult]]) -> list[MetricDelta]:
    if not pairs:
        return []

    deltas: list[MetricDelta] = []
    for dim in RUBRIC_DIMS:
        baseline_avg = _avg([getattr(baseline.scores, dim) for baseline, _ in pairs])
        pipeline_avg = _avg([getattr(pipeline.scores, dim) for _, pipeline in pairs])
        deltas.append(
            MetricDelta(
                metric=dim,
                baseline_avg=baseline_avg,
                pipeline_avg=pipeline_avg,
                delta=pipeline_avg - baseline_avg,
            )
        )

    overall_baseline = _avg([baseline.scores.overall_preference for baseline, _ in pairs])
    overall_pipeline = _avg([pipeline.scores.overall_preference for _, pipeline in pairs])
    deltas.append(
        MetricDelta(
            metric="overall_preference",
            baseline_avg=overall_baseline,
            pipeline_avg=overall_pipeline,
            delta=overall_pipeline - overall_baseline,
        )
    )
    return deltas


def find_critical_metric_regressions(
    pairs: list[tuple[JudgeResult, JudgeResult]],
    *,
    max_overall_drop: float = 0.30,
    max_dimension_drop: float = 0.40,
) -> list[RegressionViolation]:
    violations: list[RegressionViolation] = []
    for metric in summarize_metric_deltas(pairs):
        allowed_drop = max_overall_drop if metric.metric == "overall_preference" else max_dimension_drop
        if metric.delta < -allowed_drop:
            violations.append(
                RegressionViolation(
                    metric=metric.metric,
                    baseline_avg=metric.baseline_avg,
                    pipeline_avg=metric.pipeline_avg,
                    delta=metric.delta,
                    max_drop=allowed_drop,
                )
            )
    return violations

