from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from ..schemas import HumanAnnotationResult, HumanAnnotationTask

RUBRIC_DIMS = ("relevance", "groundedness", "distinctiveness")
PREFERENCE_LABELS = ("system_a", "system_b", "tie")
RATING_LABELS = (1, 2, 3, 4, 5)


class AnnotationExportInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    annotator_id: str
    path: str
    result_count: int = Field(ge=0)


class AgreementMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str
    sample_size: int = Field(ge=0)
    exact_agreement: float = Field(ge=0.0, le=1.0)
    cohen_kappa: float
    mean_abs_diff: float | None = Field(default=None, ge=0.0)
    quadratic_weighted_kappa: float | None = None


class JudgeAlignmentSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    annotator_id: str
    sample_size: int = Field(ge=0)
    exact_agreement: float = Field(ge=0.0, le=1.0)
    cohen_kappa: float
    human_preference_counts: dict[str, int] = Field(default_factory=dict)
    judge_preference_counts: dict[str, int] = Field(default_factory=dict)


class HumanAnnotationAnalysisSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tasks_dir: str
    runs_dir: str
    exports: list[AnnotationExportInfo] = Field(default_factory=list)
    overlapping_task_ids: list[str] = Field(default_factory=list)
    only_in_export_a: list[str] = Field(default_factory=list)
    only_in_export_b: list[str] = Field(default_factory=list)
    interannotator_overall_preference: AgreementMetric | None = None
    interannotator_dimensions: list[AgreementMetric] = Field(default_factory=list)
    judge_alignment: list[JudgeAlignmentSummary] = Field(default_factory=list)
    missing_task_ids: list[str] = Field(default_factory=list)
    missing_judge_prompt_ids: list[str] = Field(default_factory=list)


def _count_labels(values: list[str], labels: tuple[str, ...]) -> dict[str, int]:
    return {label: sum(1 for value in values if value == label) for label in labels}


def _cohen_kappa(left: list[Any], right: list[Any], labels: tuple[Any, ...]) -> float:
    n = len(left)
    if n == 0:
        return 0.0

    observed = sum(1 for l_value, r_value in zip(left, right, strict=False) if l_value == r_value) / n
    left_counts = [sum(1 for value in left if value == label) for label in labels]
    right_counts = [sum(1 for value in right if value == label) for label in labels]
    expected = sum(l_count * r_count for l_count, r_count in zip(left_counts, right_counts, strict=False)) / (n * n)

    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def _quadratic_weighted_kappa(left: list[int], right: list[int], labels: tuple[int, ...]) -> float:
    n = len(left)
    if n == 0:
        return 0.0

    label_to_index = {label: index for index, label in enumerate(labels)}
    size = len(labels)
    observed = [[0.0 for _ in range(size)] for _ in range(size)]
    left_counts = [0.0 for _ in range(size)]
    right_counts = [0.0 for _ in range(size)]

    for l_value, r_value in zip(left, right, strict=False):
        i = label_to_index[l_value]
        j = label_to_index[r_value]
        observed[i][j] += 1.0
        left_counts[i] += 1.0
        right_counts[j] += 1.0

    observed = [[cell / n for cell in row] for row in observed]
    left_probs = [count / n for count in left_counts]
    right_probs = [count / n for count in right_counts]

    denominator = 0.0
    numerator = 0.0
    scale = max(size - 1, 1)
    for i in range(size):
        for j in range(size):
            weight = ((i - j) ** 2) / (scale ** 2)
            expected = left_probs[i] * right_probs[j]
            numerator += weight * observed[i][j]
            denominator += weight * expected

    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else 0.0
    return 1.0 - (numerator / denominator)


def _load_annotation_export(path: Path) -> tuple[AnnotationExportInfo, dict[str, HumanAnnotationResult]]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    annotator_id = path.stem
    if isinstance(payload, dict):
        if isinstance(payload.get("annotator_id"), str) and payload["annotator_id"].strip():
            annotator_id = payload["annotator_id"].strip()
        raw_results = payload.get("results", [])
    else:
        raw_results = payload

    if not isinstance(raw_results, list):
        raise ValueError(f"annotation export must contain a list of results: {path}")

    results = TypeAdapter(list[HumanAnnotationResult]).validate_python(raw_results)
    result_map = {result.task_id: result for result in results}
    return (
        AnnotationExportInfo(
            annotator_id=annotator_id,
            path=str(path),
            result_count=len(results),
        ),
        result_map,
    )


def _load_task(task_id: str, tasks_dir: Path) -> HumanAnnotationTask | None:
    path = tasks_dir / f"{task_id}.json"
    if not path.exists():
        return None
    return HumanAnnotationTask.model_validate_json(path.read_text(encoding="utf-8"))


def _load_judge_preference(prompt_id: str, runs_dir: Path) -> str | None:
    run_dir = runs_dir / prompt_id
    baseline_path = run_dir / "judge_baseline.json"
    pipeline_path = run_dir / "judge_pipeline.json"
    if not baseline_path.exists() or not pipeline_path.exists():
        return None

    def _overall(path: Path) -> float | None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        scores = payload.get("scores")
        if not isinstance(scores, dict):
            return None
        value = scores.get("overall_preference")
        return float(value) if isinstance(value, int | float) else None

    baseline_overall = _overall(baseline_path)
    pipeline_overall = _overall(pipeline_path)
    if baseline_overall is None or pipeline_overall is None:
        return None
    if pipeline_overall > baseline_overall:
        return "pipeline"
    if pipeline_overall < baseline_overall:
        return "baseline"
    return "tie"


def _map_judge_preference_to_blinded(task: HumanAnnotationTask, judge_preference: str) -> str:
    if judge_preference == "tie":
        return "tie"
    for blinded_label, variant in task.ground_truth_mapping.items():
        if variant == judge_preference:
            return blinded_label
    raise ValueError(f"task {task.task_id!r} does not map judge preference {judge_preference!r}")


def _dimension_agreement(metric: str, left: list[int], right: list[int]) -> AgreementMetric:
    sample_size = len(left)
    exact = sum(1 for l_value, r_value in zip(left, right, strict=False) if l_value == r_value) / sample_size if sample_size else 0.0
    mean_abs_diff = sum(abs(l_value - r_value) for l_value, r_value in zip(left, right, strict=False)) / sample_size if sample_size else 0.0
    return AgreementMetric(
        metric=metric,
        sample_size=sample_size,
        exact_agreement=exact,
        cohen_kappa=_cohen_kappa(left, right, RATING_LABELS),
        mean_abs_diff=mean_abs_diff,
        quadratic_weighted_kappa=_quadratic_weighted_kappa(left, right, RATING_LABELS),
    )


def _overall_preference_agreement(metric: str, left: list[str], right: list[str]) -> AgreementMetric:
    sample_size = len(left)
    exact = sum(1 for l_value, r_value in zip(left, right, strict=False) if l_value == r_value) / sample_size if sample_size else 0.0
    return AgreementMetric(
        metric=metric,
        sample_size=sample_size,
        exact_agreement=exact,
        cohen_kappa=_cohen_kappa(left, right, PREFERENCE_LABELS),
    )


def analyze_human_annotations(
    export_a_path: Path,
    *,
    export_b_path: Path | None = None,
    tasks_dir: Path,
    runs_dir: Path,
) -> HumanAnnotationAnalysisSummary:
    export_a, results_a = _load_annotation_export(export_a_path)
    exports = [export_a]

    overlapping_task_ids: list[str] = []
    only_in_export_a: list[str] = sorted(results_a)
    only_in_export_b: list[str] = []
    interannotator_overall: AgreementMetric | None = None
    interannotator_dimensions: list[AgreementMetric] = []

    if export_b_path is not None:
        export_b, results_b = _load_annotation_export(export_b_path)
        exports.append(export_b)
        task_ids_a = set(results_a)
        task_ids_b = set(results_b)
        overlapping_task_ids = sorted(task_ids_a & task_ids_b)
        only_in_export_a = sorted(task_ids_a - task_ids_b)
        only_in_export_b = sorted(task_ids_b - task_ids_a)

        left_overall = [results_a[task_id].overall_preference for task_id in overlapping_task_ids]
        right_overall = [results_b[task_id].overall_preference for task_id in overlapping_task_ids]
        interannotator_overall = _overall_preference_agreement(
            "overall_preference",
            left_overall,
            right_overall,
        )

        for metric in RUBRIC_DIMS:
            left_scores: list[int] = []
            right_scores: list[int] = []
            for task_id in overlapping_task_ids:
                result_a = results_a[task_id]
                result_b = results_b[task_id]
                left_scores.extend([
                    int(getattr(result_a.system_a_scores, metric)),
                    int(getattr(result_a.system_b_scores, metric)),
                ])
                right_scores.extend([
                    int(getattr(result_b.system_a_scores, metric)),
                    int(getattr(result_b.system_b_scores, metric)),
                ])
            interannotator_dimensions.append(_dimension_agreement(metric, left_scores, right_scores))

    missing_task_ids: set[str] = set()
    missing_judge_prompt_ids: set[str] = set()
    judge_alignment: list[JudgeAlignmentSummary] = []

    for export_info, result_map in ((export, results_a) for export in exports[:1]) if export_b_path is None else ((exports[0], results_a), (exports[1], results_b)):
        human_labels: list[str] = []
        judge_labels: list[str] = []
        for task_id, result in sorted(result_map.items()):
            task = _load_task(task_id, tasks_dir)
            if task is None:
                missing_task_ids.add(task_id)
                continue
            judge_preference = _load_judge_preference(task.prompt.id, runs_dir)
            if judge_preference is None:
                missing_judge_prompt_ids.add(task.prompt.id)
                continue
            human_labels.append(result.overall_preference)
            judge_labels.append(_map_judge_preference_to_blinded(task, judge_preference))

        judge_alignment.append(
            JudgeAlignmentSummary(
                annotator_id=export_info.annotator_id,
                sample_size=len(human_labels),
                exact_agreement=(sum(1 for human, judge in zip(human_labels, judge_labels, strict=False) if human == judge) / len(human_labels)) if human_labels else 0.0,
                cohen_kappa=_cohen_kappa(human_labels, judge_labels, PREFERENCE_LABELS),
                human_preference_counts=_count_labels(human_labels, PREFERENCE_LABELS),
                judge_preference_counts=_count_labels(judge_labels, PREFERENCE_LABELS),
            )
        )

    return HumanAnnotationAnalysisSummary(
        tasks_dir=str(tasks_dir),
        runs_dir=str(runs_dir),
        exports=exports,
        overlapping_task_ids=overlapping_task_ids,
        only_in_export_a=only_in_export_a,
        only_in_export_b=only_in_export_b,
        interannotator_overall_preference=interannotator_overall,
        interannotator_dimensions=interannotator_dimensions,
        judge_alignment=judge_alignment,
        missing_task_ids=sorted(missing_task_ids),
        missing_judge_prompt_ids=sorted(missing_judge_prompt_ids),
    )


def write_human_annotation_report(
    summary: HumanAnnotationAnalysisSummary,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "human_annotation_summary.json"
    markdown_path = output_dir / "human_annotation_report.md"

    json_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    lines: list[str] = [
        "# Human Annotation Analysis\n",
        f"**Generated:** {summary.generated_at.isoformat()}",
        f"**Tasks dir:** `{summary.tasks_dir}`",
        f"**Runs dir:** `{summary.runs_dir}`",
        "",
        "## Exports",
        "| Annotator | Results | Path |",
        "|---|---:|---|",
    ]
    for export in summary.exports:
        lines.append(f"| {export.annotator_id} | {export.result_count} | `{export.path}` |")

    lines.append("")
    if summary.interannotator_overall_preference is not None:
        overall = summary.interannotator_overall_preference
        lines.extend([
            "## Interannotator Agreement",
            f"**Overlapping tasks:** {len(summary.overlapping_task_ids)}",
            "",
            "### Overall Preference",
            "| Metric | Samples | Exact Agreement | Cohen's Kappa |",
            "|---|---:|---:|---:|",
            f"| {overall.metric} | {overall.sample_size} | {overall.exact_agreement:.2%} | {overall.cohen_kappa:.3f} |",
            "",
            "### Rubric Dimensions",
            "| Dimension | Samples | Exact Agreement | Mean Abs Diff | Quadratic Weighted Kappa |",
            "|---|---:|---:|---:|---:|",
        ])
        for metric in summary.interannotator_dimensions:
            weighted = "n/a" if metric.quadratic_weighted_kappa is None else f"{metric.quadratic_weighted_kappa:.3f}"
            mean_abs = "n/a" if metric.mean_abs_diff is None else f"{metric.mean_abs_diff:.3f}"
            lines.append(
                f"| {metric.metric} | {metric.sample_size} | {metric.exact_agreement:.2%} | {mean_abs} | {weighted} |"
            )
        lines.append("")

    lines.extend([
        "## Judge Alignment",
        "| Annotator | Samples | Exact Agreement | Cohen's Kappa | Judge A | Judge B | Judge Tie |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary.judge_alignment:
        counts = row.judge_preference_counts
        lines.append(
            f"| {row.annotator_id} | {row.sample_size} | {row.exact_agreement:.2%} | {row.cohen_kappa:.3f} | "
            f"{counts.get('system_a', 0)} | {counts.get('system_b', 0)} | {counts.get('tie', 0)} |"
        )

    if summary.only_in_export_a or summary.only_in_export_b or summary.missing_task_ids or summary.missing_judge_prompt_ids:
        lines.append("")
        lines.append("## Missing or Unmatched Inputs")
        if summary.only_in_export_a:
            lines.append(f"- Only in export A: {', '.join(summary.only_in_export_a)}")
        if summary.only_in_export_b:
            lines.append(f"- Only in export B: {', '.join(summary.only_in_export_b)}")
        if summary.missing_task_ids:
            lines.append(f"- Missing task files: {', '.join(summary.missing_task_ids)}")
        if summary.missing_judge_prompt_ids:
            lines.append(f"- Missing judge artifacts: {', '.join(summary.missing_judge_prompt_ids)}")

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def analyze_and_write_human_annotation_report(
    export_a_path: Path,
    *,
    export_b_path: Path | None = None,
    tasks_dir: Path,
    runs_dir: Path,
    output_dir: Path,
) -> tuple[HumanAnnotationAnalysisSummary, Path, Path]:
    summary = analyze_human_annotations(
        export_a_path,
        export_b_path=export_b_path,
        tasks_dir=tasks_dir,
        runs_dir=runs_dir,
    )
    json_path, markdown_path = write_human_annotation_report(summary, output_dir)
    return summary, json_path, markdown_path
