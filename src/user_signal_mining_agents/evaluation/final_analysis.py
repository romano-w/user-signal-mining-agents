"""Final analysis report generator for completed experiment artifacts."""

from __future__ import annotations

import html
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


CURRENT_METRICS = ("relevance", "groundedness", "distinctiveness", "overall_preference")
LEGACY_JUDGE_KEYS = ("actionability", "evidence_grounding", "coverage", "contradiction", "non_redundancy")
LEGACY_ANNOTATION_KEYS = ("coverage", "contradiction")
AUTO_RETRIEVAL_SUMMARY_PATHS = (
    Path("reports/research_upgrade/retrieval_eval_summary.json"),
    Path("artifacts/retrieval_eval/retrieval_eval_summary.json"),
)


class ScoreSnapshot(BaseModel):
    relevance: float
    groundedness: float
    distinctiveness: float
    overall_preference: float


class PromptOutcome(BaseModel):
    prompt_id: str
    domain: str
    statement: str
    baseline: ScoreSnapshot
    pipeline: ScoreSnapshot
    delta: ScoreSnapshot
    winner: Literal["baseline", "pipeline", "tie"]


class DomainAggregate(BaseModel):
    domain: str
    prompt_count: int = Field(ge=0)
    pipeline_wins: int = Field(ge=0)
    baseline_wins: int = Field(ge=0)
    ties: int = Field(ge=0)
    baseline: ScoreSnapshot
    pipeline: ScoreSnapshot
    delta: ScoreSnapshot


class FailureCategorySummary(BaseModel):
    category: str
    total_count: int = Field(ge=0)
    baseline_count: int = Field(ge=0)
    pipeline_count: int = Field(ge=0)
    avg_severity: float = Field(ge=0.0)
    max_severity: int = Field(ge=0)


class FailurePromptSummary(BaseModel):
    prompt_id: str
    tag_count: int = Field(ge=0)
    max_severity: int = Field(ge=0)
    variants: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)


class RetrievalSummarySnapshot(BaseModel):
    path: str
    query_count: int = Field(ge=0)
    retrieval_mode: str
    reranker: str
    k_values: list[int] = Field(default_factory=list)
    recall_at_k: dict[str, float] = Field(default_factory=dict)
    mrr_at_k: dict[str, float] = Field(default_factory=dict)
    ndcg_at_k: dict[str, float] = Field(default_factory=dict)


class SweepAggregate(BaseModel):
    variant: str
    prompt_count: int = Field(ge=0)
    scores: ScoreSnapshot


class SweepSummary(BaseModel):
    status: Literal["missing", "current", "legacy", "mixed"]
    note: str
    variants_found: list[str] = Field(default_factory=list)
    aggregates: list[SweepAggregate] = Field(default_factory=list)


class AnnotationAnnotatorProgress(BaseModel):
    annotator_id: str
    autosave_current_count: int = Field(ge=0)
    autosave_legacy_count: int = Field(ge=0)
    export_current_count: int = Field(ge=0)
    export_legacy_count: int = Field(ge=0)


class AnnotationProgressSummary(BaseModel):
    total_tasks: int = Field(ge=0)
    current_completed_tasks: int = Field(ge=0)
    legacy_completed_tasks: int = Field(ge=0)
    export_file_count: int = Field(ge=0)
    annotators: list[AnnotationAnnotatorProgress] = Field(default_factory=list)
    note: str


class ArtifactStatus(BaseModel):
    family: str
    status: Literal["complete", "partial", "missing", "excluded"]
    detail: str


class FinalAnalysisSummary(BaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    runs_dir: str
    output_dir: str
    prompt_count: int = Field(ge=0)
    judge_panel_mode: str
    pipeline_wins: int = Field(ge=0)
    baseline_wins: int = Field(ge=0)
    ties: int = Field(ge=0)
    baseline: ScoreSnapshot
    pipeline: ScoreSnapshot
    delta: ScoreSnapshot
    domains: list[DomainAggregate] = Field(default_factory=list)
    prompt_outcomes: list[PromptOutcome] = Field(default_factory=list)
    failure_categories: list[FailureCategorySummary] = Field(default_factory=list)
    top_failure_prompts: list[FailurePromptSummary] = Field(default_factory=list)
    retrieval: RetrievalSummarySnapshot | None = None
    sweep: SweepSummary
    annotation: AnnotationProgressSummary
    artifact_statuses: list[ArtifactStatus] = Field(default_factory=list)
    figure_paths: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_scores(rows: list[ScoreSnapshot]) -> ScoreSnapshot:
    return ScoreSnapshot(
        relevance=_mean([row.relevance for row in rows]),
        groundedness=_mean([row.groundedness for row in rows]),
        distinctiveness=_mean([row.distinctiveness for row in rows]),
        overall_preference=_mean([row.overall_preference for row in rows]),
    )


def _delta_scores(left: ScoreSnapshot, right: ScoreSnapshot) -> ScoreSnapshot:
    return ScoreSnapshot(
        relevance=right.relevance - left.relevance,
        groundedness=right.groundedness - left.groundedness,
        distinctiveness=right.distinctiveness - left.distinctiveness,
        overall_preference=right.overall_preference - left.overall_preference,
    )


def _metric_label(metric: str) -> str:
    if metric == "overall_preference":
        return "Overall Preference"
    return metric.replace("_", " ").title()


def _current_score_payload(payload: dict[str, object]) -> ScoreSnapshot | None:
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        return None

    relevance = scores.get("relevance")
    overall_preference = scores.get("overall_preference")
    distinctiveness = scores.get("distinctiveness", scores.get("non_redundancy"))
    groundedness = scores.get("groundedness")

    if not isinstance(groundedness, int | float):
        legacy_groundedness_values = [
            float(value)
            for key in ("coverage", "contradiction", "evidence_grounding")
            if isinstance((value := scores.get(key)), int | float)
        ]
        if legacy_groundedness_values:
            groundedness = sum(legacy_groundedness_values) / len(legacy_groundedness_values)

    if not all(isinstance(value, int | float) for value in (relevance, groundedness, distinctiveness, overall_preference)):
        return None

    return ScoreSnapshot(
        relevance=float(relevance),
        groundedness=float(groundedness),
        distinctiveness=float(distinctiveness),
        overall_preference=float(overall_preference),
    )


def _judge_artifact_status(path: Path) -> tuple[Literal["current", "legacy", "invalid"], ScoreSnapshot | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    current = _current_score_payload(payload)
    if current is not None:
        return "current", current

    scores = payload.get("scores")
    if isinstance(scores, dict) and any(key in scores for key in LEGACY_JUDGE_KEYS):
        return "legacy", None
    return "invalid", None


def _annotation_result_status(payload: dict[str, object]) -> Literal["current", "legacy", "invalid"]:
    system_a_scores = payload.get("system_a_scores")
    if not isinstance(system_a_scores, dict):
        return "invalid"
    if all(key in system_a_scores for key in ("relevance", "groundedness", "distinctiveness")):
        return "current"
    if any(key in system_a_scores for key in LEGACY_ANNOTATION_KEYS):
        return "legacy"
    return "invalid"


def _relative_path(from_dir: Path, to_path: Path) -> str:
    return Path(os.path.relpath(to_path, from_dir)).as_posix()


def _model_dump_json(model: BaseModel, *, indent: int = 2) -> str:
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json(indent=indent)  # type: ignore[no-any-return]
    return model.json(indent=indent)


def _load_prompt_metadata(run_dir: Path) -> tuple[str, str, str]:
    for name in ("pipeline.json", "baseline.json"):
        candidate = run_dir / name
        if not candidate.exists():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        prompt = payload.get("prompt")
        if isinstance(prompt, dict):
            prompt_id = str(prompt.get("id") or run_dir.name)
            domain = str(prompt.get("domain") or "unknown")
            statement = str(prompt.get("statement") or "")
            return prompt_id, domain, statement
    return run_dir.name, "unknown", ""


def _load_prompt_outcomes(runs_dir: Path) -> tuple[list[PromptOutcome], list[str]]:
    outcomes: list[PromptOutcome] = []
    warnings: list[str] = []

    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir() and not path.name.startswith("_")):
        baseline_path = run_dir / "judge_baseline.json"
        pipeline_path = run_dir / "judge_pipeline.json"
        if not baseline_path.exists() or not pipeline_path.exists():
            continue

        baseline_status, baseline_scores = _judge_artifact_status(baseline_path)
        pipeline_status, pipeline_scores = _judge_artifact_status(pipeline_path)
        if baseline_status != "current" or pipeline_status != "current" or baseline_scores is None or pipeline_scores is None:
            warnings.append(
                f"Skipped {run_dir.name}: expected current judge schema, found baseline={baseline_status}, pipeline={pipeline_status}."
            )
            continue

        prompt_id, domain, statement = _load_prompt_metadata(run_dir)
        delta = _delta_scores(baseline_scores, pipeline_scores)
        if delta.overall_preference > 0:
            winner: Literal["baseline", "pipeline", "tie"] = "pipeline"
        elif delta.overall_preference < 0:
            winner = "baseline"
        else:
            winner = "tie"

        outcomes.append(
            PromptOutcome(
                prompt_id=prompt_id,
                domain=domain,
                statement=statement,
                baseline=baseline_scores,
                pipeline=pipeline_scores,
                delta=delta,
                winner=winner,
            )
        )

    return outcomes, warnings


def _build_domain_aggregates(outcomes: list[PromptOutcome]) -> list[DomainAggregate]:
    grouped: dict[str, list[PromptOutcome]] = defaultdict(list)
    for outcome in outcomes:
        grouped[outcome.domain].append(outcome)

    aggregates: list[DomainAggregate] = []
    for domain, rows in sorted(grouped.items()):
        baseline = _mean_scores([row.baseline for row in rows])
        pipeline = _mean_scores([row.pipeline for row in rows])
        aggregates.append(
            DomainAggregate(
                domain=domain,
                prompt_count=len(rows),
                pipeline_wins=sum(1 for row in rows if row.winner == "pipeline"),
                baseline_wins=sum(1 for row in rows if row.winner == "baseline"),
                ties=sum(1 for row in rows if row.winner == "tie"),
                baseline=baseline,
                pipeline=pipeline,
                delta=_delta_scores(baseline, pipeline),
            )
        )
    return aggregates


def _variant_from_tag_id(tag_id: str) -> str:
    if "_baseline_" in tag_id:
        return "baseline"
    if "_pipeline_" in tag_id:
        return "pipeline"
    return "unknown"


def _load_failure_summaries(failure_tags_path: Path) -> tuple[list[FailureCategorySummary], list[FailurePromptSummary], list[str]]:
    if not failure_tags_path.exists():
        return [], [], ["No failure_tags.jsonl found; skipping failure taxonomy summary."]

    category_stats: dict[str, dict[str, float]] = defaultdict(lambda: {
        "total_count": 0,
        "baseline_count": 0,
        "pipeline_count": 0,
        "severity_sum": 0.0,
        "max_severity": 0,
    })
    prompt_stats: dict[str, dict[str, object]] = defaultdict(lambda: {
        "tag_count": 0,
        "max_severity": 0,
        "variants": set(),
        "categories": set(),
    })

    for raw_line in failure_tags_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        category = str(payload.get("category") or "unknown")
        severity = int(payload.get("severity") or 0)
        prompt_id = str(payload.get("prompt_id") or "unknown")
        variant = _variant_from_tag_id(str(payload.get("tag_id") or ""))

        category_row = category_stats[category]
        category_row["total_count"] += 1
        category_row["severity_sum"] += severity
        category_row["max_severity"] = max(int(category_row["max_severity"]), severity)
        if variant == "baseline":
            category_row["baseline_count"] += 1
        elif variant == "pipeline":
            category_row["pipeline_count"] += 1

        prompt_row = prompt_stats[prompt_id]
        prompt_row["tag_count"] = int(prompt_row["tag_count"]) + 1
        prompt_row["max_severity"] = max(int(prompt_row["max_severity"]), severity)
        cast_variants = prompt_row["variants"]
        cast_categories = prompt_row["categories"]
        if isinstance(cast_variants, set):
            cast_variants.add(variant)
        if isinstance(cast_categories, set):
            cast_categories.add(category)

    categories = [
        FailureCategorySummary(
            category=category,
            total_count=int(values["total_count"]),
            baseline_count=int(values["baseline_count"]),
            pipeline_count=int(values["pipeline_count"]),
            avg_severity=(float(values["severity_sum"]) / int(values["total_count"])) if int(values["total_count"]) else 0.0,
            max_severity=int(values["max_severity"]),
        )
        for category, values in category_stats.items()
    ]
    categories.sort(key=lambda row: (-row.total_count, -row.max_severity, row.category))

    prompts = [
        FailurePromptSummary(
            prompt_id=prompt_id,
            tag_count=int(values["tag_count"]),
            max_severity=int(values["max_severity"]),
            variants=sorted(str(variant) for variant in values["variants"]),
            categories=sorted(str(category) for category in values["categories"]),
        )
        for prompt_id, values in prompt_stats.items()
    ]
    prompts.sort(key=lambda row: (-row.max_severity, -row.tag_count, row.prompt_id))

    return categories, prompts[:5], []


def _load_retrieval_summary(retrieval_summary_path: Path | None, workspace_root: Path) -> tuple[RetrievalSummarySnapshot | None, list[str]]:
    warnings: list[str] = []
    candidate_path = retrieval_summary_path
    if candidate_path is None:
        for relative in AUTO_RETRIEVAL_SUMMARY_PATHS:
            probe = workspace_root / relative
            if probe.exists():
                candidate_path = probe
                break

    if candidate_path is None or not candidate_path.exists():
        warnings.append("No retrieval_eval_summary.json found; retrieval section will be marked pending.")
        return None, warnings

    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    aggregates = payload.get("aggregates", {})
    snapshot = RetrievalSummarySnapshot(
        path=str(candidate_path),
        query_count=int(payload.get("query_count") or 0),
        retrieval_mode=str(payload.get("retrieval_mode") or "unknown"),
        reranker=str(payload.get("reranker") or "unknown"),
        k_values=[int(value) for value in payload.get("k_values", [])],
        recall_at_k={str(key): float(value) for key, value in dict(aggregates.get("recall_at_k", {})).items()},
        mrr_at_k={str(key): float(value) for key, value in dict(aggregates.get("mrr_at_k", {})).items()},
        ndcg_at_k={str(key): float(value) for key, value in dict(aggregates.get("ndcg_at_k", {})).items()},
    )
    metric_values = list(snapshot.recall_at_k.values()) + list(snapshot.mrr_at_k.values()) + list(snapshot.ndcg_at_k.values())
    if metric_values and all(value == 0.0 for value in metric_values):
        warnings.append(
            "Retrieval benchmark metrics are all zero; the labeled snippet IDs likely do not align with the current index contents."
        )
    return snapshot, warnings


def _load_sweep_summary(sweep_dir: Path) -> tuple[SweepSummary, list[str]]:
    if not sweep_dir.exists():
        return SweepSummary(status="missing", note="No prompt-sweep artifacts found.", variants_found=[]), []

    variants = sorted(path for path in sweep_dir.iterdir() if path.is_dir())
    if not variants:
        return SweepSummary(status="missing", note="No prompt-sweep artifacts found.", variants_found=[]), []

    warnings: list[str] = []
    current_aggregates: list[SweepAggregate] = []
    legacy_variants: list[str] = []
    invalid_variants: list[str] = []

    for variant_dir in variants:
        judge_paths = sorted(variant_dir.glob("*/judge_pipeline.json"))
        if not judge_paths:
            invalid_variants.append(variant_dir.name)
            continue

        snapshots: list[ScoreSnapshot] = []
        variant_status: Literal["current", "legacy", "invalid"] = "current"
        for judge_path in judge_paths:
            status, snapshot = _judge_artifact_status(judge_path)
            if status != "current" or snapshot is None:
                variant_status = status
                break
            snapshots.append(snapshot)

        if variant_status == "legacy":
            legacy_variants.append(variant_dir.name)
        elif variant_status == "invalid":
            invalid_variants.append(variant_dir.name)
        else:
            current_aggregates.append(
                SweepAggregate(
                    variant=variant_dir.name,
                    prompt_count=len(snapshots),
                    scores=_mean_scores(snapshots),
                )
            )

    variant_names = [variant.name for variant in variants]
    if legacy_variants and not current_aggregates:
        note = (
            "Existing prompt-sweep artifacts use the pre-migration rubric schema "
            f"and were excluded: {', '.join(legacy_variants)}."
        )
        warnings.append(note)
        return SweepSummary(status="legacy", note=note, variants_found=variant_names), warnings
    if legacy_variants or invalid_variants:
        note = "Prompt-sweep artifacts are mixed; excluded from summary until regenerated under the current rubric."
        warnings.append(note)
        return SweepSummary(status="mixed", note=note, variants_found=variant_names, aggregates=current_aggregates), warnings

    note = f"Loaded current-schema prompt-sweep artifacts for {len(current_aggregates)} variant(s)."
    return SweepSummary(status="current", note=note, variants_found=variant_names, aggregates=current_aggregates), warnings


def _load_annotation_progress(
    tasks_dir: Path,
    results_dir: Path,
    exports_dir: Path,
) -> tuple[AnnotationProgressSummary, list[str]]:
    warnings: list[str] = []
    total_tasks = len([path for path in tasks_dir.glob("task_*.json") if path.is_file()]) if tasks_dir.exists() else 0

    annotator_sets: dict[str, dict[str, set[str]]] = defaultdict(lambda: {
        "autosave_current": set(),
        "autosave_legacy": set(),
        "export_current": set(),
        "export_legacy": set(),
    })

    if results_dir.exists():
        for annotator_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
            for result_path in sorted(annotator_dir.glob("task_*.json")):
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                task_id = str(payload.get("task_id") or result_path.stem)
                status = _annotation_result_status(payload)
                if status == "current":
                    annotator_sets[annotator_dir.name]["autosave_current"].add(task_id)
                elif status == "legacy":
                    annotator_sets[annotator_dir.name]["autosave_legacy"].add(task_id)

    export_file_count = 0
    if exports_dir.exists():
        for export_path in sorted(path for path in exports_dir.glob("*.json") if path.is_file()):
            export_file_count += 1
            payload = json.loads(export_path.read_text(encoding="utf-8"))
            annotator_id = export_path.stem
            if isinstance(payload, dict) and isinstance(payload.get("annotator_id"), str) and payload["annotator_id"].strip():
                annotator_id = payload["annotator_id"].strip()
            raw_results = payload.get("results") if isinstance(payload, dict) else payload
            if not isinstance(raw_results, list):
                continue
            for raw_result in raw_results:
                if not isinstance(raw_result, dict):
                    continue
                task_id = str(raw_result.get("task_id") or "")
                if not task_id:
                    continue
                status = _annotation_result_status(raw_result)
                if status == "current":
                    annotator_sets[annotator_id]["export_current"].add(task_id)
                elif status == "legacy":
                    annotator_sets[annotator_id]["export_legacy"].add(task_id)

    annotators = [
        AnnotationAnnotatorProgress(
            annotator_id=annotator_id,
            autosave_current_count=len(values["autosave_current"]),
            autosave_legacy_count=len(values["autosave_legacy"]),
            export_current_count=len(values["export_current"]),
            export_legacy_count=len(values["export_legacy"]),
        )
        for annotator_id, values in sorted(annotator_sets.items())
    ]

    current_completed_tasks = len({
        task_id
        for values in annotator_sets.values()
        for label in ("autosave_current", "export_current")
        for task_id in values[label]
    })
    legacy_completed_tasks = len({
        task_id
        for values in annotator_sets.values()
        for label in ("autosave_legacy", "export_legacy")
        for task_id in values[label]
    })

    if total_tasks == 0:
        note = "No annotation task batch found."
    elif export_file_count == 0 and legacy_completed_tasks > 0 and current_completed_tasks == 0:
        note = (
            "Annotation is in progress, but only legacy-format autosaves were found. "
            "Current-schema exports are still pending."
        )
        warnings.append(note)
    elif export_file_count == 0 and legacy_completed_tasks > 0:
        note = (
            "Annotation is in progress; current-schema results exist, but legacy-format autosaves "
            "are also present and should not be used for final agreement analysis."
        )
        warnings.append(note)
    elif export_file_count == 0:
        note = "Annotation is in progress; no tracked export JSON files have been saved yet."
    else:
        note = "Tracked annotation exports are present."

    summary = AnnotationProgressSummary(
        total_tasks=total_tasks,
        current_completed_tasks=current_completed_tasks,
        legacy_completed_tasks=legacy_completed_tasks,
        export_file_count=export_file_count,
        annotators=annotators,
        note=note,
    )
    return summary, warnings


def _status_for_optional_dir(path: Path, *, family: str, detail_when_present: str) -> ArtifactStatus:
    if path.exists() and any(path.iterdir()):
        return ArtifactStatus(family=family, status="complete", detail=detail_when_present)
    return ArtifactStatus(family=family, status="missing", detail=f"No {family.replace('_', ' ')} artifacts found.")


def _svg_style() -> str:
    return (
        "text { font-family: 'Segoe UI', sans-serif; fill: #1f2937; }"
        ".title { font-size: 24px; font-weight: 700; }"
        ".axis { stroke: #94a3b8; stroke-width: 1; }"
        ".grid { stroke: #e2e8f0; stroke-width: 1; }"
        ".tick { font-size: 12px; fill: #64748b; }"
        ".label { font-size: 13px; }"
        ".value { font-size: 12px; font-weight: 600; }"
        ".legend { font-size: 12px; }"
    )


def _svg_header(title: str, width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        "<style>",
        _svg_style(),
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#fffaf1" />',
        f'<text class="title" x="24" y="40">{html.escape(title)}</text>',
    ]


def _render_empty_svg(path: Path, *, title: str, message: str) -> None:
    lines = _svg_header(title, 900, 180)
    lines.append(f'<text class="label" x="24" y="96">{html.escape(message)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_grouped_metric_chart(
    path: Path,
    *,
    title: str,
    categories: list[str],
    series: list[tuple[str, str, list[float]]],
    max_value: float,
) -> None:
    width = 980
    height = 420
    left = 80
    top = 80
    bottom = 90
    right = 30
    plot_w = width - left - right
    plot_h = height - top - bottom
    group_w = plot_w / max(len(categories), 1)
    bar_w = min(48.0, (group_w * 0.68) / max(len(series), 1))
    lines = _svg_header(title, width, height)

    for step in range(6):
        value = max_value * step / 5
        y = top + plot_h - (plot_h * step / 5)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" />')
        lines.append(f'<text class="tick" x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{value:.1f}</text>')

    lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" />')
    lines.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" />')

    for category_index, category in enumerate(categories):
        group_x = left + (category_index * group_w)
        for series_index, (_, color, values) in enumerate(series):
            value = values[category_index] if category_index < len(values) else 0.0
            height_px = 0.0 if max_value <= 0 else (value / max_value) * plot_h
            x = group_x + (group_w * 0.16) + (series_index * bar_w)
            y = top + plot_h - height_px
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 8:.1f}" height="{height_px:.1f}" '
                f'rx="8" fill="{color}" />'
            )
            lines.append(
                f'<text class="value" x="{x + (bar_w - 8) / 2:.1f}" y="{max(y - 6, top + 12):.1f}" '
                f'text-anchor="middle">{value:.2f}</text>'
            )
        lines.append(
            f'<text class="label" x="{group_x + group_w / 2:.1f}" y="{height - 34}" text-anchor="middle">{html.escape(category)}</text>'
        )

    legend_x = width - 220
    legend_y = 46
    for index, (label, color, _) in enumerate(series):
        y = legend_y + (index * 18)
        lines.append(f'<rect x="{legend_x}" y="{y - 10}" width="14" height="14" rx="3" fill="{color}" />')
        lines.append(f'<text class="legend" x="{legend_x + 22}" y="{y + 1}">{html.escape(label)}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_domain_delta_chart(path: Path, domains: list[DomainAggregate]) -> None:
    if not domains:
        _render_empty_svg(path, title="Domain Overall Preference Delta", message="No domain aggregates available.")
        return

    categories = [domain.domain for domain in domains]
    values = [domain.delta.overall_preference for domain in domains]
    max_value = max(max(values), 0.1)
    _write_grouped_metric_chart(
        path,
        title="Domain Overall Preference Delta",
        categories=categories,
        series=[("Pipeline - Baseline", "#2a9d8f", values)],
        max_value=max_value,
    )


def _write_prompt_delta_chart(path: Path, outcomes: list[PromptOutcome]) -> None:
    if not outcomes:
        _render_empty_svg(path, title="Per-Prompt Overall Preference Delta", message="No prompt outcomes available.")
        return

    rows = sorted(outcomes, key=lambda row: (-row.delta.overall_preference, row.prompt_id))
    width = 1280
    row_h = 34
    height = max(220, 100 + len(rows) * row_h)
    left = 360
    right = 90
    top = 80
    bottom = 40
    plot_w = width - left - right
    plot_h = height - top - bottom
    min_value = min(min(row.delta.overall_preference for row in rows), 0.0)
    max_value = max(max(row.delta.overall_preference for row in rows), 0.0)
    span = max(max_value - min_value, 0.5)
    zero_x = left + ((0.0 - min_value) / span) * plot_w

    lines = _svg_header("Per-Prompt Overall Preference Delta", width, height)
    for tick_index in range(6):
        value = min_value + (span * tick_index / 5)
        x = left + (plot_w * tick_index / 5)
        lines.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" />')
        lines.append(f'<text class="tick" x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle">{value:.1f}</text>')
    lines.append(f'<line class="axis" x1="{zero_x:.1f}" y1="{top}" x2="{zero_x:.1f}" y2="{top + plot_h}" />')

    for index, row in enumerate(rows):
        center_y = top + (index * row_h) + 16
        bar_h = 16
        scaled = (row.delta.overall_preference / span) * plot_w
        x = zero_x if scaled >= 0 else zero_x + scaled
        color = "#2a9d8f" if row.delta.overall_preference >= 0 else "#d06767"
        lines.append(f'<text class="label" x="{left - 14}" y="{center_y + 4}" text-anchor="end">{html.escape(row.prompt_id)}</text>')
        lines.append(
            f'<rect x="{x:.1f}" y="{center_y - bar_h / 2:.1f}" width="{abs(scaled):.1f}" height="{bar_h}" rx="6" fill="{color}" />'
        )
        value_anchor = "start" if row.delta.overall_preference >= 0 else "end"
        value_x = x + abs(scaled) + 6 if row.delta.overall_preference >= 0 else x - 6
        lines.append(
            f'<text class="value" x="{value_x:.1f}" y="{center_y + 4:.1f}" text-anchor="{value_anchor}">{row.delta.overall_preference:+.2f}</text>'
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_failure_category_chart(path: Path, categories: list[FailureCategorySummary]) -> None:
    if not categories:
        _render_empty_svg(path, title="Failure Tags by Category", message="No failure tags available.")
        return

    width = 980
    row_h = 48
    height = 130 + len(categories) * row_h
    left = 260
    top = 76
    right = 50
    plot_w = width - left - right
    max_value = max(max(row.baseline_count, row.pipeline_count) for row in categories)
    max_value = max(max_value, 1)
    lines = _svg_header("Failure Tags by Category", width, height)

    for tick_index in range(max_value + 1):
        x = left + (plot_w * tick_index / max_value)
        lines.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - 26}" />')
        lines.append(f'<text class="tick" x="{x:.1f}" y="{height - 10}" text-anchor="middle">{tick_index}</text>')

    legend_x = width - 220
    for index, (label, color) in enumerate((("Baseline", "#d06767"), ("Pipeline", "#5f83bf"))):
        y = 46 + index * 18
        lines.append(f'<rect x="{legend_x}" y="{y - 10}" width="14" height="14" rx="3" fill="{color}" />')
        lines.append(f'<text class="legend" x="{legend_x + 22}" y="{y + 1}">{label}</text>')

    for index, category in enumerate(categories):
        y = top + (index * row_h)
        lines.append(f'<text class="label" x="{left - 14}" y="{y + 20}" text-anchor="end">{html.escape(category.category)}</text>')
        baseline_w = (category.baseline_count / max_value) * plot_w
        pipeline_w = (category.pipeline_count / max_value) * plot_w
        lines.append(f'<rect x="{left}" y="{y + 4}" width="{baseline_w:.1f}" height="14" rx="6" fill="#d06767" />')
        lines.append(f'<rect x="{left}" y="{y + 24}" width="{pipeline_w:.1f}" height="14" rx="6" fill="#5f83bf" />')
        lines.append(f'<text class="value" x="{left + baseline_w + 6:.1f}" y="{y + 16}">{category.baseline_count}</text>')
        lines.append(f'<text class="value" x="{left + pipeline_w + 6:.1f}" y="{y + 36}">{category.pipeline_count}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_annotation_progress_chart(path: Path, annotation: AnnotationProgressSummary) -> None:
    if not annotation.annotators:
        _render_empty_svg(path, title="Human Annotation Progress", message=annotation.note)
        return

    width = 980
    row_h = 44
    height = 140 + len(annotation.annotators) * row_h
    left = 220
    right = 60
    top = 86
    plot_w = width - left - right
    total = max(annotation.total_tasks, 1)
    lines = _svg_header("Human Annotation Progress", width, height)
    lines.append(f'<text class="label" x="24" y="66">{html.escape(annotation.note)}</text>')

    legend_x = width - 240
    for index, (label, color) in enumerate((
        ("Current-schema results", "#2a9d8f"),
        ("Legacy-schema results", "#e9a03b"),
    )):
        y = 46 + index * 18
        lines.append(f'<rect x="{legend_x}" y="{y - 10}" width="14" height="14" rx="3" fill="{color}" />')
        lines.append(f'<text class="legend" x="{legend_x + 22}" y="{y + 1}">{label}</text>')

    for index, annotator in enumerate(annotation.annotators):
        y = top + index * row_h
        legacy_w = (annotator.autosave_legacy_count / total) * plot_w
        current_w = (annotator.autosave_current_count / total) * plot_w
        lines.append(f'<text class="label" x="{left - 14}" y="{y + 18}" text-anchor="end">{html.escape(annotator.annotator_id)}</text>')
        lines.append(f'<rect x="{left}" y="{y + 4}" width="{plot_w}" height="18" rx="8" fill="#e5e7eb" />')
        if legacy_w > 0:
            lines.append(f'<rect x="{left}" y="{y + 4}" width="{legacy_w:.1f}" height="18" rx="8" fill="#e9a03b" />')
        if current_w > 0:
            lines.append(f'<rect x="{left + legacy_w:.1f}" y="{y + 4}" width="{current_w:.1f}" height="18" rx="8" fill="#2a9d8f" />')
        lines.append(
            f'<text class="value" x="{left + plot_w + 10}" y="{y + 18}">'
            f'current {annotator.autosave_current_count}, legacy {annotator.autosave_legacy_count}, exports {annotator.export_current_count}'
            "</text>"
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_retrieval_metrics_chart(path: Path, retrieval: RetrievalSummarySnapshot) -> None:
    k_labels = [f"@{k}" for k in retrieval.k_values]
    recall = [retrieval.recall_at_k.get(str(k), 0.0) for k in retrieval.k_values]
    mrr = [retrieval.mrr_at_k.get(str(k), 0.0) for k in retrieval.k_values]
    ndcg = [retrieval.ndcg_at_k.get(str(k), 0.0) for k in retrieval.k_values]
    _write_grouped_metric_chart(
        path,
        title="Retrieval Metrics by K",
        categories=k_labels,
        series=[
            ("Recall", "#5f83bf", recall),
            ("MRR", "#e9a03b", mrr),
            ("nDCG", "#2a9d8f", ndcg),
        ],
        max_value=1.0,
    )


def _write_overall_scores_chart(path: Path, summary: FinalAnalysisSummary) -> None:
    categories = [_metric_label(metric) for metric in CURRENT_METRICS]
    baseline = [getattr(summary.baseline, metric) for metric in CURRENT_METRICS]
    pipeline = [getattr(summary.pipeline, metric) for metric in CURRENT_METRICS]
    _write_grouped_metric_chart(
        path,
        title="Average Scores Across All Prompts",
        categories=categories,
        series=[
            ("Baseline", "#d06767", baseline),
            ("Pipeline", "#2a9d8f", pipeline),
        ],
        max_value=5.0,
    )


def _generate_figures(summary: FinalAnalysisSummary, output_dir: Path) -> list[Path]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    overall_path = figures_dir / "overall_scores.svg"
    domain_path = figures_dir / "domain_overall_delta.svg"
    prompt_path = figures_dir / "prompt_overall_deltas.svg"
    failure_path = figures_dir / "failure_categories.svg"
    annotation_path = figures_dir / "annotation_progress.svg"

    _write_overall_scores_chart(overall_path, summary)
    _write_domain_delta_chart(domain_path, summary.domains)
    _write_prompt_delta_chart(prompt_path, summary.prompt_outcomes)
    _write_failure_category_chart(failure_path, summary.failure_categories)
    _write_annotation_progress_chart(annotation_path, summary.annotation)

    figure_paths = [overall_path, domain_path, prompt_path, failure_path, annotation_path]
    if summary.retrieval is not None:
        retrieval_path = figures_dir / "retrieval_metrics.svg"
        _write_retrieval_metrics_chart(retrieval_path, summary.retrieval)
        figure_paths.append(retrieval_path)

    return figure_paths


def _status_table(
    outcomes: list[PromptOutcome],
    failures_present: bool,
    retrieval: RetrievalSummarySnapshot | None,
    sweep: SweepSummary,
    annotation: AnnotationProgressSummary,
    runs_dir: Path,
) -> list[ArtifactStatus]:
    return [
        ArtifactStatus(
            family="evaluation_runs",
            status="complete" if outcomes else "missing",
            detail=f"{len(outcomes)} current-schema prompt evaluations loaded." if outcomes else "No current-schema evaluation runs found.",
        ),
        ArtifactStatus(
            family="failure_taxonomy",
            status="complete" if failures_present else "missing",
            detail="failure_tags.jsonl present and summarized." if failures_present else "No failure_tags.jsonl found.",
        ),
        ArtifactStatus(
            family="retrieval_eval",
            status="complete" if retrieval is not None else "partial",
            detail=(
                f"Loaded retrieval benchmark from {retrieval.path}."
                if retrieval is not None
                else "Retrieval benchmark not found in the default output locations."
            ),
        ),
        ArtifactStatus(
            family="prompt_sweep",
            status="complete" if sweep.status == "current" else ("excluded" if sweep.status in {"legacy", "mixed"} else "missing"),
            detail=sweep.note,
        ),
        ArtifactStatus(
            family="human_annotation",
            status="partial" if annotation.total_tasks > 0 else "missing",
            detail=annotation.note,
        ),
        _status_for_optional_dir(runs_dir.parent / "variant_runs", family="variant_runs", detail_when_present="Variant evaluation artifacts found."),
        _status_for_optional_dir(runs_dir.parent / "robustness_runs", family="robustness_runs", detail_when_present="Robustness evaluation artifacts found."),
    ]


def _top_outcomes(outcomes: list[PromptOutcome], *, positive: bool, limit: int) -> list[PromptOutcome]:
    filtered = [row for row in outcomes if (row.delta.overall_preference > 0 if positive else row.delta.overall_preference < 0)]
    filtered.sort(
        key=lambda row: row.delta.overall_preference,
        reverse=positive,
    )
    return filtered[:limit]


def _report_link(output_dir: Path, path: Path) -> str:
    return _relative_path(output_dir, path)


def _write_markdown_report(summary: FinalAnalysisSummary, summary_path: Path, output_dir: Path) -> Path:
    report_path = output_dir / "final_analysis_report.md"
    top_gains = _top_outcomes(summary.prompt_outcomes, positive=True, limit=5)
    regressions = _top_outcomes(summary.prompt_outcomes, positive=False, limit=len(summary.prompt_outcomes))
    figure_rel = {Path(path).name: _relative_path(output_dir, Path(path)) for path in summary.figure_paths}

    lines: list[str] = [
        "# Final Analysis Report",
        "",
        f"**Generated:** {summary.generated_at.isoformat()}",
        f"**Runs dir:** `{summary.runs_dir}`",
        f"**Summary JSON:** [`{summary_path.name}`]({_report_link(output_dir, summary_path)})",
        "",
        "## Executive Summary",
        "",
        f"- Pipeline wins `{summary.pipeline_wins}` of `{summary.prompt_count}` evaluated prompts; baseline wins `{summary.baseline_wins}`, ties `{summary.ties}`.",
        f"- Top-line `overall_preference` improved from `{summary.baseline.overall_preference:.2f}` to `{summary.pipeline.overall_preference:.2f}` (`{summary.delta.overall_preference:+.2f}`).",
        f"- Judge panel mode: `{summary.judge_panel_mode}`.",
        f"- Human annotation status: {summary.annotation.note}",
        "",
        "## Artifact Status",
        "",
        "| Family | Status | Detail |",
        "|---|---|---|",
    ]

    for status in summary.artifact_statuses:
        lines.append(f"| {status.family} | {status.status} | {status.detail} |")

    lines.extend([
        "",
        "## Figures",
        "",
        "### Aggregate rubric scores",
        f"![Aggregate rubric scores]({figure_rel['overall_scores.svg']})",
        "",
        "### Domain-level overall preference deltas",
        f"![Domain overall preference deltas]({figure_rel['domain_overall_delta.svg']})",
        "",
        "### Prompt-level overall preference deltas",
        f"![Prompt-level overall preference deltas]({figure_rel['prompt_overall_deltas.svg']})",
        "",
        "### Failure-tag concentration by system",
        f"![Failure tags by category]({figure_rel['failure_categories.svg']})",
        "",
        "### Human annotation progress",
        f"![Human annotation progress]({figure_rel['annotation_progress.svg']})",
    ])

    if summary.retrieval is not None and "retrieval_metrics.svg" in figure_rel:
        lines.extend([
            "",
            "### Retrieval benchmark metrics",
            f"![Retrieval metrics by K]({figure_rel['retrieval_metrics.svg']})",
        ])

    lines.extend([
        "",
        "## Score Summary",
        "",
        "| Metric | Baseline | Pipeline | Delta |",
        "|---|---:|---:|---:|",
    ])
    for metric in CURRENT_METRICS:
        lines.append(
            f"| {_metric_label(metric)} | {getattr(summary.baseline, metric):.2f} | "
            f"{getattr(summary.pipeline, metric):.2f} | {getattr(summary.delta, metric):+.2f} |"
        )

    lines.extend([
        "",
        "## Domain Breakdown",
        "",
        "| Domain | Prompts | Pipeline Wins | Baseline Wins | Baseline Overall | Pipeline Overall | Delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for domain in summary.domains:
        lines.append(
            f"| {domain.domain} | {domain.prompt_count} | {domain.pipeline_wins} | {domain.baseline_wins} | "
            f"{domain.baseline.overall_preference:.2f} | {domain.pipeline.overall_preference:.2f} | "
            f"{domain.delta.overall_preference:+.2f} |"
        )

    lines.extend([
        "",
        "## Strongest Gains",
        "",
        "| Prompt | Domain | Overall Delta |",
        "|---|---|---:|",
    ])
    for row in top_gains:
        lines.append(f"| {row.prompt_id} | {row.domain} | {row.delta.overall_preference:+.2f} |")

    lines.extend([
        "",
        "## Regressions",
        "",
        "| Prompt | Domain | Overall Delta |",
        "|---|---|---:|",
    ])
    if regressions:
        for row in regressions:
            lines.append(f"| {row.prompt_id} | {row.domain} | {row.delta.overall_preference:+.2f} |")
    else:
        lines.append("| none | n/a | 0.00 |")

    lines.extend([
        "",
        "## Failure Taxonomy",
        "",
        "| Category | Total Tags | Baseline | Pipeline | Avg Severity | Max Severity |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for category in summary.failure_categories:
        lines.append(
            f"| {category.category} | {category.total_count} | {category.baseline_count} | {category.pipeline_count} | "
            f"{category.avg_severity:.2f} | {category.max_severity} |"
        )

    if summary.top_failure_prompts:
        lines.extend([
            "",
            "### Highest-severity Prompt Failures",
            "",
            "| Prompt | Tag Count | Max Severity | Variants | Categories |",
            "|---|---:|---:|---|---|",
        ])
        for prompt in summary.top_failure_prompts:
            lines.append(
                f"| {prompt.prompt_id} | {prompt.tag_count} | {prompt.max_severity} | "
                f"{', '.join(prompt.variants)} | {', '.join(prompt.categories)} |"
            )

    lines.extend([
        "",
        "## Retrieval Evaluation",
        "",
    ])
    if summary.retrieval is None:
        lines.append("Retrieval benchmark summary was not available when this report was generated.")
    else:
        lines.extend([
            f"- Queries evaluated: `{summary.retrieval.query_count}`",
            f"- Mode: `{summary.retrieval.retrieval_mode}` | Reranker: `{summary.retrieval.reranker}`",
            "",
            "| Metric | " + " | ".join(f"@{k}" for k in summary.retrieval.k_values) + " |",
            "|---|" + "|".join("---:" for _ in summary.retrieval.k_values) + "|",
            "| Recall | " + " | ".join(f"{summary.retrieval.recall_at_k.get(str(k), 0.0):.4f}" for k in summary.retrieval.k_values) + " |",
            "| MRR | " + " | ".join(f"{summary.retrieval.mrr_at_k.get(str(k), 0.0):.4f}" for k in summary.retrieval.k_values) + " |",
            "| nDCG | " + " | ".join(f"{summary.retrieval.ndcg_at_k.get(str(k), 0.0):.4f}" for k in summary.retrieval.k_values) + " |",
        ])

    lines.extend([
        "",
        "## Human Annotation",
        "",
        f"- Task batch size: `{summary.annotation.total_tasks}`",
        f"- Compatible completed tasks: `{summary.annotation.current_completed_tasks}`",
        f"- Legacy completed tasks: `{summary.annotation.legacy_completed_tasks}`",
        f"- Tracked export files: `{summary.annotation.export_file_count}`",
        "",
        "| Annotator | Autosave Current | Autosave Legacy | Export Current | Export Legacy |",
        "|---|---:|---:|---:|---:|",
    ])
    if summary.annotation.annotators:
        for annotator in summary.annotation.annotators:
            lines.append(
                f"| {annotator.annotator_id} | {annotator.autosave_current_count} | {annotator.autosave_legacy_count} | "
                f"{annotator.export_current_count} | {annotator.export_legacy_count} |"
            )
    else:
        lines.append("| none | 0 | 0 | 0 | 0 |")

    lines.extend([
        "",
        "## Exclusions And Caveats",
        "",
        f"- Prompt sweep: {summary.sweep.note}",
        "- Variant and robustness suites are reported only if their artifact directories exist; they were not part of the completed run set in this workspace snapshot.",
        "- Main evaluation results are still judge-only; human validation is not yet ready for agreement analysis.",
    ])
    if summary.warnings:
        lines.extend([
            "",
            "## Warnings",
            "",
        ])
        for warning in summary.warnings:
            lines.append(f"- {warning}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def build_analysis_report(
    *,
    runs_dir: Path,
    output_dir: Path,
    sweep_dir: Path,
    retrieval_summary_path: Path | None,
    annotation_tasks_dir: Path,
    annotation_results_dir: Path,
    annotation_exports_dir: Path,
) -> tuple[FinalAnalysisSummary, Path, Path]:
    workspace_root = runs_dir.resolve().parents[1]
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_outcomes, warnings = _load_prompt_outcomes(runs_dir)
    if not prompt_outcomes:
        raise ValueError(f"No current-schema evaluation runs found under {runs_dir}")

    baseline = _mean_scores([row.baseline for row in prompt_outcomes])
    pipeline = _mean_scores([row.pipeline for row in prompt_outcomes])
    domains = _build_domain_aggregates(prompt_outcomes)
    failure_categories, top_failure_prompts, failure_warnings = _load_failure_summaries(runs_dir / "failure_tags.jsonl")
    retrieval, retrieval_warnings = _load_retrieval_summary(retrieval_summary_path, workspace_root)
    sweep, sweep_warnings = _load_sweep_summary(sweep_dir)
    annotation, annotation_warnings = _load_annotation_progress(
        annotation_tasks_dir,
        annotation_results_dir,
        annotation_exports_dir,
    )
    warnings.extend(failure_warnings)
    warnings.extend(retrieval_warnings)
    warnings.extend(sweep_warnings)
    warnings.extend(annotation_warnings)

    summary = FinalAnalysisSummary(
        runs_dir=str(runs_dir),
        output_dir=str(output_dir),
        prompt_count=len(prompt_outcomes),
        judge_panel_mode="disabled (single judge)",
        pipeline_wins=sum(1 for row in prompt_outcomes if row.winner == "pipeline"),
        baseline_wins=sum(1 for row in prompt_outcomes if row.winner == "baseline"),
        ties=sum(1 for row in prompt_outcomes if row.winner == "tie"),
        baseline=baseline,
        pipeline=pipeline,
        delta=_delta_scores(baseline, pipeline),
        domains=domains,
        prompt_outcomes=sorted(prompt_outcomes, key=lambda row: row.prompt_id),
        failure_categories=failure_categories,
        top_failure_prompts=top_failure_prompts,
        retrieval=retrieval,
        sweep=sweep,
        annotation=annotation,
        warnings=warnings,
    )

    summary.artifact_statuses = _status_table(
        prompt_outcomes,
        failures_present=bool(failure_categories),
        retrieval=retrieval,
        sweep=sweep,
        annotation=annotation,
        runs_dir=runs_dir,
    )

    figure_paths = _generate_figures(summary, output_dir)
    summary.figure_paths = [str(path) for path in figure_paths]

    summary_path = output_dir / "final_analysis_summary.json"
    summary_path.write_text(_model_dump_json(summary, indent=2), encoding="utf-8")
    report_path = _write_markdown_report(summary, summary_path, output_dir)
    return summary, summary_path, report_path
