"""Deterministic failure-tag taxonomy from persisted evaluation artifacts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..schemas import FailureTag, JudgeResult


DEFAULT_SCORE_THRESHOLD = 3.5


_DIMENSION_RULES: list[tuple[str, str, str]] = [
    (
        "relevance",
        "relevance_miss",
        "Refocus each focus point on the founder prompt and remove tangential themes.",
    ),
    (
        "contradiction",
        "contradiction_blindness",
        "Surface counter-signals and state how they change confidence in each claim.",
    ),
    (
        "coverage",
        "coverage_gap",
        "Cover missing high-signal themes from the evidence set and anchor each to concrete snippets.",
    ),
    (
        "distinctiveness",
        "distinctiveness_gap",
        "Deduplicate overlapping points and preserve only distinct user-signal clusters.",
    ),
    (
        "overall_preference",
        "overall_preference_gap",
        "Strengthen synthesis quality across all dimensions so this output is clearly preferable overall.",
    ),
]


def _severity_from_score(score: float) -> int:
    gap = 5.0 - score
    if gap >= 3.0:
        return 5
    if gap >= 2.0:
        return 4
    if gap >= 1.5:
        return 3
    if gap >= 1.0:
        return 2
    return 1


def _try_load_judge(path: Path) -> JudgeResult | None:
    if not path.exists():
        return None
    return JudgeResult.model_validate_json(path.read_text(encoding="utf-8"))


def classify_judge_result(
    prompt_id: str,
    variant: str,
    judge: JudgeResult,
    *,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> list[FailureTag]:
    """Create deterministic FailureTag entries from one judge artifact."""

    tags: list[FailureTag] = []
    low_score_categories: list[str] = []
    judge_ref = f"{prompt_id}/judge_{variant}.json"

    for dimension, category, guidance in _DIMENSION_RULES:
        score = getattr(judge.scores, dimension)
        if score >= score_threshold:
            continue

        low_score_categories.append(category)
        tags.append(
            FailureTag(
                tag_id=f"ft_{prompt_id}_{variant}_{category}",
                category=category,
                severity=_severity_from_score(score),
                prompt_id=prompt_id,
                description=(
                    f"{variant} scored {score:.1f}/5 on {dimension.replace('_', ' ')} "
                    f"(threshold {score_threshold:.1f}). {guidance}"
                ),
                evidence_refs=[
                    f"{judge_ref}#scores.{dimension}",
                    f"{judge_ref}#scores.rationale",
                    f"{prompt_id}/{variant}.json",
                ],
            )
        )

    overall = judge.scores.overall_preference
    if len(low_score_categories) >= 2 and overall < score_threshold:
        tags.append(
            FailureTag(
                tag_id=f"ft_{prompt_id}_{variant}_overall_quality_drop",
                category="overall_quality_drop",
                severity=_severity_from_score(overall),
                prompt_id=prompt_id,
                description=(
                    f"{variant} has multiple low rubric dimensions "
                    f"({', '.join(low_score_categories)}) with overall {overall:.2f}/5."
                ),
                evidence_refs=[
                    f"{judge_ref}#scores",
                    f"{judge_ref}#scores.rationale",
                ],
            )
        )

    return tags


def generate_failure_tags(
    run_artifacts_dir: Path,
    *,
    prompt_ids: list[str] | None = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> list[FailureTag]:
    """Classify low-quality outputs from per-prompt run artifacts."""

    selected = set(prompt_ids) if prompt_ids else None
    tags: list[FailureTag] = []

    run_dirs = [
        run_dir
        for run_dir in sorted(run_artifacts_dir.iterdir(), key=lambda item: item.name)
        if run_dir.is_dir() and not run_dir.name.startswith("_")
    ]

    for run_dir in run_dirs:
        prompt_id = run_dir.name
        if selected is not None and prompt_id not in selected:
            continue

        for variant in ("baseline", "pipeline"):
            judge = _try_load_judge(run_dir / f"judge_{variant}.json")
            if judge is None:
                continue
            tags.extend(
                classify_judge_result(
                    prompt_id,
                    variant,
                    judge,
                    score_threshold=score_threshold,
                )
            )

    return tags


def write_failure_tags(tags: list[FailureTag], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tags_path = output_dir / "failure_tags.jsonl"
    lines = [tag.model_dump_json() for tag in tags]
    payload = "\n".join(lines)
    if payload:
        payload += "\n"
    tags_path.write_text(payload, encoding="utf-8")
    return tags_path


def _format_ref(ref: str) -> str:
    link_target = ref.split("#", maxsplit=1)[0]
    return f"[{ref}]({link_target})"


def generate_root_cause_report(tags: list[FailureTag], output_dir: Path) -> Path:
    """Write a markdown root-cause summary grouped by category and severity."""

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "failure_taxonomy_report.md"

    lines: list[str] = [
        "# Failure Taxonomy Report\n",
        f"**Failure tags generated:** {len(tags)}\n",
    ]

    if not tags:
        lines.append("No low-quality outputs were tagged for the configured threshold.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    by_category: dict[str, list[FailureTag]] = defaultdict(list)
    severity_counts: dict[int, int] = defaultdict(int)
    for tag in tags:
        by_category[tag.category].append(tag)
        severity_counts[tag.severity] += 1

    lines.append("## Category Overview\n")
    lines.append("| Category | Count | Avg Severity | Max Severity |")
    lines.append("|---|---:|---:|---:|")

    categories = sorted(
        by_category.items(),
        key=lambda item: (-len(item[1]), -max(tag.severity for tag in item[1]), item[0]),
    )
    for category, category_tags in categories:
        avg_severity = sum(tag.severity for tag in category_tags) / len(category_tags)
        max_severity = max(tag.severity for tag in category_tags)
        lines.append(f"| {category} | {len(category_tags)} | {avg_severity:.2f} | {max_severity} |")

    lines.append("")
    lines.append("## Severity Distribution\n")
    lines.append("| Severity | Count |")
    lines.append("|---|---:|")
    for severity in sorted(severity_counts, reverse=True):
        lines.append(f"| {severity} | {severity_counts[severity]} |")

    lines.append("")
    lines.append("## Tagged Instances\n")
    lines.append("| Category | Prompt | Severity | Evidence | Description |")
    lines.append("|---|---|---:|---|---|")

    ordered_tags = sorted(
        tags,
        key=lambda tag: (-tag.severity, tag.category, tag.prompt_id or "", tag.tag_id),
    )
    for tag in ordered_tags:
        refs = ", ".join(_format_ref(ref) for ref in tag.evidence_refs[:2]) or "-"
        prompt_ref = tag.prompt_id or "-"
        lines.append(
            f"| {tag.category} | {prompt_ref} | {tag.severity} | {refs} | {tag.description} |"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def generate_failure_taxonomy(
    run_artifacts_dir: Path,
    *,
    prompt_ids: list[str] | None = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> tuple[list[FailureTag], Path, Path]:
    """Run classification from artifacts and persist tags plus root-cause report."""

    tags = generate_failure_tags(
        run_artifacts_dir,
        prompt_ids=prompt_ids,
        score_threshold=score_threshold,
    )
    tags_path = write_failure_tags(tags, run_artifacts_dir)
    report_path = generate_root_cause_report(tags, run_artifacts_dir)
    return tags, tags_path, report_path

