"""Variant evaluation runner: compare experimental pipeline variants against control."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..agents.judge import judge_named_pair
from ..agents.variant_pipeline import (
    default_candidate_variants,
    get_variant_spec,
    run_variant_pipeline,
)
from ..config import Settings, get_settings
from ..domain_packs import load_founder_prompts
from .. import console as con
from ..schemas import FounderPrompt, JudgeResult, SynthesisResult


RUBRIC_DIMS = [
    "relevance",
    "actionability",
    "evidence_grounding",
    "contradiction_handling",
    "non_redundancy",
]

DEFAULT_STAGED_PROMPT_IDS = [
    "restaurant-takeout-speed",
    "restaurant-dietary-restriction-trust",
    "restaurant-local-discovery-standout",
]


@dataclass
class VariantPromptComparison:
    prompt: FounderPrompt
    control_scores: JudgeResult
    variant_scores: JudgeResult


@dataclass
class VariantAggregate:
    variant: str
    description: str
    control_scores: dict[str, float]
    variant_scores: dict[str, float]
    control_overall: float
    variant_overall: float
    delta_overall: float


@dataclass
class VariantEvaluationSummary:
    control_variant: str
    prompt_ids: list[str]
    aggregates: list[VariantAggregate] = field(default_factory=list)
    comparisons_by_variant: dict[str, list[VariantPromptComparison]] = field(default_factory=dict)


def _select_prompts(
    prompts: list[FounderPrompt],
    prompt_ids: list[str] | None,
) -> list[FounderPrompt]:
    if prompt_ids:
        allowed = set(prompt_ids)
        selected = [p for p in prompts if p.id in allowed]
        if not selected:
            raise ValueError(f"None of the requested prompt_ids were found: {prompt_ids}")
        return selected

    staged = [p for p in prompts if p.id in set(DEFAULT_STAGED_PROMPT_IDS)]
    if staged:
        return staged

    return prompts[:3]


def _variant_root(settings: Settings) -> Path:
    return settings.run_artifacts_dir.parent / "variant_runs"


def _try_load_synthesis(path: Path) -> SynthesisResult | None:
    if not path.exists():
        return None
    return SynthesisResult.model_validate_json(path.read_text(encoding="utf-8"))


def _try_load_judge(path: Path) -> JudgeResult | None:
    if not path.exists():
        return None
    return JudgeResult.model_validate_json(path.read_text(encoding="utf-8"))


def _run_or_load_variant(
    prompt: FounderPrompt,
    variant: str,
    settings: Settings,
    root: Path,
    *,
    skip_cached: bool,
) -> SynthesisResult:
    synthesis_path = root / variant / prompt.id / "synthesis.json"
    if skip_cached:
        cached = _try_load_synthesis(synthesis_path)
        if cached:
            con.cached("variant", f"[{variant}] Using cached synthesis for {prompt.id}")
            return cached

    return run_variant_pipeline(prompt, variant, settings, output_root=root, persist=True)


def _judge_or_load(
    prompt: FounderPrompt,
    control_result: SynthesisResult,
    variant_result: SynthesisResult,
    variant: str,
    settings: Settings,
    root: Path,
    *,
    skip_cached: bool,
) -> tuple[JudgeResult, JudgeResult]:
    run_dir = root / variant / prompt.id
    judge_control_path = run_dir / "judge_control.json"
    judge_variant_path = run_dir / "judge_variant.json"

    if skip_cached:
        cached_control = _try_load_judge(judge_control_path)
        cached_variant = _try_load_judge(judge_variant_path)
        if cached_control and cached_variant:
            con.cached("judge", f"[{variant}] Using cached judge scores for {prompt.id}")
            return cached_control, cached_variant

    control_judge, variant_judge = judge_named_pair(
        prompt,
        control_result,
        variant_result,
        left_variant="control",
        right_variant=variant,
        settings=settings,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    judge_control_path.write_text(control_judge.model_dump_json(indent=2), encoding="utf-8")
    judge_variant_path.write_text(variant_judge.model_dump_json(indent=2), encoding="utf-8")
    return control_judge, variant_judge


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_variant_evaluation(
    settings: Settings | None = None,
    *,
    variant_ids: list[str] | None = None,
    prompt_ids: list[str] | None = None,
    domain_ids: list[str] | None = None,
    skip_cached: bool = True,
) -> VariantEvaluationSummary:
    """Run control-vs-variant judge comparisons for selected prompts."""

    s = settings or get_settings()
    root = _variant_root(s)

    prompts = _select_prompts(load_founder_prompts(s, domain_ids=domain_ids), prompt_ids)

    selected_variants = variant_ids or default_candidate_variants()
    # Maintain stable ordering while removing duplicates.
    selected_variants = list(dict.fromkeys(v for v in selected_variants if v != "control"))

    for variant in selected_variants:
        get_variant_spec(variant)  # validates name early

    con.header(
        "Variant Evaluation",
        f"{len(prompts)} prompt(s) | {len(selected_variants)} variant(s) | model: {s.llm_model}",
    )

    comparisons_by_variant: dict[str, list[VariantPromptComparison]] = {
        variant: [] for variant in selected_variants
    }

    for i, prompt in enumerate(prompts, start=1):
        con.prompt_table(prompt.id, i, len(prompts))

        control_result = _run_or_load_variant(
            prompt,
            "control",
            s,
            root,
            skip_cached=skip_cached,
        )

        for variant in selected_variants:
            variant_result = _run_or_load_variant(
                prompt,
                variant,
                s,
                root,
                skip_cached=skip_cached,
            )
            control_judge, variant_judge = _judge_or_load(
                prompt,
                control_result,
                variant_result,
                variant,
                s,
                root,
                skip_cached=skip_cached,
            )
            comparisons_by_variant[variant].append(
                VariantPromptComparison(
                    prompt=prompt,
                    control_scores=control_judge,
                    variant_scores=variant_judge,
                )
            )

    aggregates: list[VariantAggregate] = []
    for variant in selected_variants:
        rows = comparisons_by_variant[variant]
        description = get_variant_spec(variant).description

        control_scores = {
            dim: _avg([getattr(row.control_scores.scores, dim) for row in rows])
            for dim in RUBRIC_DIMS
        }
        variant_scores = {
            dim: _avg([getattr(row.variant_scores.scores, dim) for row in rows])
            for dim in RUBRIC_DIMS
        }
        control_overall = _avg(list(control_scores.values()))
        variant_overall = _avg(list(variant_scores.values()))

        aggregates.append(
            VariantAggregate(
                variant=variant,
                description=description,
                control_scores=control_scores,
                variant_scores=variant_scores,
                control_overall=control_overall,
                variant_overall=variant_overall,
                delta_overall=variant_overall - control_overall,
            )
        )

    aggregates.sort(key=lambda item: item.delta_overall, reverse=True)

    return VariantEvaluationSummary(
        control_variant="control",
        prompt_ids=[p.id for p in prompts],
        aggregates=aggregates,
        comparisons_by_variant=comparisons_by_variant,
    )
