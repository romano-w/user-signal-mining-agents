"""Evaluation runner: run both systems + judge across all founder prompts."""

from __future__ import annotations

from pathlib import Path

from ..agents.baseline import run_baseline
from ..agents.judge import judge_pair
from ..agents.pipeline import run_pipeline
from ..config import Settings, get_settings
from ..domain_packs import load_founder_prompts
from .. import console as con
from ..schemas import (
    EvaluationSummary,
    JudgeResult,
    PromptEvaluationPair,
    SynthesisResult,
)


def _try_load_synthesis(path: Path) -> SynthesisResult | None:
    if not path.exists():
        return None
    return SynthesisResult.model_validate_json(path.read_text(encoding="utf-8"))


def _try_load_judge(path: Path) -> JudgeResult | None:
    if not path.exists():
        return None
    return JudgeResult.model_validate_json(path.read_text(encoding="utf-8"))


def run_evaluation(
    settings: Settings | None = None,
    *,
    prompt_ids: list[str] | None = None,
    domain_ids: list[str] | None = None,
    skip_cached: bool = True,
) -> EvaluationSummary:
    """Run baseline + pipeline + judge for each founder prompt."""

    s = settings or get_settings()
    prompts = load_founder_prompts(s, domain_ids=domain_ids)

    if prompt_ids:
        allowed = set(prompt_ids)
        prompts = [p for p in prompts if p.id in allowed]

    con.header(
        "Founder-Grounded Review Mining Evaluation",
        f"{len(prompts)} prompt(s) | model: {s.llm_model} | provider: {s.llm_provider}",
    )

    pairs: list[PromptEvaluationPair] = []

    for i, prompt in enumerate(prompts, start=1):
        run_dir = s.run_artifacts_dir / prompt.id

        # Files
        judge_b_path = run_dir / "judge_baseline.json"
        judge_p_path = run_dir / "judge_pipeline.json"
        baseline_path = run_dir / "baseline.json"
        pipeline_path = run_dir / "pipeline.json"

        con.prompt_table(prompt.id, i, len(prompts))

        # Check if fully complete
        if skip_cached:
            cached_judge_b = _try_load_judge(judge_b_path)
            cached_judge_p = _try_load_judge(judge_p_path)
            if cached_judge_b and cached_judge_p:
                con.cached("COMPLETE", f"Using cached results for {prompt.id}")
                pairs.append(PromptEvaluationPair(
                    prompt=prompt,
                    baseline_scores=cached_judge_b,
                    pipeline_scores=cached_judge_p,
                ))
                continue

        # Baseline
        baseline_result = _try_load_synthesis(baseline_path) if skip_cached else None
        if baseline_result:
            con.cached("baseline", "Using cached result")
        else:
            baseline_result = run_baseline(prompt, s)

        # Pipeline
        pipeline_result = _try_load_synthesis(pipeline_path) if skip_cached else None
        if pipeline_result:
            con.cached("pipeline", "Using cached result")
        else:
            pipeline_result = run_pipeline(prompt, s)

        # Judge
        baseline_judge, pipeline_judge = judge_pair(
            prompt, baseline_result, pipeline_result, s
        )

        # Persist judge results
        run_dir.mkdir(parents=True, exist_ok=True)
        judge_b_path.write_text(
            baseline_judge.model_dump_json(indent=2),
            encoding="utf-8",
        )
        judge_p_path.write_text(
            pipeline_judge.model_dump_json(indent=2),
            encoding="utf-8",
        )

        # Show per-prompt scores
        b = baseline_judge.scores
        p = pipeline_judge.scores
        con.step("scores", f"B={b.overall_avg:.1f} vs P={p.overall_avg:.1f}")

        pairs.append(PromptEvaluationPair(
            prompt=prompt,
            baseline_scores=baseline_judge,
            pipeline_scores=pipeline_judge,
        ))

    return EvaluationSummary(pairs=pairs)
