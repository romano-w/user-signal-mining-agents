"""Evaluation runner: run both systems + judge across all founder prompts."""

from __future__ import annotations

from pathlib import Path

from .. import console as con
from ..agents.baseline import run_baseline
from ..agents.judge import judge_pair, judge_panel_pair
from ..agents.pipeline import run_pipeline
from ..config import Settings, get_settings
from ..domain_packs import load_founder_prompts
from ..schemas import (
    EvaluationSummary,
    JudgePanelResult,
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


def _try_load_judge_panel(path: Path) -> JudgePanelResult | None:
    if not path.exists():
        return None
    return JudgePanelResult.model_validate_json(path.read_text(encoding="utf-8"))


def _judge_from_panel(panel: JudgePanelResult) -> JudgeResult:
    return JudgeResult(
        prompt_id=panel.prompt_id,
        system_variant=panel.system_variant,
        scores=panel.aggregate_scores,
    )


def run_evaluation(
    settings: Settings | None = None,
    *,
    prompt_ids: list[str] | None = None,
    domain_ids: list[str] | None = None,
    skip_cached: bool = True,
) -> EvaluationSummary:
    """Run baseline + pipeline + judge for each founder prompt."""

    s = settings or get_settings()
    panel_size = max(1, s.judge_panel_size)
    panel_mode = panel_size > 1

    prompts = load_founder_prompts(s, domain_ids=domain_ids)
    if prompt_ids:
        allowed = set(prompt_ids)
        prompts = [p for p in prompts if p.id in allowed]

    mode_label = f"panel={panel_size}" if panel_mode else "single-judge"
    con.header(
        "Founder-Grounded Review Mining Evaluation",
        f"{len(prompts)} prompt(s) | model: {s.llm_model} | provider: {s.llm_provider} | {mode_label}",
    )

    pairs: list[PromptEvaluationPair] = []

    for i, prompt in enumerate(prompts, start=1):
        run_dir = s.run_artifacts_dir / prompt.id

        judge_b_path = run_dir / "judge_baseline.json"
        judge_p_path = run_dir / "judge_pipeline.json"
        panel_b_path = run_dir / "judge_panel_baseline.json"
        panel_p_path = run_dir / "judge_panel_pipeline.json"
        baseline_path = run_dir / "baseline.json"
        pipeline_path = run_dir / "pipeline.json"

        con.prompt_table(prompt.id, i, len(prompts))

        if skip_cached:
            if panel_mode:
                cached_panel_b = _try_load_judge_panel(panel_b_path)
                cached_panel_p = _try_load_judge_panel(panel_p_path)
                if cached_panel_b and cached_panel_p:
                    con.cached("COMPLETE", f"Using cached panel results for {prompt.id}")
                    pairs.append(
                        PromptEvaluationPair(
                            prompt=prompt,
                            baseline_scores=_judge_from_panel(cached_panel_b),
                            pipeline_scores=_judge_from_panel(cached_panel_p),
                            baseline_panel=cached_panel_b,
                            pipeline_panel=cached_panel_p,
                        )
                    )
                    continue
            else:
                cached_judge_b = _try_load_judge(judge_b_path)
                cached_judge_p = _try_load_judge(judge_p_path)
                if cached_judge_b and cached_judge_p:
                    con.cached("COMPLETE", f"Using cached results for {prompt.id}")
                    pairs.append(
                        PromptEvaluationPair(
                            prompt=prompt,
                            baseline_scores=cached_judge_b,
                            pipeline_scores=cached_judge_p,
                        )
                    )
                    continue

        baseline_result = _try_load_synthesis(baseline_path) if skip_cached else None
        if baseline_result:
            con.cached("baseline", "Using cached result")
        else:
            baseline_result = run_baseline(prompt, s)

        pipeline_result = _try_load_synthesis(pipeline_path) if skip_cached else None
        if pipeline_result:
            con.cached("pipeline", "Using cached result")
        else:
            pipeline_result = run_pipeline(prompt, s)

        run_dir.mkdir(parents=True, exist_ok=True)

        if panel_mode:
            baseline_panel, pipeline_panel = judge_panel_pair(
                prompt,
                baseline_result,
                pipeline_result,
                panel_size=panel_size,
                settings=s,
            )
            baseline_judge = _judge_from_panel(baseline_panel)
            pipeline_judge = _judge_from_panel(pipeline_panel)

            panel_b_path.write_text(baseline_panel.model_dump_json(indent=2), encoding="utf-8")
            panel_p_path.write_text(pipeline_panel.model_dump_json(indent=2), encoding="utf-8")
            judge_b_path.write_text(baseline_judge.model_dump_json(indent=2), encoding="utf-8")
            judge_p_path.write_text(pipeline_judge.model_dump_json(indent=2), encoding="utf-8")

            b = baseline_judge.scores
            p = pipeline_judge.scores
            con.step("scores", f"B={b.overall_avg:.1f} vs P={p.overall_avg:.1f} | panel={panel_size}")

            pairs.append(
                PromptEvaluationPair(
                    prompt=prompt,
                    baseline_scores=baseline_judge,
                    pipeline_scores=pipeline_judge,
                    baseline_panel=baseline_panel,
                    pipeline_panel=pipeline_panel,
                )
            )
            continue

        baseline_judge, pipeline_judge = judge_pair(prompt, baseline_result, pipeline_result, s)

        judge_b_path.write_text(
            baseline_judge.model_dump_json(indent=2),
            encoding="utf-8",
        )
        judge_p_path.write_text(
            pipeline_judge.model_dump_json(indent=2),
            encoding="utf-8",
        )

        b = baseline_judge.scores
        p = pipeline_judge.scores
        con.step("scores", f"B={b.overall_avg:.1f} vs P={p.overall_avg:.1f}")

        pairs.append(
            PromptEvaluationPair(
                prompt=prompt,
                baseline_scores=baseline_judge,
                pipeline_scores=pipeline_judge,
            )
        )

    return EvaluationSummary(pairs=pairs)
