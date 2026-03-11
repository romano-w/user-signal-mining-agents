"""LLM Judge: score baseline vs pipeline outputs on the 5-dimension rubric."""

from __future__ import annotations

import random

from ..config import Settings, get_settings
from ..llm_client import call_llm_json
from .. import console as con
from ..schemas import (
    FocusPoint,
    FounderPrompt,
    JudgeResult,
    JudgeScores,
    SynthesisResult,
)


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


def judge_pair(
    prompt: FounderPrompt,
    baseline_result: SynthesisResult,
    pipeline_result: SynthesisResult,
    settings: Settings | None = None,
) -> tuple[JudgeResult, JudgeResult]:
    """Score both system outputs for a single founder prompt.

    Randomizes A/B position to remove positional bias, then maps scores back.
    """

    s = settings or get_settings()
    system_prompt = _load_prompt_template(s)

    # Randomize which system is presented as A vs B
    baseline_first = random.random() < 0.5

    if baseline_first:
        a_label, b_label = "System A", "System B"
        a_result, b_result = baseline_result, pipeline_result
        con.step("judge", f"Scoring prompt {prompt.id!r} (A=baseline, B=pipeline)...")
    else:
        a_label, b_label = "System A", "System B"
        a_result, b_result = pipeline_result, baseline_result
        con.step("judge", f"Scoring prompt {prompt.id!r} (A=pipeline, B=baseline)...")

    a_block = _format_focus_points(a_label, a_result.focus_points)
    b_block = _format_focus_points(b_label, b_result.focus_points)

    user_prompt = (
        f"Founder statement:\n{prompt.statement}\n\n"
        f"{a_block}\n\n"
        f"{b_block}\n\n"
        "Score both systems. Return JSON with keys \"system_a\" and \"system_b\", "
        "each containing: relevance, actionability, evidence_grounding, "
        "contradiction_handling, non_redundancy (all 1-5), and rationale."
    )

    raw = call_llm_json(system_prompt=system_prompt, user_prompt=user_prompt, settings=s)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict from judge LLM, got {type(raw).__name__}")

    a_scores = JudgeScores.model_validate(raw["system_a"])
    b_scores = JudgeScores.model_validate(raw["system_b"])

    # Map scores back to correct systems
    if baseline_first:
        baseline_scores, pipeline_scores = a_scores, b_scores
    else:
        baseline_scores, pipeline_scores = b_scores, a_scores

    baseline_judge = JudgeResult(
        prompt_id=prompt.id,
        system_variant="baseline",
        scores=baseline_scores,
    )
    pipeline_judge = JudgeResult(
        prompt_id=prompt.id,
        system_variant="pipeline",
        scores=pipeline_scores,
    )
    return baseline_judge, pipeline_judge
