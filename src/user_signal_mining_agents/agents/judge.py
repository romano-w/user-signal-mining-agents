"""LLM Judge: score baseline vs pipeline outputs on the 5-dimension rubric."""

from __future__ import annotations

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
    """Score both system outputs for a single founder prompt."""

    s = settings or get_settings()
    system_prompt = _load_prompt_template(s)

    baseline_block = _format_focus_points("System A (Baseline)", baseline_result.focus_points)
    pipeline_block = _format_focus_points("System B (Pipeline)", pipeline_result.focus_points)

    user_prompt = (
        f"Founder statement:\n{prompt.statement}\n\n"
        f"{baseline_block}\n\n"
        f"{pipeline_block}\n\n"
        "Score both systems. Return JSON with keys \"system_a\" and \"system_b\", "
        "each containing: relevance, actionability, evidence_grounding, "
        "contradiction_handling, non_redundancy (all 1-5), and rationale."
    )

    con.step("judge", f"Scoring prompt {prompt.id!r}...")
    raw = call_llm_json(system_prompt=system_prompt, user_prompt=user_prompt, settings=s)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict from judge LLM, got {type(raw).__name__}")

    baseline_judge = JudgeResult(
        prompt_id=prompt.id,
        system_variant="baseline",
        scores=JudgeScores.model_validate(raw["system_a"]),
    )
    pipeline_judge = JudgeResult(
        prompt_id=prompt.id,
        system_variant="pipeline",
        scores=JudgeScores.model_validate(raw["system_b"]),
    )
    return baseline_judge, pipeline_judge
