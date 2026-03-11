"""Synthesis agent: grounded multi-step synthesis from intent + evidence."""

from __future__ import annotations

from ..config import Settings, get_settings
from .. import console as con
from ..llm_client import call_llm_json
from ..schemas import (
    EvidenceSnippet,
    FocusPoint,
    FounderPrompt,
    IntentBundle,
    SynthesisResult,
)


def _load_prompt_template(settings: Settings) -> str:
    path = settings.prompts_dir / "synthesis.md"
    return path.read_text(encoding="utf-8")


def _format_intent_block(intent: IntentBundle) -> str:
    lines = [
        f"Keywords: {', '.join(intent.problem_keywords)}",
    ]
    if intent.target_user:
        lines.append(f"Target user: {intent.target_user}")
    if intent.usage_context:
        lines.append(f"Usage context: {intent.usage_context}")
    if intent.counter_hypotheses:
        lines.append("Counter-hypotheses:")
        for ch in intent.counter_hypotheses:
            lines.append(f"  - {ch}")
    return "\n".join(lines)


def _format_evidence_block(snippets: list[EvidenceSnippet]) -> str:
    lines: list[str] = []
    for i, snippet in enumerate(snippets, start=1):
        biz = snippet.business_name or snippet.business_id
        stars = f" ({snippet.stars}★)" if snippet.stars else ""
        score = f" [relevance={snippet.relevance_score:.3f}]" if snippet.relevance_score else ""
        lines.append(f"[{i}] {biz}{stars}{score}: {snippet.text}")
    return "\n".join(lines)


def _coerce_to_str(val: object) -> str:
    """Coerce various LLM output shapes to a plain string."""
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return str(val.get("text", val))
    if isinstance(val, list):
        return " ".join(_coerce_to_str(v) for v in val)
    return str(val)


def _normalize_focus_point(raw: dict) -> dict:
    """Fix common LLM response quirks before Pydantic validation."""
    out = dict(raw)
    cs = out.get("counter_signal")
    if cs is not None and not isinstance(cs, str):
        out["counter_signal"] = _coerce_to_str(cs)
    ss = out.get("supporting_snippets")
    if isinstance(ss, list):
        out["supporting_snippets"] = [_coerce_to_str(s) for s in ss][:5]
    known = {"label", "why_it_matters", "supporting_snippets", "counter_signal", "next_validation_question"}
    out = {k: v for k, v in out.items() if k in known}
    return out


def run_synthesis(
    prompt: FounderPrompt,
    intent: IntentBundle,
    evidence: list[EvidenceSnippet],
    settings: Settings | None = None,
) -> SynthesisResult:
    """Synthesize focus points from intent-decomposed evidence."""

    s = settings or get_settings()
    system_prompt = _load_prompt_template(s)

    intent_block = _format_intent_block(intent)
    evidence_block = _format_evidence_block(evidence)
    user_prompt = (
        f"Founder statement:\n{prompt.statement}\n\n"
        f"Intent decomposition:\n{intent_block}\n\n"
        f"Evidence snippets:\n{evidence_block}"
    )

    con.step("synthesis", f"Calling LLM for prompt {prompt.id!r}...")
    raw = call_llm_json(system_prompt=system_prompt, user_prompt=user_prompt, settings=s)

    focus_list = raw if isinstance(raw, list) else raw.get("focus_points", raw)
    focus_points = [FocusPoint.model_validate(_normalize_focus_point(fp)) for fp in focus_list]

    return SynthesisResult(
        system_variant="pipeline",
        prompt=prompt,
        intent_bundle=intent,
        retrieved_evidence=evidence,
        focus_points=focus_points,
    )
