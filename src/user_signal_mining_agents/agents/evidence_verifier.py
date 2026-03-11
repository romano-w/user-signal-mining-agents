"""Evidence-Grounding Verifier: post-synthesis check for evidence quality."""

from __future__ import annotations

from ..config import Settings, get_settings
from .. import console as con
from ..llm_client import call_llm_json
from ..schemas import EvidenceSnippet, FocusPoint, FounderPrompt, IntentBundle, SynthesisResult


def _load_prompt_template(settings: Settings) -> str:
    path = settings.prompts_dir / "evidence_verifier.md"
    return path.read_text(encoding="utf-8")


def _coerce_to_str(val: object) -> str:
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
    out.setdefault("counter_signal", "No counter-signal identified.")
    out.setdefault("next_validation_question", "What additional data would help validate this finding?")
    out.setdefault("supporting_snippets", [])
    known = {"label", "why_it_matters", "supporting_snippets", "counter_signal", "next_validation_question"}
    out = {k: v for k, v in out.items() if k in known}
    return out


def _format_evidence_block(snippets: list[EvidenceSnippet]) -> str:
    lines: list[str] = []
    for i, snippet in enumerate(snippets, start=1):
        biz = snippet.business_name or snippet.business_id
        stars = f" ({snippet.stars}★)" if snippet.stars else ""
        lines.append(f"[{i}] {biz}{stars}: {snippet.text}")
    return "\n".join(lines)


def _format_focus_points_block(points: list[FocusPoint]) -> str:
    lines: list[str] = []
    for i, fp in enumerate(points, start=1):
        lines.append(f"[{i}] {fp.label}")
        lines.append(f"    Why: {fp.why_it_matters}")
        lines.append(f"    Evidence: {', '.join(fp.supporting_snippets)}")
        lines.append(f"    Counter: {fp.counter_signal}")
        lines.append(f"    Next Q: {fp.next_validation_question}")
    return "\n".join(lines)


def verify_evidence(
    result: SynthesisResult,
    evidence: list[EvidenceSnippet],
    settings: Settings | None = None,
) -> SynthesisResult:
    """Verify and fix evidence grounding in synthesis output."""

    s = settings or get_settings()
    system_prompt = _load_prompt_template(s)

    evidence_block = _format_evidence_block(evidence)
    focus_block = _format_focus_points_block(result.focus_points)

    user_prompt = (
        f"Founder statement:\n{result.prompt.statement}\n\n"
        f"Evidence snippets:\n{evidence_block}\n\n"
        f"Focus points to verify:\n{focus_block}"
    )

    con.step("verifier", "Verifying evidence grounding...")
    raw = call_llm_json(system_prompt=system_prompt, user_prompt=user_prompt, settings=s)

    focus_list = raw if isinstance(raw, list) else raw.get("focus_points", raw)
    verified_points = [FocusPoint.model_validate(_normalize_focus_point(fp)) for fp in focus_list]

    con.success("verifier", f"Verified {len(verified_points)} focus points")

    return SynthesisResult(
        system_variant=result.system_variant,
        prompt=result.prompt,
        intent_bundle=result.intent_bundle,
        retrieved_evidence=result.retrieved_evidence,
        focus_points=verified_points,
    )
