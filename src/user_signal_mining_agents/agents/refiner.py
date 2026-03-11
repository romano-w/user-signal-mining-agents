"""Refiner agent: rewrites focus points using critic feedback."""

from __future__ import annotations

from ..config import Settings, get_settings
from ..llm_client import call_llm_json
from .. import console as con
from ..schemas import EvidenceSnippet, FocusPoint, FounderPrompt, IntentBundle


def _load_prompt_template(settings: Settings) -> str:
    path = settings.prompts_dir / "refiner.md"
    return path.read_text(encoding="utf-8")


def _format_evidence(snippets: list[EvidenceSnippet], limit: int = 12) -> str:
    lines: list[str] = []
    for i, snippet in enumerate(snippets[:limit], start=1):
        biz = snippet.business_name or snippet.business_id
        lines.append(f"[{i}] {biz}: {snippet.text}")
    return "\n".join(lines)


def _format_focus_points(points: list[FocusPoint]) -> str:
    lines: list[str] = []
    for i, fp in enumerate(points, start=1):
        lines.append(f"[{i}] {fp.label}")
        lines.append(f"    Why: {fp.why_it_matters}")
        lines.append(f"    Evidence: {', '.join(fp.supporting_snippets)}")
        lines.append(f"    Counter: {fp.counter_signal}")
        lines.append(f"    Next Q: {fp.next_validation_question}")
    return "\n".join(lines)


def _coerce_to_str(val: object) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return str(val.get("text", val))
    if isinstance(val, list):
        return " ".join(_coerce_to_str(v) for v in val)
    return str(val)


def _normalize_focus_point(raw: dict) -> dict:
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
    return {k: v for k, v in out.items() if k in known}


def refine_focus_points(
    prompt: FounderPrompt,
    intent: IntentBundle,
    evidence: list[EvidenceSnippet],
    focus_points: list[FocusPoint],
    critic_feedback: list[str],
    settings: Settings | None = None,
) -> list[FocusPoint]:
    """Rewrite focus points using critic feedback while staying evidence-grounded."""

    if not critic_feedback:
        return focus_points

    s = settings or get_settings()
    system_prompt = _load_prompt_template(s)

    feedback_block = "\n".join(f"- {item}" for item in critic_feedback)
    user_prompt = (
        f"Founder statement:\n{prompt.statement}\n\n"
        f"Intent keywords:\n{intent.problem_keywords}\n\n"
        f"Evidence sample:\n{_format_evidence(evidence)}\n\n"
        f"Current focus points:\n{_format_focus_points(focus_points)}\n\n"
        f"Critic feedback:\n{feedback_block}"
    )

    con.step("refiner", f"Refining draft for prompt {prompt.id!r}...")
    raw = call_llm_json(system_prompt=system_prompt, user_prompt=user_prompt, settings=s)
    focus_list = raw if isinstance(raw, list) else raw.get("focus_points", raw)
    refined = [FocusPoint.model_validate(_normalize_focus_point(fp)) for fp in focus_list]
    con.step("refiner", f"Produced {len(refined)} refined focus points")
    return refined
