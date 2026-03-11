"""Critic agent: evaluates draft focus points and returns targeted revision notes."""

from __future__ import annotations

from ..config import Settings, get_settings
from ..llm_client import call_llm_json
from .. import console as con
from ..schemas import EvidenceSnippet, FocusPoint, FounderPrompt, IntentBundle


def _load_prompt_template(settings: Settings) -> str:
    path = settings.prompts_dir / "critic.md"
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


def _normalize_feedback(raw: object) -> list[str]:
    if isinstance(raw, dict):
        values = raw.get("feedback") or raw.get("criticisms") or []
    elif isinstance(raw, list):
        values = raw
    else:
        raise ValueError(f"Expected list/dict from critic LLM, got {type(raw).__name__}")

    out: list[str] = []
    for item in values:
        text = str(item).strip()
        if text:
            out.append(text)
    return out[:6]


def critique_focus_points(
    prompt: FounderPrompt,
    intent: IntentBundle,
    evidence: list[EvidenceSnippet],
    focus_points: list[FocusPoint],
    settings: Settings | None = None,
) -> list[str]:
    """Return concise revision notes for weak spots in a draft synthesis."""

    s = settings or get_settings()
    system_prompt = _load_prompt_template(s)

    user_prompt = (
        f"Founder statement:\n{prompt.statement}\n\n"
        f"Intent keywords:\n{intent.problem_keywords}\n\n"
        f"Evidence sample:\n{_format_evidence(evidence)}\n\n"
        f"Draft focus points:\n{_format_focus_points(focus_points)}"
    )

    con.step("critic", f"Reviewing draft focus points for prompt {prompt.id!r}...")
    raw = call_llm_json(system_prompt=system_prompt, user_prompt=user_prompt, settings=s)
    feedback = _normalize_feedback(raw)
    con.step("critic", f"Returned {len(feedback)} revision notes")
    return feedback
