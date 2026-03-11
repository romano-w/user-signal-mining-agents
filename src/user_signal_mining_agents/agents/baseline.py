"""Baseline system: single-shot founder statement → focus points."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import Settings, get_settings
from ..llm_client import call_llm_json
from ..retrieval.index import search_dense_index
from ..schemas import (
    EvidenceSnippet,
    FocusPoint,
    FounderPrompt,
    SynthesisResult,
)


def _load_prompt_template(settings: Settings) -> str:
    path = settings.prompts_dir / "baseline.md"
    return path.read_text(encoding="utf-8")


def _format_evidence_block(snippets: list[EvidenceSnippet]) -> str:
    lines: list[str] = []
    for i, snippet in enumerate(snippets, start=1):
        biz = snippet.business_name or snippet.business_id
        stars = f" ({snippet.stars}★)" if snippet.stars else ""
        lines.append(f"[{i}] {biz}{stars}: {snippet.text}")
    return "\n".join(lines)


def _coerce_to_str(val: object) -> str:
    """Coerce various LLM output shapes to a plain string."""
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        # e.g. {"id": "15", "text": "some quote"} → use the text value
        return str(val.get("text", val))
    if isinstance(val, list):
        return " ".join(_coerce_to_str(v) for v in val)
    return str(val)


def _normalize_focus_point(raw: dict) -> dict:
    """Fix common LLM response quirks before Pydantic validation."""
    out = dict(raw)
    # counter_signal: may arrive as str, list, dict, or list-of-dicts
    cs = out.get("counter_signal")
    if cs is not None and not isinstance(cs, str):
        out["counter_signal"] = _coerce_to_str(cs)
    # supporting_snippets: coerce each item to str and cap at 5
    ss = out.get("supporting_snippets")
    if isinstance(ss, list):
        out["supporting_snippets"] = [_coerce_to_str(s) for s in ss][:5]
    # Drop any extra keys Gemini invented that Pydantic would reject
    known = {"label", "why_it_matters", "supporting_snippets", "counter_signal", "next_validation_question"}
    out = {k: v for k, v in out.items() if k in known}
    return out


def run_baseline(
    prompt: FounderPrompt,
    settings: Settings | None = None,
) -> SynthesisResult:
    """Run the zero-shot baseline for a single founder prompt."""

    s = settings or get_settings()
    system_prompt = _load_prompt_template(s)

    # 1. Retrieve top-K snippets using the raw founder statement
    hits = search_dense_index(
        prompt.statement,
        index_dir=s.index_dir,
        top_k=s.retrieval_top_k,
    )
    evidence = [hit.snippet for hit in hits]

    # 2. Build user prompt
    evidence_block = _format_evidence_block(evidence)
    user_prompt = (
        f"Founder statement:\n{prompt.statement}\n\n"
        f"Evidence snippets:\n{evidence_block}"
    )

    # 3. Call LLM
    print(f"  [baseline] Calling LLM for prompt {prompt.id!r}...")
    raw = call_llm_json(system_prompt=system_prompt, user_prompt=user_prompt, settings=s)

    # 4. Parse focus points
    focus_list = raw if isinstance(raw, list) else raw.get("focus_points", raw)
    focus_points = [FocusPoint.model_validate(_normalize_focus_point(fp)) for fp in focus_list]

    result = SynthesisResult(
        system_variant="baseline",
        prompt=prompt,
        retrieved_evidence=evidence,
        focus_points=focus_points,
    )

    # 5. Persist
    run_dir = s.run_artifacts_dir / prompt.id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "baseline.json"
    output_path.write_text(
        result.model_dump_json(indent=2, exclude_none=True),
        encoding="utf-8",
    )
    print(f"  [baseline] Saved -> {output_path}")
    return result
