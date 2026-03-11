"""Counter-evidence miner: proposes contradiction-focused retrieval queries."""

from __future__ import annotations

from ..config import Settings, get_settings
from ..llm_client import call_llm_json
from .. import console as con
from ..schemas import EvidenceSnippet, FounderPrompt, IntentBundle


def _load_prompt_template(settings: Settings) -> str:
    path = settings.prompts_dir / "counterevidence_miner.md"
    return path.read_text(encoding="utf-8")


def _normalize_queries(raw: object) -> list[str]:
    if isinstance(raw, dict):
        values = raw.get("queries") or raw.get("retrieval_queries") or []
    elif isinstance(raw, list):
        values = raw
    else:
        raise ValueError(f"Expected list/dict from counter-evidence LLM, got {type(raw).__name__}")

    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        query = str(item).strip()
        if not query or query in seen:
            continue
        seen.add(query)
        out.append(query)
    return out


def _format_evidence(snippets: list[EvidenceSnippet], limit: int = 10) -> str:
    lines: list[str] = []
    for i, snippet in enumerate(snippets[:limit], start=1):
        biz = snippet.business_name or snippet.business_id
        lines.append(f"[{i}] {biz}: {snippet.text}")
    return "\n".join(lines)


def mine_counterevidence_queries(
    prompt: FounderPrompt,
    intent: IntentBundle,
    evidence: list[EvidenceSnippet],
    settings: Settings | None = None,
) -> list[str]:
    """Generate contradiction-seeking queries from current evidence."""

    s = settings or get_settings()
    system_prompt = _load_prompt_template(s)

    user_prompt = (
        f"Founder statement:\n{prompt.statement}\n\n"
        f"Intent counter hypotheses:\n{intent.counter_hypotheses}\n\n"
        f"Current evidence sample:\n{_format_evidence(evidence)}"
    )

    con.step("counter-miner", f"Mining contradiction queries for prompt {prompt.id!r}...")
    raw = call_llm_json(system_prompt=system_prompt, user_prompt=user_prompt, settings=s)
    queries = _normalize_queries(raw)

    limited = queries[:4]
    con.step("counter-miner", f"Generated {len(limited)} contradiction queries")
    return limited
