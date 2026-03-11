"""Query planner agent: expands retrieval queries beyond the base intent bundle."""

from __future__ import annotations

from ..config import Settings, get_settings
from ..llm_client import call_llm_json
from .. import console as con
from ..schemas import FounderPrompt, IntentBundle


def _load_prompt_template(settings: Settings) -> str:
    path = settings.prompts_dir / "query_planner.md"
    return path.read_text(encoding="utf-8")


def _normalize_queries(raw: object) -> list[str]:
    if isinstance(raw, dict):
        values = raw.get("queries") or raw.get("retrieval_queries") or []
    elif isinstance(raw, list):
        values = raw
    else:
        raise ValueError(f"Expected list/dict from query planner LLM, got {type(raw).__name__}")

    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        query = str(item).strip()
        if not query or query in seen:
            continue
        seen.add(query)
        out.append(query)
    return out


def plan_retrieval_queries(
    prompt: FounderPrompt,
    intent: IntentBundle,
    settings: Settings | None = None,
) -> list[str]:
    """Generate additional retrieval queries for broader evidence coverage."""

    s = settings or get_settings()
    system_prompt = _load_prompt_template(s)

    intent_lines = [
        f"Problem keywords: {', '.join(intent.problem_keywords) if intent.problem_keywords else 'n/a'}",
        f"Target user: {intent.target_user or 'n/a'}",
        f"Usage context: {intent.usage_context or 'n/a'}",
        "Counter hypotheses:",
    ]
    for hypothesis in intent.counter_hypotheses:
        intent_lines.append(f"- {hypothesis}")

    intent_block = "\n".join(intent_lines)
    user_prompt = (
        f"Founder statement:\n{prompt.statement}\n\n"
        f"Intent bundle:\n{intent_block}\n\n"
        f"Existing retrieval queries:\n{intent.retrieval_queries}"
    )

    con.step("query-planner", f"Planning query expansion for prompt {prompt.id!r}...")
    raw = call_llm_json(system_prompt=system_prompt, user_prompt=user_prompt, settings=s)
    extra_queries = _normalize_queries(raw)

    # Keep this bounded to avoid runaway retrieval fan-out.
    limited = extra_queries[:6]
    con.step("query-planner", f"Generated {len(limited)} additional queries")
    return limited
