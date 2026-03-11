"""Intent agent: decompose a founder statement into an IntentBundle."""

from __future__ import annotations

from ..config import Settings, get_settings
from ..llm_client import call_llm_json
from .. import console as con
from ..schemas import FounderPrompt, IntentBundle


def _load_prompt_template(settings: Settings) -> str:
    path = settings.prompts_dir / "intent.md"
    return path.read_text(encoding="utf-8")


def decompose_intent(
    prompt: FounderPrompt,
    settings: Settings | None = None,
) -> IntentBundle:
    """Decompose a founder prompt into keywords, queries, and counter-hypotheses."""

    s = settings or get_settings()
    system_prompt = _load_prompt_template(s)
    user_prompt = f"Founder statement:\n{prompt.statement}"

    con.step("intent", f"Decomposing intent for prompt {prompt.id!r}...")
    raw = call_llm_json(system_prompt=system_prompt, user_prompt=user_prompt, settings=s)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict from intent LLM, got {type(raw).__name__}")

    return IntentBundle.model_validate(raw)
