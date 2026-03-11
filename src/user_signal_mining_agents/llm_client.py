"""LLM client with multi-provider support and Gemini dual-key rotation."""

from __future__ import annotations

import itertools
import json
import re
import time
from typing import Any

import openai

from .config import Settings, get_settings
from . import console as con


_PROVIDER_BASE_URLS: dict[str, str] = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

# Round-robin key cycler for Gemini — shared across all calls in this process.
_gemini_key_cycle: itertools.cycle[str] | None = None


def _get_gemini_keys(s: Settings) -> list[str]:
    """Collect all non-empty Gemini API keys."""
    keys = []
    if s.gemini_api_key_1:
        keys.append(s.gemini_api_key_1)
    if s.gemini_api_key_2:
        keys.append(s.gemini_api_key_2)
    return keys


def _next_gemini_key(s: Settings) -> str:
    """Return the next Gemini key in round-robin order."""
    global _gemini_key_cycle
    if _gemini_key_cycle is None:
        keys = _get_gemini_keys(s)
        if not keys:
            raise ValueError(
                "No Gemini API keys found. Set GEMINI_API_KEY_1 (and optionally GEMINI_API_KEY_2) in .env."
            )
        _gemini_key_cycle = itertools.cycle(keys)
        key_count = len(keys)
        con.step("LLM", f"Gemini key rotation: {key_count} key(s) available")
    return next(_gemini_key_cycle)


def _build_client(settings: Settings, *, api_key_override: str | None = None) -> openai.OpenAI:
    s = settings
    provider = s.llm_provider.lower()

    base_url = s.llm_base_url or _PROVIDER_BASE_URLS.get(provider, "https://api.openai.com/v1")

    if api_key_override:
        api_key = api_key_override
    elif provider == "gemini":
        api_key = _next_gemini_key(s)
    elif provider == "openrouter":
        api_key = s.openrouter_api_key
    elif provider == "openai":
        api_key = s.openai_api_key
    else:
        api_key = s.openai_api_key

    if not api_key:
        raise ValueError(
            f"No API key found for provider {provider!r}. "
            f"Set the appropriate key in .env."
        )

    return openai.OpenAI(api_key=api_key, base_url=base_url)


def _parse_retry_delay(exc: Exception) -> float | None:
    """Try to extract the server-suggested retry delay from an error message."""
    msg = str(exc)
    m = re.search(r"(?:retry\s*(?:in|Delay)['\"]?[:\s]*['\"]?)(\d+(?:\.\d+)?)\s*s", msg, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def call_llm(
    *,
    system_prompt: str,
    user_prompt: str,
    settings: Settings | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_retries: int = 6,
) -> str:
    """Send a plain-text chat completion and return the assistant message."""

    s = settings or get_settings()
    resolved_model = model or s.llm_model
    resolved_temperature = temperature if temperature is not None else s.llm_temperature

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(1, max_retries + 1):
        # Build a fresh client each attempt so Gemini rotates keys
        client = _build_client(s)
        try:
            response = client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                temperature=resolved_temperature,
            )
            usage = response.usage
            if usage:
                con.llm_tokens(resolved_model, usage.prompt_tokens, usage.completion_tokens)
            return response.choices[0].message.content or ""

        except (openai.RateLimitError, openai.APITimeoutError) as exc:
            if attempt == max_retries:
                raise
            server_delay = _parse_retry_delay(exc)
            # With key rotation, try the other key quickly first (5s),
            # then fall back to longer waits if both keys are exhausted
            if server_delay and server_delay > 10:
                wait = server_delay + 2  # respect server hint
            else:
                wait = min(5 * attempt, 60)  # 5, 10, 15, 20... capped at 60
            con.llm_rate_limited(wait, attempt, max_retries)
            time.sleep(wait)

    return ""  # unreachable


def call_llm_json(
    *,
    system_prompt: str,
    user_prompt: str,
    settings: Settings | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_retries: int = 6,
) -> dict[str, Any] | list[Any]:
    """Call the LLM and parse the response as JSON.

    Strips markdown fences if the model wraps JSON in ```json ... ```.
    """

    raw = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        settings=settings,
        model=model,
        temperature=temperature,
        max_retries=max_retries,
    )

    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return json.loads(text)
