from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from user_signal_mining_agents import llm_client
from user_signal_mining_agents.config import Settings


class FakeRateLimitError(Exception):
    pass


class FakeTimeoutError(Exception):
    pass


def _settings(**overrides) -> Settings:
    alias_map = {
        "openai_api_key": "OPENAI_API_KEY",
        "gemini_api_key_1": "GEMINI_API_KEY_1",
        "gemini_api_key_2": "GEMINI_API_KEY_2",
        "openrouter_api_key": "OPENROUTER_API_KEY",
    }
    normalized = dict(overrides)
    for field_name, alias_name in alias_map.items():
        if field_name in normalized:
            normalized[alias_name] = normalized.pop(field_name)

    defaults = dict(
        llm_provider="openai",
        llm_model="model",
        llm_temperature=0.2,
        OPENAI_API_KEY="openai-key",
        GEMINI_API_KEY_1="g1",
        GEMINI_API_KEY_2="g2",
        OPENROUTER_API_KEY="r1",
    )
    defaults.update(normalized)
    return Settings(_env_file=None, **defaults)


def test_get_gemini_keys_filters_empty() -> None:
    settings = _settings(gemini_api_key_1="a", gemini_api_key_2="")
    assert llm_client._get_gemini_keys(settings) == ["a"]


def test_next_gemini_key_rotates() -> None:
    settings = _settings(gemini_api_key_1="a", gemini_api_key_2="b")
    first = llm_client._next_gemini_key(settings)
    second = llm_client._next_gemini_key(settings)
    third = llm_client._next_gemini_key(settings)
    assert (first, second, third) == ("a", "b", "a")


def test_build_client_uses_provider_key_and_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def _openai_ctor(*, api_key: str, base_url: str):
        seen["api_key"] = api_key
        seen["base_url"] = base_url
        return "client"

    monkeypatch.setattr(llm_client.openai, "OpenAI", _openai_ctor)
    monkeypatch.setattr(llm_client, "_next_gemini_key", lambda _s: "gem-key")

    settings = _settings(llm_provider="gemini")
    client = llm_client._build_client(settings)

    assert client == "client"
    assert seen == {
        "api_key": "gem-key",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    }


def test_build_client_honors_custom_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def _openai_ctor(*, api_key: str, base_url: str):
        seen["api_key"] = api_key
        seen["base_url"] = base_url
        return "client"

    monkeypatch.setattr(llm_client.openai, "OpenAI", _openai_ctor)

    settings = _settings(llm_provider="openai", llm_base_url="https://example.com/v1")
    llm_client._build_client(settings)

    assert seen["api_key"] == "openai-key"
    assert seen["base_url"] == "https://example.com/v1"


def test_build_client_raises_when_api_key_missing() -> None:
    settings = _settings(llm_provider="openai", openai_api_key="")
    with pytest.raises(ValueError, match="No API key found"):
        llm_client._build_client(settings)


def test_parse_retry_delay_matches_message() -> None:
    exc = Exception("Please retry in 12.5 s")
    assert llm_client._parse_retry_delay(exc) == 12.5


def test_call_llm_retries_on_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_client.openai, "RateLimitError", FakeRateLimitError)
    monkeypatch.setattr(llm_client.openai, "APITimeoutError", FakeTimeoutError)

    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
    )

    class _Completions:
        def __init__(self):
            self.calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise FakeRateLimitError("retry in 1 s")
            return response

    completions = _Completions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    monkeypatch.setattr(llm_client, "_build_client", lambda _s: fake_client)
    sleeps: list[float] = []
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: sleeps.append(s))

    result = llm_client.call_llm(
        system_prompt="sys",
        user_prompt="user",
        settings=_settings(),
        max_retries=2,
    )

    assert result == "ok"
    assert sleeps == [5]


def test_call_llm_json_strips_markdown_fences(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm_client,
        "call_llm",
        lambda **_kwargs: "```json\n{\"focus_points\": []}\n```",
    )

    payload = llm_client.call_llm_json(system_prompt="s", user_prompt="u", settings=_settings())
    assert payload == {"focus_points": []}


def test_call_llm_json_repairs_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = iter(["not json", "still not json"])
    monkeypatch.setattr(llm_client, "call_llm", lambda **_kwargs: next(calls))
    warnings: list[str] = []
    monkeypatch.setattr(llm_client.con, "warning", lambda msg: warnings.append(msg))

    with pytest.raises(json.JSONDecodeError):
        llm_client.call_llm_json(
            system_prompt="s",
            user_prompt="u",
            settings=_settings(),
            json_attempts=2,
        )

    assert warnings == ["JSON parse failed, retrying LLM call..."]


def test_call_llm_json_succeeds_on_third_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = iter(["not json", "still not json", "{\"focus_points\": []}"])
    monkeypatch.setattr(llm_client, "call_llm", lambda **_kwargs: next(calls))
    warnings: list[str] = []
    monkeypatch.setattr(llm_client.con, "warning", lambda msg: warnings.append(msg))

    payload = llm_client.call_llm_json(system_prompt="s", user_prompt="u", settings=_settings())

    assert payload == {"focus_points": []}
    assert warnings == [
        "JSON parse failed, retrying LLM call...",
        "JSON parse failed, retrying LLM call...",
    ]


def test_repair_json_removes_trailing_commas_and_extracts_json() -> None:
    raw = "Here is output: {\"a\": 1,}\nThanks"
    assert llm_client._repair_json(raw) == '{"a": 1}'
