from __future__ import annotations

import pytest

from user_signal_mining_agents.agents import intent


def test_decompose_intent_validates_dict_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
    founder_prompt,
) -> None:
    monkeypatch.setattr(
        intent,
        "call_llm_json",
        lambda **_kwargs: {
            "problem_keywords": ["slow service"],
            "target_user": "dine-in",
            "usage_context": "weekend",
            "counter_hypotheses": ["short staffed"],
            "retrieval_queries": ["slow service", "long wait"],
        },
    )

    bundle = intent.decompose_intent(founder_prompt, tmp_settings)

    assert bundle.problem_keywords == ["slow service"]
    assert len(bundle.retrieval_queries) == 2


def test_decompose_intent_rejects_non_dict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
    founder_prompt,
) -> None:
    monkeypatch.setattr(intent, "call_llm_json", lambda **_kwargs: [])

    with pytest.raises(ValueError, match="Expected dict"):
        intent.decompose_intent(founder_prompt, tmp_settings)
