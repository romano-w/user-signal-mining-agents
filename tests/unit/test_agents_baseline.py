from __future__ import annotations

import json
from pathlib import Path

import pytest

from user_signal_mining_agents.agents import baseline
from user_signal_mining_agents.retrieval.index import DenseRetrievalHit


def test_normalize_focus_point_coerces_and_filters_extra_keys() -> None:
    raw = {
        "label": "Wait times",
        "why_it_matters": "Hurts repeat visits",
        "supporting_snippets": [{"text": "Guest waited 40 minutes"}, ["table not ready"]],
        "counter_signal": {"text": "Some guests saw fast service"},
        "next_validation_question": "What hours are worst?",
        "extra": "drop me",
    }

    normalized = baseline._normalize_focus_point(raw)
    assert normalized["supporting_snippets"] == ["Guest waited 40 minutes", "table not ready"]
    assert normalized["counter_signal"] == "Some guests saw fast service"
    assert "extra" not in normalized


def test_run_baseline_calls_dependencies_and_persists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
    founder_prompt,
    evidence_factory,
) -> None:
    hits = [
        DenseRetrievalHit(snippet=evidence_factory(1, text="slow service quote"), score=0.9),
        DenseRetrievalHit(snippet=evidence_factory(2, text="another quote"), score=0.8),
    ]
    monkeypatch.setattr(baseline, "search_dense_index", lambda *_args, **_kwargs: hits)

    llm_payload = {
        "focus_points": [
            {
                "label": "Service latency",
                "why_it_matters": "Guests leave before ordering dessert",
                "supporting_snippets": [{"text": "slow service quote"}],
                "counter_signal": ["fast at lunch"],
                "next_validation_question": "Which stations create backlog?",
            },
            {
                "label": "Order sequencing confusion",
                "why_it_matters": "Creates table idle time",
                "supporting_snippets": ["another quote"],
                "counter_signal": "Some tables report smooth pacing",
                "next_validation_question": "Is host-to-kitchen handoff delayed?",
            },
            {
                "label": "Staffing peaks",
                "why_it_matters": "Weekend demand exceeds staffing",
                "supporting_snippets": ["weekend delay quote"],
                "counter_signal": "Weekday service is stable",
                "next_validation_question": "Can shifts be staggered?",
            },
        ]
    }
    monkeypatch.setattr(baseline, "call_llm_json", lambda **_kwargs: llm_payload)

    result = baseline.run_baseline(founder_prompt, tmp_settings)

    assert result.system_variant == "baseline"
    assert len(result.focus_points) == 3

    output = tmp_settings.run_artifacts_dir / founder_prompt.id / "baseline.json"
    assert output.exists()
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["system_variant"] == "baseline"
    assert persisted["prompt"]["id"] == founder_prompt.id
