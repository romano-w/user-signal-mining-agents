from __future__ import annotations

import pytest

from user_signal_mining_agents.agents import evidence_verifier, synthesis


def test_format_intent_block_includes_optional_fields(intent_bundle) -> None:
    block = synthesis._format_intent_block(intent_bundle)
    assert "Keywords: slow service, wait time" in block
    assert "Target user: dine-in guests" in block
    assert "Counter-hypotheses:" in block


def test_synthesis_normalize_focus_point_defaults() -> None:
    normalized = synthesis._normalize_focus_point(
        {
            "label": "Label",
            "why_it_matters": "Impact",
            "supporting_snippets": [{"text": "quote"}],
            "counter_signal": ["counter"],
            "extra": "drop",
        }
    )
    assert normalized["supporting_snippets"] == ["quote"]
    assert normalized["counter_signal"] == "counter"
    assert "extra" not in normalized
    assert "next_validation_question" in normalized


def test_run_synthesis_returns_pipeline_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
    founder_prompt,
    intent_bundle,
    evidence_factory,
) -> None:
    evidence = [evidence_factory(1, score=0.91), evidence_factory(2, score=0.82)]
    monkeypatch.setattr(
        synthesis,
        "call_llm_json",
        lambda **_kwargs: [
            {
                "label": "Service bottleneck",
                "why_it_matters": "Drives abandonment",
                "supporting_snippets": ["quote 1"],
                "counter_signal": "Some guests mention speed",
                "next_validation_question": "Where is queue longest?",
            },
            {
                "label": "Pickup friction",
                "why_it_matters": "Hurts reliability",
                "supporting_snippets": ["quote 2"],
                "counter_signal": "Some pickups are smooth",
                "next_validation_question": "How often are bags mislabeled?",
            },
            {
                "label": "Staff handoff",
                "why_it_matters": "Adds idle time",
                "supporting_snippets": ["quote 3"],
                "counter_signal": "Some shifts are staffed",
                "next_validation_question": "Can runner role be added?",
            },
        ],
    )

    result = synthesis.run_synthesis(founder_prompt, intent_bundle, evidence, tmp_settings)

    assert result.system_variant == "pipeline"
    assert result.intent_bundle == intent_bundle
    assert len(result.focus_points) == 3


def test_verify_evidence_rewrites_focus_points(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
    synthesis_result,
    evidence_factory,
) -> None:
    evidence = [evidence_factory(1), evidence_factory(2)]
    monkeypatch.setattr(
        evidence_verifier,
        "call_llm_json",
        lambda **_kwargs: {
            "focus_points": [
                {
                    "label": "Focus 1",
                    "why_it_matters": "Reason 1",
                    "supporting_snippets": [{"text": "Updated quote"}],
                    "counter_signal": "Counter 1",
                    "next_validation_question": "Question 1",
                },
                {
                    "label": "Focus 2",
                    "why_it_matters": "Reason 2",
                    "supporting_snippets": ["Quote 2"],
                    "counter_signal": "Counter 2",
                    "next_validation_question": "Question 2",
                },
                {
                    "label": "Focus 3",
                    "why_it_matters": "Reason 3",
                    "supporting_snippets": ["Quote 3"],
                    "counter_signal": "Counter 3",
                    "next_validation_question": "Question 3",
                },
            ]
        },
    )

    verified = evidence_verifier.verify_evidence(synthesis_result, evidence, tmp_settings)

    assert verified.prompt == synthesis_result.prompt
    assert verified.intent_bundle == synthesis_result.intent_bundle
    assert verified.focus_points[0].supporting_snippets == ["Updated quote"]


def test_verify_evidence_falls_back_to_original_supporting_snippets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
    synthesis_result,
    evidence_factory,
) -> None:
    evidence = [evidence_factory(1), evidence_factory(2)]
    monkeypatch.setattr(
        evidence_verifier,
        "call_llm_json",
        lambda **_kwargs: {
            "focus_points": [
                {
                    "label": "Focus 1",
                    "why_it_matters": "Reason 1",
                    "supporting_snippets": [],
                    "counter_signal": "Counter 1",
                    "next_validation_question": "Question 1",
                },
                {
                    "label": "Focus 2",
                    "why_it_matters": "Reason 2",
                    "supporting_snippets": ["Quote 2"],
                    "counter_signal": "Counter 2",
                    "next_validation_question": "Question 2",
                },
                {
                    "label": "Focus 3",
                    "why_it_matters": "Reason 3",
                    "supporting_snippets": ["Quote 3"],
                    "counter_signal": "Counter 3",
                    "next_validation_question": "Question 3",
                },
            ]
        },
    )

    verified = evidence_verifier.verify_evidence(synthesis_result, evidence, tmp_settings)

    assert verified.focus_points[0].supporting_snippets == synthesis_result.focus_points[0].supporting_snippets
