from __future__ import annotations

import json

import pytest

from user_signal_mining_agents.agents import judge, pipeline
from user_signal_mining_agents.schemas import JudgeResult, JudgeScores, SynthesisResult


def _make_result(system_variant: str, founder_prompt, focus_point_factory) -> SynthesisResult:
    return SynthesisResult(
        system_variant=system_variant,
        prompt=founder_prompt,
        retrieved_evidence=[],
        focus_points=[focus_point_factory(1), focus_point_factory(2), focus_point_factory(3)],
    )


def _judge_payload(a_value: float, b_value: float) -> dict[str, dict[str, object]]:
    def _scores(v: float) -> dict[str, object]:
        return {
            "relevance": v,
            "actionability": v,
            "evidence_grounding": v,
            "contradiction_handling": v,
            "non_redundancy": v,
            "rationale": f"score {v}",
        }

    return {"system_a": _scores(a_value), "system_b": _scores(b_value)}


def test_judge_pair_maps_scores_when_baseline_is_a(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
    founder_prompt,
    focus_point_factory,
) -> None:
    baseline_result = _make_result("baseline", founder_prompt, focus_point_factory)
    pipeline_result = _make_result("pipeline", founder_prompt, focus_point_factory)

    monkeypatch.setattr(judge.random, "random", lambda: 0.1)
    monkeypatch.setattr(judge, "call_llm_json", lambda **_kwargs: _judge_payload(4.0, 2.0))

    baseline_judge, pipeline_judge = judge.judge_pair(
        founder_prompt,
        baseline_result,
        pipeline_result,
        tmp_settings,
    )

    assert baseline_judge.system_variant == "baseline"
    assert baseline_judge.scores.relevance == 4.0
    assert pipeline_judge.scores.relevance == 2.0


def test_judge_pair_maps_scores_when_pipeline_is_a(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
    founder_prompt,
    focus_point_factory,
) -> None:
    baseline_result = _make_result("baseline", founder_prompt, focus_point_factory)
    pipeline_result = _make_result("pipeline", founder_prompt, focus_point_factory)

    monkeypatch.setattr(judge.random, "random", lambda: 0.9)
    monkeypatch.setattr(judge, "call_llm_json", lambda **_kwargs: _judge_payload(4.0, 2.0))

    baseline_judge, pipeline_judge = judge.judge_pair(
        founder_prompt,
        baseline_result,
        pipeline_result,
        tmp_settings,
    )

    assert baseline_judge.scores.relevance == 2.0
    assert pipeline_judge.scores.relevance == 4.0


def test_judge_pair_rejects_non_dict_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
    founder_prompt,
    focus_point_factory,
) -> None:
    baseline_result = _make_result("baseline", founder_prompt, focus_point_factory)
    pipeline_result = _make_result("pipeline", founder_prompt, focus_point_factory)

    monkeypatch.setattr(judge, "call_llm_json", lambda **_kwargs: [])

    with pytest.raises(ValueError, match="Expected dict"):
        judge.judge_pair(founder_prompt, baseline_result, pipeline_result, tmp_settings)


def test_pipeline_runs_steps_in_order_and_persists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
    founder_prompt,
    intent_bundle,
    evidence_factory,
    synthesis_result,
) -> None:
    calls: list[str] = []

    def _decompose(prompt, settings):
        assert prompt == founder_prompt
        assert settings == tmp_settings
        calls.append("intent")
        return intent_bundle

    def _retrieve(prompt, intent, settings):
        assert prompt == founder_prompt
        assert intent == intent_bundle
        assert settings == tmp_settings
        calls.append("evidence")
        return [evidence_factory(1), evidence_factory(2)]

    def _synthesis(prompt, intent, evidence, settings):
        assert prompt == founder_prompt
        assert intent == intent_bundle
        assert len(evidence) == 2
        assert settings == tmp_settings
        calls.append("synthesis")
        return synthesis_result

    verified = synthesis_result.model_copy(update={"focus_points": synthesis_result.focus_points})

    def _verify(result, evidence, settings):
        assert result == synthesis_result
        assert len(evidence) == 2
        assert settings == tmp_settings
        calls.append("verify")
        return verified

    monkeypatch.setattr(pipeline, "decompose_intent", _decompose)
    monkeypatch.setattr(pipeline, "retrieve_and_filter", _retrieve)
    monkeypatch.setattr(pipeline, "run_synthesis", _synthesis)
    monkeypatch.setattr(pipeline, "verify_evidence", _verify)

    result = pipeline.run_pipeline(founder_prompt, tmp_settings)

    assert calls == ["intent", "evidence", "synthesis", "verify"]
    assert result == verified

    output = tmp_settings.run_artifacts_dir / founder_prompt.id / "pipeline.json"
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["system_variant"] == "pipeline"

def test_judge_named_pair_supports_custom_variant_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
    founder_prompt,
    focus_point_factory,
) -> None:
    control_result = _make_result("control", founder_prompt, focus_point_factory)
    hybrid_result = _make_result("full_hybrid", founder_prompt, focus_point_factory)

    monkeypatch.setattr(judge.random, "random", lambda: 0.1)
    monkeypatch.setattr(judge, "call_llm_json", lambda **_kwargs: _judge_payload(5.0, 3.0))

    control_judge, hybrid_judge = judge.judge_named_pair(
        founder_prompt,
        control_result,
        hybrid_result,
        left_variant="control",
        right_variant="full_hybrid",
        settings=tmp_settings,
    )

    assert control_judge.system_variant == "control"
    assert hybrid_judge.system_variant == "full_hybrid"
    assert control_judge.scores.relevance == 5.0
    assert hybrid_judge.scores.relevance == 3.0
