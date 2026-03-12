from __future__ import annotations

import json

import pytest

from user_signal_mining_agents.agents import variant_pipeline


def test_list_variant_specs_includes_expected_variants() -> None:
    names = [spec.name for spec in variant_pipeline.list_variant_specs()]
    assert names == ["control", "retrieval_hybrid", "critic_loop", "full_hybrid"]


def test_run_variant_pipeline_control_executes_expected_stage_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
    founder_prompt,
    intent_bundle,
    evidence_factory,
    synthesis_result,
) -> None:
    calls: list[str] = []

    def _intent(prompt, settings):
        assert prompt == founder_prompt
        assert settings == tmp_settings
        calls.append("intent")
        return intent_bundle

    def _retrieve(prompt, queries, settings):
        assert prompt == founder_prompt
        assert settings == tmp_settings
        assert queries[0] == founder_prompt.statement
        calls.append("evidence")
        return [evidence_factory(1), evidence_factory(2)]

    def _synth(prompt, intent, evidence, settings):
        assert prompt == founder_prompt
        assert intent == intent_bundle
        assert len(evidence) == 2
        assert settings == tmp_settings
        calls.append("synthesis")
        return synthesis_result

    def _verify(result, evidence, settings):
        assert result == synthesis_result
        assert len(evidence) == 2
        assert settings == tmp_settings
        calls.append("verifier")
        return result

    monkeypatch.setattr(variant_pipeline, "decompose_intent", _intent)
    monkeypatch.setattr(variant_pipeline, "retrieve_for_queries", _retrieve)
    monkeypatch.setattr(variant_pipeline, "run_synthesis", _synth)
    monkeypatch.setattr(variant_pipeline, "verify_evidence", _verify)

    result = variant_pipeline.run_variant_pipeline(founder_prompt, "control", tmp_settings)

    assert calls == ["intent", "evidence", "synthesis", "verifier"]
    assert result.system_variant == "control"

    output = tmp_settings.run_artifacts_dir.parent / "variant_runs" / "control" / founder_prompt.id / "synthesis.json"
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["system_variant"] == "control"


def test_run_variant_pipeline_full_hybrid_includes_all_new_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
    founder_prompt,
    intent_bundle,
    evidence_factory,
    synthesis_result,
    focus_point_factory,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(variant_pipeline, "decompose_intent", lambda *_args, **_kwargs: calls.append("intent") or intent_bundle)
    monkeypatch.setattr(variant_pipeline, "plan_retrieval_queries", lambda *_args, **_kwargs: calls.append("query_planner") or ["query 1"])
    monkeypatch.setattr(variant_pipeline, "retrieve_for_queries", lambda *_args, **_kwargs: calls.append("evidence") or [evidence_factory(1), evidence_factory(2)])
    monkeypatch.setattr(variant_pipeline, "mine_counterevidence_queries", lambda *_args, **_kwargs: calls.append("counterevidence") or ["counter query"])
    monkeypatch.setattr(variant_pipeline, "run_synthesis", lambda *_args, **_kwargs: calls.append("synthesis") or synthesis_result)
    monkeypatch.setattr(variant_pipeline, "critique_focus_points", lambda *_args, **_kwargs: calls.append("critic") or ["improve distinctiveness"])
    monkeypatch.setattr(
        variant_pipeline,
        "refine_focus_points",
        lambda *_args, **_kwargs: calls.append("refiner") or [focus_point_factory(1), focus_point_factory(2), focus_point_factory(3)],
    )
    monkeypatch.setattr(variant_pipeline, "verify_evidence", lambda result, *_args, **_kwargs: calls.append("verifier") or result)

    result = variant_pipeline.run_variant_pipeline(founder_prompt, "full_hybrid", tmp_settings, persist=False)

    assert calls == [
        "intent",
        "query_planner",
        "evidence",
        "counterevidence",
        "evidence",
        "synthesis",
        "critic",
        "refiner",
        "verifier",
    ]
    assert result.system_variant == "full_hybrid"


def test_run_variant_pipeline_rejects_unknown_variant(founder_prompt, tmp_settings) -> None:
    with pytest.raises(ValueError, match="Unknown variant"):
        variant_pipeline.run_variant_pipeline(founder_prompt, "missing", tmp_settings, persist=False)
