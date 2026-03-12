from __future__ import annotations

import json
from pathlib import Path

import pytest

from user_signal_mining_agents.evaluation import variant_report, variant_runner
from user_signal_mining_agents.schemas import (
    FocusPoint,
    FounderPrompt,
    JudgeResult,
    JudgeScores,
    SynthesisResult,
)


def _focus_points() -> list[FocusPoint]:
    return [
        FocusPoint(
            label="L1",
            why_it_matters="W1",
            supporting_snippets=["Q1"],
            counter_signal="C1",
            next_validation_question="N1",
        ),
        FocusPoint(
            label="L2",
            why_it_matters="W2",
            supporting_snippets=["Q2"],
            counter_signal="C2",
            next_validation_question="N2",
        ),
        FocusPoint(
            label="L3",
            why_it_matters="W3",
            supporting_snippets=["Q3"],
            counter_signal="C3",
            next_validation_question="N3",
        ),
    ]


def _synthesis(prompt: FounderPrompt, variant: str) -> SynthesisResult:
    return SynthesisResult(
        system_variant=variant,
        prompt=prompt,
        retrieved_evidence=[],
        focus_points=_focus_points(),
    )


def _scores(value: float) -> JudgeScores:
    return JudgeScores(
        relevance=value,
        overall_preference=value,
        coverage=value,
        contradiction=value,
        distinctiveness=value,
        rationale=f"score {value}",
    )


def _judge(prompt_id: str, variant: str, value: float) -> JudgeResult:
    return JudgeResult(prompt_id=prompt_id, system_variant=variant, scores=_scores(value))


def test_run_variant_evaluation_runs_pairwise_and_persists_judge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
) -> None:
    prompt = FounderPrompt(id="p1", statement="Question", domain="restaurants")
    tmp_settings.founder_prompts_path.write_text(json.dumps([prompt.model_dump()]), encoding="utf-8")

    monkeypatch.setattr(variant_runner, "default_candidate_variants", lambda: ["retrieval_hybrid"])

    def _run_variant(prompt_obj, variant, *_args, **_kwargs):
        return _synthesis(prompt_obj, variant)

    monkeypatch.setattr(variant_runner, "run_variant_pipeline", _run_variant)
    monkeypatch.setattr(
        variant_runner,
        "judge_named_pair",
        lambda prompt_obj, *_args, **_kwargs: (
            _judge(prompt_obj.id, "control", 3.0),
            _judge(prompt_obj.id, "retrieval_hybrid", 4.0),
        ),
    )

    summary = variant_runner.run_variant_evaluation(
        tmp_settings,
        variant_ids=["retrieval_hybrid"],
        prompt_ids=["p1"],
        skip_cached=False,
    )

    assert summary.prompt_ids == ["p1"]
    assert len(summary.aggregates) == 1
    assert summary.aggregates[0].variant == "retrieval_hybrid"
    assert summary.aggregates[0].delta_overall == pytest.approx(1.0)

    judge_control = tmp_settings.run_artifacts_dir.parent / "variant_runs" / "retrieval_hybrid" / "p1" / "judge_control.json"
    judge_variant = tmp_settings.run_artifacts_dir.parent / "variant_runs" / "retrieval_hybrid" / "p1" / "judge_variant.json"
    assert judge_control.exists()
    assert judge_variant.exists()


def test_run_variant_evaluation_uses_cached_results_when_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
) -> None:
    prompt = FounderPrompt(id="p1", statement="Question", domain="restaurants")
    tmp_settings.founder_prompts_path.write_text(json.dumps([prompt.model_dump()]), encoding="utf-8")

    root = tmp_settings.run_artifacts_dir.parent / "variant_runs"
    control_dir = root / "control" / "p1"
    variant_dir = root / "critic_loop" / "p1"
    control_dir.mkdir(parents=True, exist_ok=True)
    variant_dir.mkdir(parents=True, exist_ok=True)

    (control_dir / "synthesis.json").write_text(_synthesis(prompt, "control").model_dump_json(indent=2), encoding="utf-8")
    (variant_dir / "synthesis.json").write_text(_synthesis(prompt, "critic_loop").model_dump_json(indent=2), encoding="utf-8")
    (variant_dir / "judge_control.json").write_text(_judge("p1", "control", 4.0).model_dump_json(indent=2), encoding="utf-8")
    (variant_dir / "judge_variant.json").write_text(_judge("p1", "critic_loop", 4.5).model_dump_json(indent=2), encoding="utf-8")

    monkeypatch.setattr(variant_runner, "run_variant_pipeline", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not run variant")))
    monkeypatch.setattr(variant_runner, "judge_named_pair", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not judge")))

    summary = variant_runner.run_variant_evaluation(
        tmp_settings,
        variant_ids=["critic_loop"],
        prompt_ids=["p1"],
        skip_cached=True,
    )

    assert len(summary.aggregates) == 1
    assert summary.aggregates[0].variant == "critic_loop"
    assert summary.aggregates[0].delta_overall == pytest.approx(0.5)


def test_generate_variant_report_writes_markdown(tmp_path: Path) -> None:
    summary = variant_runner.VariantEvaluationSummary(
        control_variant="control",
        prompt_ids=["p1"],
        aggregates=[
            variant_runner.VariantAggregate(
                variant="full_hybrid",
                description="test",
                control_scores={dim: 3.0 for dim in variant_runner.RUBRIC_DIMS},
                variant_scores={dim: 4.0 for dim in variant_runner.RUBRIC_DIMS},
                control_overall=3.0,
                variant_overall=4.0,
                delta_overall=1.0,
            )
        ],
        comparisons_by_variant={},
    )

    path = variant_report.generate_variant_report(summary, tmp_path)
    text = path.read_text(encoding="utf-8")

    assert "# Variant Evaluation Report" in text
    assert "Aggregate Ranking" in text
    assert "Variant: `full_hybrid`" in text

