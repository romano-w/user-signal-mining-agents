from __future__ import annotations

import json
from pathlib import Path

import pytest

from user_signal_mining_agents.evaluation import prompt_sweep, report, runner
from user_signal_mining_agents.schemas import (
    EvaluationSummary,
    FocusPoint,
    FounderPrompt,
    JudgeResult,
    JudgeScores,
    PromptEvaluationPair,
    SynthesisResult,
)


def _scores(value: float, rationale: str = "ok") -> JudgeScores:
    return JudgeScores(
        relevance=value,
        actionability=value,
        evidence_grounding=value,
        contradiction_handling=value,
        non_redundancy=value,
        rationale=rationale,
    )


def _judge(prompt_id: str, variant: str, value: float) -> JudgeResult:
    return JudgeResult(prompt_id=prompt_id, system_variant=variant, scores=_scores(value))


def _synthesis(prompt: FounderPrompt, variant: str) -> SynthesisResult:
    return SynthesisResult(
        system_variant=variant,
        prompt=prompt,
        retrieved_evidence=[],
        focus_points=[
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
        ],
    )


def test_generate_report_writes_markdown(tmp_path: Path) -> None:
    prompt = FounderPrompt(id="p1", statement="Question", domain="restaurants")
    summary = EvaluationSummary(
        pairs=[
            PromptEvaluationPair(
                prompt=prompt,
                baseline_scores=_judge("p1", "baseline", 3.0),
                pipeline_scores=_judge("p1", "pipeline", 4.0),
            )
        ]
    )

    path = report.generate_report(summary, tmp_path)
    text = path.read_text(encoding="utf-8")

    assert "# Evaluation Report" in text
    assert "Aggregate Scores" in text
    assert "Per-Prompt Breakdown" in text


def test_run_evaluation_uses_cached_judge_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
) -> None:
    prompt = FounderPrompt(id="p1", statement="Question", domain="restaurants")
    tmp_settings.founder_prompts_path.write_text(json.dumps([prompt.model_dump()]), encoding="utf-8")

    run_dir = tmp_settings.run_artifacts_dir / prompt.id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "judge_baseline.json").write_text(_judge("p1", "baseline", 3.0).model_dump_json(), encoding="utf-8")
    (run_dir / "judge_pipeline.json").write_text(_judge("p1", "pipeline", 4.0).model_dump_json(), encoding="utf-8")

    monkeypatch.setattr(runner, "run_baseline", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not run baseline")))
    monkeypatch.setattr(runner, "run_pipeline", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not run pipeline")))

    summary = runner.run_evaluation(tmp_settings, skip_cached=True)

    assert len(summary.pairs) == 1
    assert summary.pairs[0].pipeline_scores.scores.relevance == 4.0


def test_run_evaluation_filters_prompt_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
) -> None:
    prompts = [
        FounderPrompt(id="p1", statement="Question1", domain="restaurants"),
        FounderPrompt(id="p2", statement="Question2", domain="restaurants"),
    ]
    tmp_settings.founder_prompts_path.write_text(json.dumps([p.model_dump() for p in prompts]), encoding="utf-8")

    monkeypatch.setattr(runner, "run_baseline", lambda prompt, _settings: _synthesis(prompt, "baseline"))
    monkeypatch.setattr(runner, "run_pipeline", lambda prompt, _settings: _synthesis(prompt, "pipeline"))
    monkeypatch.setattr(runner, "judge_pair", lambda prompt, *_args: (_judge(prompt.id, "baseline", 3.0), _judge(prompt.id, "pipeline", 4.0)))

    summary = runner.run_evaluation(tmp_settings, prompt_ids=["p2"], skip_cached=False)

    assert len(summary.pairs) == 1
    assert summary.pairs[0].prompt.id == "p2"


def test_run_evaluation_runs_uncached_and_persists_judge_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
) -> None:
    prompt = FounderPrompt(id="p1", statement="Question", domain="restaurants")
    tmp_settings.founder_prompts_path.write_text(json.dumps([prompt.model_dump()]), encoding="utf-8")

    monkeypatch.setattr(runner, "run_baseline", lambda p, _s: _synthesis(p, "baseline"))
    monkeypatch.setattr(runner, "run_pipeline", lambda p, _s: _synthesis(p, "pipeline"))
    monkeypatch.setattr(runner, "judge_pair", lambda p, *_args: (_judge(p.id, "baseline", 3.0), _judge(p.id, "pipeline", 4.0)))

    summary = runner.run_evaluation(tmp_settings, skip_cached=False)

    assert len(summary.pairs) == 1
    run_dir = tmp_settings.run_artifacts_dir / prompt.id
    assert (run_dir / "judge_baseline.json").exists()
    assert (run_dir / "judge_pipeline.json").exists()


def test_run_sweep_applies_variant_and_restores_prompts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
) -> None:
    original = (tmp_settings.prompts_dir / "synthesis.md").read_text(encoding="utf-8")

    variant = prompt_sweep.SweepVariant(
        name="test-variant",
        description="override synthesis",
        overrides={"synthesis.md": "patched synthesis"},
    )

    summary = EvaluationSummary(
        pairs=[
            PromptEvaluationPair(
                prompt=FounderPrompt(id="p1", statement="Q", domain="restaurants"),
                baseline_scores=_judge("p1", "baseline", 3.0),
                pipeline_scores=_judge("p1", "pipeline", 4.0),
            )
        ]
    )
    monkeypatch.setattr(prompt_sweep, "run_evaluation", lambda *_args, **_kwargs: summary)

    results = prompt_sweep.run_sweep(tmp_settings, variants=[variant], prompt_ids=["p1"])

    assert len(results) == 1
    assert results[0].variant == "test-variant"
    assert (tmp_settings.prompts_dir / "synthesis.md").read_text(encoding="utf-8") == original


def test_run_sweep_restores_prompts_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
) -> None:
    original = (tmp_settings.prompts_dir / "synthesis.md").read_text(encoding="utf-8")

    variant = prompt_sweep.SweepVariant(
        name="failing",
        description="failing variant",
        overrides={"synthesis.md": "patched synthesis"},
    )

    monkeypatch.setattr(prompt_sweep, "run_evaluation", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        prompt_sweep.run_sweep(tmp_settings, variants=[variant])

    assert (tmp_settings.prompts_dir / "synthesis.md").read_text(encoding="utf-8") == original
    assert not (tmp_settings.prompts_dir.parent / "prompts_backup").exists()
