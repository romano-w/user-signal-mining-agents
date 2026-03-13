from __future__ import annotations

import json
from pathlib import Path

import pytest

from user_signal_mining_agents.evaluation import prompt_sweep, report, runner
from user_signal_mining_agents.schemas import (
    EvaluationSummary,
    FocusPoint,
    FounderPrompt,
    JudgePanelResult,
    JudgeResult,
    JudgeScores,
    MetricWithCI,
    PromptEvaluationPair,
    SignificanceResult,
    SynthesisResult,
)


def _scores(value: float, rationale: str = "ok") -> JudgeScores:
    return JudgeScores(
        relevance=value,
        overall_preference=value,
        groundedness=value,
        distinctiveness=value,
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
    restaurant_prompt = FounderPrompt(id="p1", statement="Restaurant question", domain="restaurants")
    saas_prompt = FounderPrompt(id="p2", statement="SaaS question", domain="saas")
    summary = EvaluationSummary(
        pairs=[
            PromptEvaluationPair(
                prompt=restaurant_prompt,
                baseline_scores=_judge("p1", "baseline", 3.0),
                pipeline_scores=_judge("p1", "pipeline", 4.0),
            ),
            PromptEvaluationPair(
                prompt=saas_prompt,
                baseline_scores=_judge("p2", "baseline", 3.5),
                pipeline_scores=_judge("p2", "pipeline", 4.2),
            ),
        ]
    )

    path = report.generate_report(summary, tmp_path)
    text = path.read_text(encoding="utf-8")

    assert "# Evaluation Report" in text
    assert "Aggregate Scores" in text
    assert "Domain Quality Breakdown" in text
    assert "Domain Transfer Deltas" in text
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
    tmp_settings.founder_prompts_path.write_text(json.dumps([prompt.model_dump() for prompt in prompts]), encoding="utf-8")

    monkeypatch.setattr(runner, "run_baseline", lambda prompt, _settings: _synthesis(prompt, "baseline"))
    monkeypatch.setattr(runner, "run_pipeline", lambda prompt, _settings: _synthesis(prompt, "pipeline"))
    monkeypatch.setattr(runner, "judge_pair", lambda prompt, *_args: (_judge(prompt.id, "baseline", 3.0), _judge(prompt.id, "pipeline", 4.0)))

    summary = runner.run_evaluation(tmp_settings, prompt_ids=["p2"], skip_cached=False)

    assert len(summary.pairs) == 1
    assert summary.pairs[0].prompt.id == "p2"


def test_run_evaluation_filters_domain_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
    tmp_path: Path,
) -> None:
    restaurant_prompts = [
        FounderPrompt(id="p1", statement="Restaurant question", domain="restaurants"),
    ]
    saas_prompts = [
        FounderPrompt(id="s1", statement="SaaS question", domain="saas"),
    ]

    restaurants_path = tmp_path / "restaurants.json"
    saas_path = tmp_path / "saas.json"
    restaurants_path.write_text(json.dumps([prompt.model_dump() for prompt in restaurant_prompts]), encoding="utf-8")
    saas_path.write_text(json.dumps([prompt.model_dump() for prompt in saas_prompts]), encoding="utf-8")

    domain_packs_path = tmp_path / "domain_packs.json"
    domain_packs_path.write_text(
        json.dumps(
            [
                {
                    "domain_id": "restaurants",
                    "title": "Restaurants",
                    "founder_prompts_path": str(restaurants_path),
                    "enabled": True,
                },
                {
                    "domain_id": "saas",
                    "title": "SaaS",
                    "founder_prompts_path": str(saas_path),
                    "enabled": True,
                },
            ]
        ),
        encoding="utf-8",
    )
    settings = tmp_settings.model_copy(update={"domain_packs_path": domain_packs_path})

    monkeypatch.setattr(runner, "run_baseline", lambda prompt, _settings: _synthesis(prompt, "baseline"))
    monkeypatch.setattr(runner, "run_pipeline", lambda prompt, _settings: _synthesis(prompt, "pipeline"))
    monkeypatch.setattr(runner, "judge_pair", lambda prompt, *_args: (_judge(prompt.id, "baseline", 3.0), _judge(prompt.id, "pipeline", 4.0)))

    summary = runner.run_evaluation(settings, domain_ids=["saas"], skip_cached=False)

    assert len(summary.pairs) == 1
    assert summary.pairs[0].prompt.id == "s1"
    assert summary.pairs[0].prompt.domain == "saas"


def test_run_evaluation_runs_uncached_and_persists_judge_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
) -> None:
    prompt = FounderPrompt(id="p1", statement="Question", domain="restaurants")
    tmp_settings.founder_prompts_path.write_text(json.dumps([prompt.model_dump()]), encoding="utf-8")

    monkeypatch.setattr(runner, "run_baseline", lambda prompt, _settings: _synthesis(prompt, "baseline"))
    monkeypatch.setattr(runner, "run_pipeline", lambda prompt, _settings: _synthesis(prompt, "pipeline"))
    monkeypatch.setattr(runner, "judge_pair", lambda prompt, *_args: (_judge(prompt.id, "baseline", 3.0), _judge(prompt.id, "pipeline", 4.0)))

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


def _panel(prompt_id: str, variant: str, values: list[float], *, p_value: float) -> JudgePanelResult:
    per_judge = [_scores(value, rationale=f"judge-{idx}") for idx, value in enumerate(values, start=1)]
    mean_value = sum(values) / len(values)
    return JudgePanelResult(
        prompt_id=prompt_id,
        system_variant=variant,
        panel_size=len(values),
        per_judge_scores=per_judge,
        aggregate_scores=_scores(mean_value, rationale=f"panel-{variant}"),
        metrics_with_ci=[
            MetricWithCI(
                metric="overall_preference",
                mean=mean_value,
                ci95_lower=mean_value - 0.1,
                ci95_upper=mean_value + 0.1,
                sample_size=len(values),
            )
        ],
        significance=[
            SignificanceResult(
                metric="overall_preference",
                p_value=p_value,
                is_significant=p_value < 0.05,
                effect_size=0.3,
            )
        ],
    )


def test_run_evaluation_panel_mode_persists_panel_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
) -> None:
    prompt = FounderPrompt(id="p1", statement="Question", domain="restaurants")
    tmp_settings.founder_prompts_path.write_text(json.dumps([prompt.model_dump()]), encoding="utf-8")
    panel_settings = tmp_settings.model_copy(update={"judge_panel_size": 3})

    monkeypatch.setattr(runner, "run_baseline", lambda p, _s: _synthesis(p, "baseline"))
    monkeypatch.setattr(runner, "run_pipeline", lambda p, _s: _synthesis(p, "pipeline"))
    monkeypatch.setattr(
        runner,
        "judge_panel_pair",
        lambda p, *_args, **_kwargs: (
            _panel(p.id, "baseline", [3.0, 3.5, 2.5], p_value=0.2),
            _panel(p.id, "pipeline", [4.0, 4.2, 4.1], p_value=0.01),
        ),
    )

    summary = runner.run_evaluation(panel_settings, skip_cached=False)

    assert len(summary.pairs) == 1
    pair = summary.pairs[0]
    assert pair.baseline_panel is not None
    assert pair.pipeline_panel is not None
    assert pair.pipeline_scores.scores.relevance == pytest.approx(4.1)

    run_dir = panel_settings.run_artifacts_dir / prompt.id
    assert (run_dir / "judge_baseline.json").exists()
    assert (run_dir / "judge_pipeline.json").exists()
    assert (run_dir / "judge_panel_baseline.json").exists()
    assert (run_dir / "judge_panel_pipeline.json").exists()


def test_generate_report_includes_panel_confidence_context(tmp_path: Path) -> None:
    prompt = FounderPrompt(id="p1", statement="Question", domain="restaurants")
    baseline_panel = _panel("p1", "baseline", [3.0, 3.2, 2.8], p_value=0.4)
    pipeline_panel = _panel("p1", "pipeline", [4.1, 4.0, 4.2], p_value=0.01)

    summary = EvaluationSummary(
        pairs=[
            PromptEvaluationPair(
                prompt=prompt,
                baseline_scores=_judge("p1", "baseline", 3.0),
                pipeline_scores=_judge("p1", "pipeline", 4.0),
                baseline_panel=baseline_panel,
                pipeline_panel=pipeline_panel,
            )
        ]
    )

    path = report.generate_report(summary, tmp_path)
    text = path.read_text(encoding="utf-8")

    assert "Judge panel mode:** enabled" in text
    assert "Confidence context" in text
    assert "Panel Confidence (3 judges)" in text
    assert "p-value (Pipeline vs Baseline)" in text

