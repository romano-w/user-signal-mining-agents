from __future__ import annotations

import json
from pathlib import Path

import pytest

from user_signal_mining_agents.evaluation import robustness_report, robustness_runner
from user_signal_mining_agents.schemas import (
    FocusPoint,
    FounderPrompt,
    JudgeResult,
    JudgeScores,
    RobustnessCase,
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
        groundedness=value,
        distinctiveness=value,
        rationale=f"score {value}",
    )


def _judge(prompt_id: str, variant: str, value: float) -> JudgeResult:
    return JudgeResult(
        prompt_id=prompt_id,
        system_variant=variant,
        scores=_scores(value),
    )


def test_apply_perturbation_is_deterministic_across_families() -> None:
    statement = "Why are diners not returning to this restaurant after takeout orders?"

    negation_case = RobustnessCase(
        case_id="neg",
        family="negation",
        description="negate",
        transform_spec={"strategy": "negation_flip"},
        expected_behavior="stable",
    )
    noise_case = RobustnessCase(
        case_id="noise",
        family="noise",
        description="noise",
        transform_spec={"seed": 7, "injection_rate": 0.3, "token": "[n]"},
        expected_behavior="stable",
    )
    context_case = RobustnessCase(
        case_id="ctx",
        family="context_shift",
        description="shift",
        transform_spec={
            "context_prefix": "Assume this is a SaaS product question.",
            "replacements": [["restaurant", "product"], ["diners", "users"]],
        },
        expected_behavior="stable",
    )

    for case in (negation_case, noise_case, context_case):
        first = robustness_runner.apply_perturbation(statement, case)
        second = robustness_runner.apply_perturbation(statement, case)
        assert first == second
        assert first != statement

    alternate_noise = noise_case.model_copy(update={"transform_spec": {**noise_case.transform_spec, "seed": 99}})
    assert robustness_runner.apply_perturbation(statement, noise_case) != robustness_runner.apply_perturbation(
        statement,
        alternate_noise,
    )


def test_evaluate_case_thresholds_flags_drops() -> None:
    thresholds = robustness_runner.RobustnessGateThresholds(
        max_overall_drop=0.5,
        max_dimension_drop=0.6,
        min_case_pass_rate=1.0,
    )

    passing, _, delta_overall, failures = robustness_runner.evaluate_case_thresholds(
        _scores(4.0),
        _scores(3.7),
        thresholds,
    )
    assert passing is True
    assert delta_overall == pytest.approx(-0.3)
    assert failures == []

    failing_scores = JudgeScores(
        relevance=2.8,
        overall_preference=2.9,
        groundedness=2.8,
        distinctiveness=2.8,
        rationale="drop",
    )
    passing, _, delta_overall, failures = robustness_runner.evaluate_case_thresholds(
        _scores(4.0),
        failing_scores,
        thresholds,
    )
    assert passing is False
    assert delta_overall < -0.5
    assert any("overall drop" in failure for failure in failures)


def test_run_robustness_suite_surfaces_gate_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
) -> None:
    prompt = FounderPrompt(id="p1", statement="Question", domain="restaurants")
    tmp_settings.founder_prompts_path.write_text(json.dumps([prompt.model_dump()]), encoding="utf-8")

    monkeypatch.setattr(
        robustness_runner,
        "run_variant_pipeline",
        lambda prompt_obj, *_args, **_kwargs: _synthesis(prompt_obj, "control"),
    )
    monkeypatch.setattr(
        robustness_runner,
        "judge_named_pair",
        lambda prompt_obj, *_args, **_kwargs: (
            _judge(prompt_obj.id, "control", 4.0),
            _judge(prompt_obj.id, "perturbed", 3.0),
        ),
    )

    summary = robustness_runner.run_robustness_suite(
        tmp_settings,
        suite_id="adversarial_core",
        prompt_ids=["p1"],
        skip_cached=False,
    )

    assert summary.suite_id == "adversarial_core"
    assert summary.total_cases == len(robustness_runner.get_suite_spec("adversarial_core").cases)
    assert summary.passed_cases == 0
    assert summary.failed_cases == summary.total_cases
    assert summary.gate_passed is False
    assert summary.gate_failure_reasons

    judge_path = (
        robustness_runner.suite_output_dir(tmp_settings, "adversarial_core")
        / "judges"
        / "rb_negation_flip"
        / "p1"
        / "judge_perturbed.json"
    )
    assert judge_path.exists()


def test_generate_robustness_report_writes_markdown(tmp_path: Path) -> None:
    summary = robustness_runner.RobustnessSuiteSummary(
        suite_id="default",
        suite_description="test",
        prompt_ids=["p1"],
        thresholds=robustness_runner.RobustnessGateThresholds(
            max_overall_drop=0.5,
            max_dimension_drop=1.0,
            min_case_pass_rate=1.0,
        ),
        total_cases=1,
        passed_cases=1,
        failed_cases=0,
        pass_rate=1.0,
        gate_passed=True,
        failed_case_keys=[],
        gate_failure_reasons=[],
        outcomes=[
            robustness_runner.RobustnessCaseOutcome(
                prompt_id="p1",
                case_id="rb_noise_injection",
                family="noise",
                description="noise",
                expected_behavior="stable",
                perturbed_prompt_id="p1__rb_noise_injection",
                perturbed_statement="[noise] Question",
                control_scores=_scores(4.0),
                perturbed_scores=_scores(3.9),
                dimension_deltas={dim: -0.1 for dim in robustness_runner.RUBRIC_DIMS},
                delta_overall=-0.1,
                passed=True,
                failure_reasons=[],
            )
        ],
    )

    path = robustness_report.generate_robustness_report(summary, tmp_path)
    text = path.read_text(encoding="utf-8")

    assert "# Robustness Evaluation Report" in text
    assert "Case Outcomes" in text
    assert "rb_noise_injection" in text

