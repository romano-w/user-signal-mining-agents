from __future__ import annotations

from pathlib import Path

import pytest

from user_signal_mining_agents.evaluation import gates
from user_signal_mining_agents.schemas import JudgeResult, JudgeScores


def _scores(value: float) -> JudgeScores:
    return JudgeScores(
        relevance=value,
        actionability=value,
        evidence_grounding=value,
        contradiction_handling=value,
        non_redundancy=value,
        rationale=f"score={value}",
    )


def _judge(prompt_id: str, variant: str, value: float) -> JudgeResult:
    return JudgeResult(
        prompt_id=prompt_id,
        system_variant=variant,
        scores=_scores(value),
    )


def _write_pair(root: Path, prompt_id: str, baseline: float, pipeline: float) -> None:
    run_dir = root / prompt_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "judge_baseline.json").write_text(
        _judge(prompt_id, "baseline", baseline).model_dump_json(indent=2),
        encoding="utf-8",
    )
    (run_dir / "judge_pipeline.json").write_text(
        _judge(prompt_id, "pipeline", pipeline).model_dump_json(indent=2),
        encoding="utf-8",
    )


def test_load_judge_pairs_reads_complete_run_directories(tmp_path: Path) -> None:
    _write_pair(tmp_path, "p1", baseline=3.0, pipeline=4.0)
    _write_pair(tmp_path, "p2", baseline=4.0, pipeline=4.2)
    (tmp_path / "incomplete").mkdir(parents=True, exist_ok=True)

    pairs = gates.load_judge_pairs(tmp_path)

    assert len(pairs) == 2
    assert {baseline.prompt_id for baseline, _ in pairs} == {"p1", "p2"}


def test_summarize_metric_deltas_includes_all_dimensions_and_overall() -> None:
    pairs = [
        (_judge("p1", "baseline", 3.0), _judge("p1", "pipeline", 4.0)),
        (_judge("p2", "baseline", 4.0), _judge("p2", "pipeline", 4.0)),
    ]

    summary = {item.metric: item for item in gates.summarize_metric_deltas(pairs)}

    assert set(summary.keys()) == {*gates.RUBRIC_DIMS, "overall"}
    assert summary["relevance"].baseline_avg == pytest.approx(3.5)
    assert summary["relevance"].pipeline_avg == pytest.approx(4.0)
    assert summary["overall"].delta == pytest.approx(0.5)


def test_find_critical_metric_regressions_returns_empty_when_within_thresholds() -> None:
    pairs = [(_judge("p1", "baseline", 4.2), _judge("p1", "pipeline", 3.9))]

    violations = gates.find_critical_metric_regressions(
        pairs,
        max_overall_drop=0.50,
        max_dimension_drop=0.50,
    )

    assert violations == []


def test_find_critical_metric_regressions_flags_dimension_and_overall_drops() -> None:
    pairs = [(_judge("p1", "baseline", 4.8), _judge("p1", "pipeline", 3.8))]

    violations = gates.find_critical_metric_regressions(
        pairs,
        max_overall_drop=0.30,
        max_dimension_drop=0.40,
    )

    metrics = {violation.metric for violation in violations}
    assert metrics == {*gates.RUBRIC_DIMS, "overall"}
