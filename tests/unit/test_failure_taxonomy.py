from __future__ import annotations

from pathlib import Path

from user_signal_mining_agents.evaluation.failure_taxonomy import (
    classify_judge_result,
    generate_failure_taxonomy,
    generate_root_cause_report,
)
from user_signal_mining_agents.schemas import FailureTag, JudgeResult, JudgeScores


def _judge(
    prompt_id: str,
    variant: str,
    *,
    relevance: float,
    overall_preference: float,
    coverage: float,
    contradiction: float,
    distinctiveness: float,
) -> JudgeResult:
    return JudgeResult(
        prompt_id=prompt_id,
        system_variant=variant,
        scores=JudgeScores(
            relevance=relevance,
            overall_preference=overall_preference,
            coverage=coverage,
            contradiction=contradiction,
            distinctiveness=distinctiveness,
            rationale=f"{variant} rationale",
        ),
    )


def test_classify_judge_result_flags_low_dimensions_deterministically() -> None:
    judge = _judge(
        "p1",
        "pipeline",
        relevance=2.5,
        overall_preference=3.4,
        coverage=3.2,
        contradiction=2.0,
        distinctiveness=4.0,
    )

    tags = classify_judge_result("p1", "pipeline", judge)

    assert [tag.tag_id for tag in tags] == [
        "ft_p1_pipeline_relevance_miss",
        "ft_p1_pipeline_contradiction_blindness",
        "ft_p1_pipeline_coverage_gap",
        "ft_p1_pipeline_overall_preference_gap",
        "ft_p1_pipeline_overall_quality_drop",
    ]
    assert [tag.severity for tag in tags] == [4, 5, 3, 3, 3]
    assert tags[0].evidence_refs[0] == "p1/judge_pipeline.json#scores.relevance"


def test_generate_failure_taxonomy_writes_artifacts_from_run_files(tmp_path: Path) -> None:
    p1_dir = tmp_path / "p1"
    p2_dir = tmp_path / "p2"
    p1_dir.mkdir(parents=True, exist_ok=True)
    p2_dir.mkdir(parents=True, exist_ok=True)


    (p1_dir / "judge_baseline.json").write_text(
        _judge(
            "p1",
            "baseline",
            relevance=4.8,
            overall_preference=4.7,
            coverage=4.9,
            contradiction=4.6,
            distinctiveness=4.8,
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    (p1_dir / "judge_pipeline.json").write_text(
        _judge(
            "p1",
            "pipeline",
            relevance=2.5,
            overall_preference=3.4,
            coverage=3.2,
            contradiction=2.0,
            distinctiveness=4.0,
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    (p2_dir / "judge_baseline.json").write_text(
        _judge(
            "p2",
            "baseline",
            relevance=4.5,
            overall_preference=4.6,
            coverage=4.4,
            contradiction=4.7,
            distinctiveness=4.6,
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    (p2_dir / "judge_pipeline.json").write_text(
        _judge(
            "p2",
            "pipeline",
            relevance=4.7,
            overall_preference=4.5,
            coverage=4.6,
            contradiction=4.8,
            distinctiveness=4.7,
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    tags, tags_path, report_path = generate_failure_taxonomy(tmp_path, prompt_ids=["p1"])

    assert [tag.tag_id for tag in tags] == [
        "ft_p1_pipeline_relevance_miss",
        "ft_p1_pipeline_contradiction_blindness",
        "ft_p1_pipeline_coverage_gap",
        "ft_p1_pipeline_overall_preference_gap",
        "ft_p1_pipeline_overall_quality_drop",
    ]
    assert tags_path.exists()
    assert report_path.exists()

    jsonl = tags_path.read_text(encoding="utf-8").strip().splitlines()
    parsed = [FailureTag.model_validate_json(line) for line in jsonl]
    assert [tag.tag_id for tag in parsed] == [tag.tag_id for tag in tags]

    report_text = report_path.read_text(encoding="utf-8")
    assert "Category Overview" in report_text
    assert "p1/judge_pipeline.json" in report_text
    assert "p2" not in report_text

    rerun_tags, _, _ = generate_failure_taxonomy(tmp_path, prompt_ids=["p1"])
    assert [tag.tag_id for tag in rerun_tags] == [tag.tag_id for tag in tags]


def test_generate_root_cause_report_handles_no_tags(tmp_path: Path) -> None:
    report_path = generate_root_cause_report([], tmp_path)
    text = report_path.read_text(encoding="utf-8")

    assert "Failure tags generated" in text
    assert "No low-quality outputs were tagged" in text


