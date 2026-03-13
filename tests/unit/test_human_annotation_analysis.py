from __future__ import annotations

import json
from pathlib import Path

from user_signal_mining_agents.evaluation.human_annotation_analysis import analyze_and_write_human_annotation_report
from user_signal_mining_agents.schemas import (
    EvidenceSnippet,
    FocusPoint,
    FounderPrompt,
    HumanAnnotationResult,
    HumanAnnotationScores,
    HumanAnnotationTask,
)


def _focus_point(index: int) -> FocusPoint:
    return FocusPoint(
        label=f"focus-{index}",
        why_it_matters=f"Why {index}",
        supporting_snippets=[f"snippet-{index}"],
        counter_signal=f"Counter {index}",
        next_validation_question=f"Question {index}?",
    )


def _task(task_id: str, *, prompt_id: str, mapping: dict[str, str]) -> HumanAnnotationTask:
    return HumanAnnotationTask(
        task_id=task_id,
        prompt=FounderPrompt(id=prompt_id, statement=f"Prompt for {prompt_id}", domain="ecommerce"),
        retrieved_evidence=[
            EvidenceSnippet(
                snippet_id="snippet-1",
                review_id="review-1",
                business_id="biz-1",
                text="Customers mentioned checkout friction and shipping surprises.",
            )
        ],
        system_a_focus_points=[_focus_point(1), _focus_point(2), _focus_point(3)],
        system_b_focus_points=[_focus_point(4), _focus_point(5), _focus_point(6)],
        ground_truth_mapping=mapping,
    )


def _result(
    task_id: str,
    annotator_id: str,
    *,
    overall_preference: str,
    system_a_relevance: int,
    system_b_relevance: int,
) -> HumanAnnotationResult:
    return HumanAnnotationResult(
        task_id=task_id,
        annotator_id=annotator_id,
        system_a_scores=HumanAnnotationScores(
            relevance=system_a_relevance,
            groundedness=5,
            distinctiveness=4,
            rationale="System A rationale",
        ),
        system_b_scores=HumanAnnotationScores(
            relevance=system_b_relevance,
            groundedness=4,
            distinctiveness=3,
            rationale="System B rationale",
        ),
        overall_preference=overall_preference,
        difficulty_rating=2,
    )


def _write_task(path: Path, task: HumanAnnotationTask) -> None:
    path.write_text(task.model_dump_json(indent=2), encoding="utf-8")


def _write_export(path: Path, annotator_id: str, results: list[HumanAnnotationResult]) -> None:
    payload = {
        "annotator_id": annotator_id,
        "count": len(results),
        "results": [result.model_dump(mode="json") for result in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_judge_scores(runs_dir: Path, prompt_id: str, *, baseline_overall: float, pipeline_overall: float) -> None:
    prompt_dir = runs_dir / prompt_id
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "judge_baseline.json").write_text(
        json.dumps({"scores": {"overall_preference": baseline_overall}}, indent=2),
        encoding="utf-8",
    )
    (prompt_dir / "judge_pipeline.json").write_text(
        json.dumps({"scores": {"overall_preference": pipeline_overall}}, indent=2),
        encoding="utf-8",
    )



def test_analyze_and_write_human_annotation_report_computes_agreement(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    runs_dir = tmp_path / "runs"
    output_dir = tmp_path / "reports"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    _write_task(
        tasks_dir / "task_prompt-1.json",
        _task(
            "task_prompt-1",
            prompt_id="prompt-1",
            mapping={"system_a": "baseline", "system_b": "pipeline"},
        ),
    )
    _write_task(
        tasks_dir / "task_prompt-2.json",
        _task(
            "task_prompt-2",
            prompt_id="prompt-2",
            mapping={"system_a": "pipeline", "system_b": "baseline"},
        ),
    )

    _write_judge_scores(runs_dir, "prompt-1", baseline_overall=3.0, pipeline_overall=4.5)
    _write_judge_scores(runs_dir, "prompt-2", baseline_overall=4.2, pipeline_overall=3.2)

    export_a = tmp_path / "reviewer_1.json"
    export_b = tmp_path / "reviewer_2.json"
    _write_export(
        export_a,
        "reviewer_1",
        [
            _result("task_prompt-1", "reviewer_1", overall_preference="system_b", system_a_relevance=3, system_b_relevance=5),
            _result("task_prompt-2", "reviewer_1", overall_preference="system_a", system_a_relevance=4, system_b_relevance=2),
        ],
    )
    _write_export(
        export_b,
        "reviewer_2",
        [
            _result("task_prompt-1", "reviewer_2", overall_preference="system_b", system_a_relevance=3, system_b_relevance=4),
            _result("task_prompt-2", "reviewer_2", overall_preference="tie", system_a_relevance=5, system_b_relevance=2),
        ],
    )

    summary, json_path, markdown_path = analyze_and_write_human_annotation_report(
        export_a,
        export_b_path=export_b,
        tasks_dir=tasks_dir,
        runs_dir=runs_dir,
        output_dir=output_dir,
    )

    assert summary.overlapping_task_ids == ["task_prompt-1", "task_prompt-2"]
    assert summary.interannotator_overall_preference is not None
    assert summary.interannotator_overall_preference.sample_size == 2
    assert summary.interannotator_overall_preference.exact_agreement == 0.5
    assert [metric.metric for metric in summary.interannotator_dimensions] == [
        "relevance",
        "groundedness",
        "distinctiveness",
    ]
    assert all(metric.sample_size == 4 for metric in summary.interannotator_dimensions)

    assert len(summary.judge_alignment) == 2
    reviewer_1 = next(row for row in summary.judge_alignment if row.annotator_id == "reviewer_1")
    reviewer_2 = next(row for row in summary.judge_alignment if row.annotator_id == "reviewer_2")
    assert reviewer_1.sample_size == 2
    assert reviewer_2.sample_size == 2
    assert reviewer_1.exact_agreement == 0.5
    assert reviewer_2.exact_agreement == 0.5
    assert summary.missing_task_ids == []
    assert summary.missing_judge_prompt_ids == []

    assert json_path.exists()
    assert markdown_path.exists()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Human Annotation Analysis" in markdown
    assert "## Interannotator Agreement" in markdown
    assert "reviewer_1" in markdown

