from __future__ import annotations

from pathlib import Path

import pytest

from user_signal_mining_agents.evaluation.human_annotation_gui import AnnotationWorkspace, _build_index_html
from user_signal_mining_agents.schemas import HumanAnnotationTask


def _write_task(
    tasks_dir: Path,
    *,
    task_id: str,
    founder_prompt,
    evidence_factory,
    focus_point_factory,
) -> None:
    task = HumanAnnotationTask(
        task_id=task_id,
        prompt=founder_prompt,
        retrieved_evidence=[evidence_factory(1)],
        system_a_focus_points=[
            focus_point_factory(1),
            focus_point_factory(2),
            focus_point_factory(3),
        ],
        system_b_focus_points=[
            focus_point_factory(4),
            focus_point_factory(5),
            focus_point_factory(6),
        ],
        ground_truth_mapping={
            "system_a": "baseline",
            "system_b": "pipeline",
        },
    )
    (tasks_dir / f"{task_id}.json").write_text(task.model_dump_json(indent=2), encoding="utf-8")


def _payload(task_id: str, annotator_id: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "annotator_id": annotator_id,
        "system_a_scores": {
            "relevance": 4,
            "groundedness": 5,
            "distinctiveness": 4,
            "rationale": "A was generally tighter.",
        },
        "system_b_scores": {
            "relevance": 3,
            "groundedness": 4,
            "distinctiveness": 3,
            "rationale": "B repeated one theme.",
        },
        "overall_preference": "system_a",
        "difficulty_rating": 2,
    }


def test_annotation_workspace_lists_tasks_and_hides_mapping(
    tmp_path: Path,
    founder_prompt,
    evidence_factory,
    focus_point_factory,
) -> None:
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    _write_task(
        tasks_dir,
        task_id="task_prompt-1",
        founder_prompt=founder_prompt,
        evidence_factory=evidence_factory,
        focus_point_factory=focus_point_factory,
    )

    workspace = AnnotationWorkspace(tasks_dir)
    tasks = workspace.list_tasks()
    assert len(tasks) == 1
    assert tasks[0]["task_id"] == "task_prompt-1"

    public_task = workspace.get_public_task("task_prompt-1")
    assert "ground_truth_mapping" not in public_task


def test_annotation_workspace_saves_and_exports_results(
    tmp_path: Path,
    founder_prompt,
    evidence_factory,
    focus_point_factory,
) -> None:
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    _write_task(
        tasks_dir,
        task_id="task_prompt-1",
        founder_prompt=founder_prompt,
        evidence_factory=evidence_factory,
        focus_point_factory=focus_point_factory,
    )

    workspace = AnnotationWorkspace(tasks_dir)
    payload = _payload("task_prompt-1", "reviewer_1")
    workspace.save_result(task_id="task_prompt-1", annotator_id="reviewer_1", payload=payload)

    saved = workspace.load_result("task_prompt-1", "reviewer_1")
    assert saved is not None
    assert saved["overall_preference"] == "system_a"

    exported = workspace.export_results("reviewer_1")
    assert len(exported) == 1
    assert exported[0]["task_id"] == "task_prompt-1"

    out_file = tasks_dir / "_results" / "reviewer_1" / "task_prompt-1.json"
    assert out_file.exists()


def test_annotation_workspace_rejects_invalid_annotator_id() -> None:
    with pytest.raises(ValueError, match="annotator_id"):
        AnnotationWorkspace.normalize_annotator_id("x")

    with pytest.raises(ValueError, match="annotator_id"):
        AnnotationWorkspace.normalize_annotator_id("invalid id")


def test_annotation_html_explains_the_prestart_gate() -> None:
    html = _build_index_html("")

    assert "Tasks stay hidden until annotation begins" in html
    assert "Session not started" in html
    assert "Begin Annotation" in html

