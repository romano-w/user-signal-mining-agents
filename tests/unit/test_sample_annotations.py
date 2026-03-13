from __future__ import annotations

import json
from pathlib import Path

from user_signal_mining_agents.evaluation import sample_annotations
from user_signal_mining_agents.schemas import HumanAnnotationTask, SynthesisResult


def _write_run_artifacts(
    root: Path,
    synthesis: SynthesisResult,
    *,
    system_variant: str,
    overall_preference: float,
) -> None:
    run_dir = root / synthesis.prompt.id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{system_variant}.json").write_text(synthesis.model_dump_json(indent=2), encoding="utf-8")
    judge_payload = {
        "prompt_id": synthesis.prompt.id,
        "system_variant": system_variant,
        "scores": {
            "relevance": 4.0,
            "contradiction": 4.0,
            "coverage": 4.0,
            "distinctiveness": 4.0,
            "overall_preference": overall_preference,
            "rationale": f"{system_variant} rationale",
        },
    }
    (run_dir / f"judge_{system_variant}.json").write_text(json.dumps(judge_payload, indent=2), encoding="utf-8")


def test_sample_for_annotation_supports_legacy_judge_artifacts(
    monkeypatch,
    tmp_settings,
    founder_prompt,
    evidence_factory,
    focus_point_factory,
) -> None:
    baseline = SynthesisResult(
        system_variant="baseline",
        prompt=founder_prompt,
        retrieved_evidence=[evidence_factory(1), evidence_factory(2)],
        focus_points=[focus_point_factory(1), focus_point_factory(2), focus_point_factory(3)],
    )
    pipeline = baseline.model_copy(update={"system_variant": "pipeline"})

    _write_run_artifacts(
        tmp_settings.run_artifacts_dir,
        baseline,
        system_variant="baseline",
        overall_preference=3.0,
    )
    _write_run_artifacts(
        tmp_settings.run_artifacts_dir,
        pipeline,
        system_variant="pipeline",
        overall_preference=4.0,
    )

    monkeypatch.setattr(sample_annotations, "get_settings", lambda: tmp_settings)

    output_dir_a = tmp_settings.run_artifacts_dir / "_human_annotations_a"
    output_dir_b = tmp_settings.run_artifacts_dir / "_human_annotations_b"
    sample_annotations.sample_for_annotation(num_samples=20, output_dir=output_dir_a, seed=11)
    sample_annotations.sample_for_annotation(num_samples=20, output_dir=output_dir_b, seed=11)

    task_path_a = output_dir_a / f"task_{founder_prompt.id}.json"
    task_path_b = output_dir_b / f"task_{founder_prompt.id}.json"
    assert task_path_a.exists()
    assert task_path_b.exists()

    task_a = HumanAnnotationTask.model_validate_json(task_path_a.read_text(encoding="utf-8"))
    task_b = HumanAnnotationTask.model_validate_json(task_path_b.read_text(encoding="utf-8"))

    assert task_a.model_dump(mode="json") == task_b.model_dump(mode="json")
    assert task_a.prompt.id == founder_prompt.id
    assert set(task_a.ground_truth_mapping.values()) == {"baseline", "pipeline"}
    assert len(task_a.system_a_focus_points) == 3
    assert len(task_a.system_b_focus_points) == 3

