from __future__ import annotations

import json
from pathlib import Path

from user_signal_mining_agents.evaluation.final_analysis import build_analysis_report


def _write_prompt_run(
    runs_dir: Path,
    *,
    prompt_id: str,
    domain: str,
    statement: str,
    baseline_overall: float,
    pipeline_overall: float,
) -> None:
    run_dir = runs_dir / prompt_id
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_payload = {
        "prompt": {
            "id": prompt_id,
            "domain": domain,
            "statement": statement,
        }
    }
    (run_dir / "baseline.json").write_text(json.dumps(prompt_payload, indent=2), encoding="utf-8")
    (run_dir / "pipeline.json").write_text(json.dumps(prompt_payload, indent=2), encoding="utf-8")

    (run_dir / "judge_baseline.json").write_text(
        json.dumps(
            {
                "prompt_id": prompt_id,
                "system_variant": "baseline",
                "scores": {
                    "relevance": 4.0,
                    "groundedness": 3.5,
                    "distinctiveness": 3.0,
                    "overall_preference": baseline_overall,
                    "rationale": "baseline rationale",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "judge_pipeline.json").write_text(
        json.dumps(
            {
                "prompt_id": prompt_id,
                "system_variant": "pipeline",
                "scores": {
                    "relevance": 5.0,
                    "groundedness": 4.5,
                    "distinctiveness": 4.0,
                    "overall_preference": pipeline_overall,
                    "rationale": "pipeline rationale",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_failure_tags(runs_dir: Path) -> None:
    payloads = [
        {
            "tag_id": "ft_prompt-a_baseline_groundedness_gap",
            "category": "groundedness_gap",
            "severity": 4,
            "prompt_id": "prompt-a",
            "description": "baseline gap",
            "evidence_refs": [],
        },
        {
            "tag_id": "ft_prompt-b_pipeline_overall_preference_gap",
            "category": "overall_preference_gap",
            "severity": 5,
            "prompt_id": "prompt-b",
            "description": "pipeline gap",
            "evidence_refs": [],
        },
    ]
    (runs_dir / "failure_tags.jsonl").write_text(
        "\n".join(json.dumps(payload) for payload in payloads),
        encoding="utf-8",
    )


def _write_legacy_sweep(sweep_dir: Path) -> None:
    legacy_path = sweep_dir / "control" / "prompt-a"
    legacy_path.mkdir(parents=True, exist_ok=True)
    (legacy_path / "judge_pipeline.json").write_text(
        json.dumps(
            {
                "prompt_id": "prompt-a",
                "system_variant": "pipeline",
                "scores": {
                    "relevance": 5.0,
                    "actionability": 4.0,
                    "evidence_grounding": 4.0,
                    "non_redundancy": 3.0,
                    "rationale": "legacy schema",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_retrieval_summary(workspace_root: Path) -> None:
    retrieval_dir = workspace_root / "reports" / "research_upgrade"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    (retrieval_dir / "retrieval_eval_summary.json").write_text(
        json.dumps(
            {
                "query_count": 3,
                "retrieval_mode": "hybrid",
                "reranker": "none",
                "k_values": [1, 3, 5],
                "aggregates": {
                    "recall_at_k": {"1": 0.33, "3": 0.67, "5": 1.0},
                    "mrr_at_k": {"1": 0.33, "3": 0.50, "5": 0.50},
                    "ndcg_at_k": {"1": 0.33, "3": 0.61, "5": 0.79},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_annotation_progress(runs_dir: Path) -> None:
    tasks_dir = runs_dir / "_human_annotations"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / "task_prompt-a.json").write_text("{}", encoding="utf-8")
    (tasks_dir / "task_prompt-b.json").write_text("{}", encoding="utf-8")

    legacy_dir = tasks_dir / "_results" / "reviewer_legacy"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    (legacy_dir / "task_prompt-a.json").write_text(
        json.dumps(
            {
                "task_id": "task_prompt-a",
                "annotator_id": "reviewer_legacy",
                "system_a_scores": {"relevance": 3, "coverage": 4, "contradiction": 3, "distinctiveness": 4},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    current_dir = tasks_dir / "_results" / "reviewer_current"
    current_dir.mkdir(parents=True, exist_ok=True)
    (current_dir / "task_prompt-b.json").write_text(
        json.dumps(
            {
                "task_id": "task_prompt-b",
                "annotator_id": "reviewer_current",
                "system_a_scores": {"relevance": 4, "groundedness": 4, "distinctiveness": 5},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_build_analysis_report_generates_outputs_and_flags_legacy_artifacts(tmp_path: Path) -> None:
    runs_dir = tmp_path / "artifacts" / "runs"
    sweep_dir = tmp_path / "artifacts" / "sweep_runs"
    output_dir = tmp_path / "reports" / "final_analysis"

    _write_prompt_run(
        runs_dir,
        prompt_id="prompt-a",
        domain="restaurants",
        statement="Why do guests stop returning?",
        baseline_overall=3.0,
        pipeline_overall=5.0,
    )
    _write_prompt_run(
        runs_dir,
        prompt_id="prompt-b",
        domain="saas",
        statement="Why do users escalate support?",
        baseline_overall=4.0,
        pipeline_overall=3.0,
    )
    _write_failure_tags(runs_dir)
    _write_legacy_sweep(sweep_dir)
    _write_retrieval_summary(tmp_path)
    _write_annotation_progress(runs_dir)

    summary, summary_path, report_path = build_analysis_report(
        runs_dir=runs_dir,
        output_dir=output_dir,
        sweep_dir=sweep_dir,
        retrieval_summary_path=None,
        annotation_tasks_dir=runs_dir / "_human_annotations",
        annotation_results_dir=runs_dir / "_human_annotations" / "_results",
        annotation_exports_dir=tmp_path / "reports" / "human_annotation" / "exports",
    )

    assert summary.prompt_count == 2
    assert summary.pipeline_wins == 1
    assert summary.baseline_wins == 1
    assert summary.retrieval is not None
    assert summary.retrieval.query_count == 3
    assert summary.sweep.status == "legacy"
    assert summary.annotation.total_tasks == 2
    assert summary.annotation.current_completed_tasks == 1
    assert summary.annotation.legacy_completed_tasks == 1
    assert any(annotator.annotator_id == "reviewer_legacy" for annotator in summary.annotation.annotators)
    assert any(status.family == "prompt_sweep" and status.status == "excluded" for status in summary.artifact_statuses)

    assert summary_path.exists()
    assert report_path.exists()
    assert all(Path(path).exists() for path in summary.figure_paths)

    markdown = report_path.read_text(encoding="utf-8")
    assert "# Final Analysis Report" in markdown
    assert "Prompt sweep" in markdown
    assert "legacy-format autosaves" in markdown
    assert "retrieval benchmark metrics" in markdown.lower()
