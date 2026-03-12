from __future__ import annotations

import json
from pathlib import Path

import pytest

from user_signal_mining_agents import cli
from user_signal_mining_agents.integration.gates import default_gate_inputs, run_integration_gates


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_required_reports(
    report_dir: Path,
    *,
    failure_tags: list[dict[str, object]] | None = None,
) -> None:
    _write_json(
        report_dir / "schema_compatibility.json",
        {
            "report_type": "schema_compatibility",
            "status": "pass",
            "checked_contracts": ["DatasetRecord", "FailureTag"],
        },
    )
    _write_json(
        report_dir / "retrieval_eval_summary.json",
        {
            "generated_at": "2026-03-12T00:00:00+00:00",
            "label_set_path": "artifacts/retrieval_labels.jsonl",
            "retrieval_mode": "hybrid",
            "reranker": "token_overlap",
            "top_k": 10,
            "k_values": [1, 3, 5, 10],
            "query_count": 2,
            "dense_weight": 1.0,
            "lexical_weight": 1.0,
            "fusion_k": 60,
            "reranker_weight": 0.25,
            "candidate_pool": 200,
            "aggregates": {
                "recall_at_k": {"1": 0.50, "3": 1.0, "5": 1.0, "10": 1.0},
                "mrr_at_k": {"1": 0.50, "3": 0.67, "5": 0.67, "10": 0.67},
                "ndcg_at_k": {"1": 0.50, "3": 0.75, "5": 0.75, "10": 0.75},
            },
            "queries": [],
        },
    )
    _write_json(
        report_dir / "robustness_report.json",
        {
            "report_type": "robustness",
            "status": "pass",
            "suite": "core",
        },
    )
    _write_json(
        report_dir / "domain_transfer_report.json",
        {
            "report_type": "domain_transfer",
            "status": "pass",
            "domains": ["restaurants", "saas"],
        },
    )
    _write_json(
        report_dir / "failure_tags_report.json",
        {
            "report_type": "failure_taxonomy",
            "tags": failure_tags
            if failure_tags is not None
            else [
                {
                    "tag_id": "ft_low",
                    "category": "minor_style",
                    "severity": 2,
                    "description": "Minor style issue",
                    "evidence_refs": [],
                }
            ],
        },
    )


def test_default_gate_inputs_uses_expected_filenames(tmp_path: Path) -> None:
    inputs = default_gate_inputs(tmp_path)

    assert inputs.schema_compatibility_path == tmp_path / "schema_compatibility.json"
    assert inputs.retrieval_report_path == tmp_path / "retrieval_eval_summary.json"
    assert inputs.robustness_report_path == tmp_path / "robustness_report.json"
    assert inputs.domain_transfer_report_path == tmp_path / "domain_transfer_report.json"
    assert inputs.failure_tags_report_path == tmp_path / "failure_tags_report.json"


def test_run_integration_gates_passes_with_complete_reports(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    _write_required_reports(report_dir)

    summary = run_integration_gates(default_gate_inputs(report_dir))

    assert summary.overall_status == "pass"
    assert summary.blocking_failures == 0
    assert all(check.status == "pass" for check in summary.checks)


def test_run_integration_gates_fails_when_required_report_is_missing(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    _write_required_reports(report_dir)
    (report_dir / "retrieval_eval_summary.json").unlink()

    summary = run_integration_gates(default_gate_inputs(report_dir))

    assert summary.overall_status == "fail"
    retrieval_check = next(check for check in summary.checks if check.check == "retrieval")
    assert retrieval_check.status == "fail"
    assert "missing" in retrieval_check.details


def test_run_integration_gates_fails_when_retrieval_metrics_are_missing(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    _write_required_reports(report_dir)
    _write_json(
        report_dir / "retrieval_eval_summary.json",
        {
            "generated_at": "2026-03-12T00:00:00+00:00",
            "label_set_path": "artifacts/retrieval_labels.jsonl",
            "retrieval_mode": "hybrid",
            "reranker": "none",
            "top_k": 10,
            "k_values": [1, 3, 5, 10],
            "query_count": 2,
            "dense_weight": 1.0,
            "lexical_weight": 1.0,
            "fusion_k": 60,
            "reranker_weight": 0.0,
            "candidate_pool": 200,
            "aggregates": {
                "recall_at_k": {"1": 0.5, "3": 0.8, "5": 1.0, "10": 1.0},
                "mrr_at_k": {"1": 0.5, "3": 0.6, "5": 0.6, "10": 0.6},
            },
            "queries": [],
        },
    )

    summary = run_integration_gates(default_gate_inputs(report_dir))

    assert summary.overall_status == "fail"
    retrieval_check = next(check for check in summary.checks if check.check == "retrieval")
    assert retrieval_check.status == "fail"
    assert "missing aggregate metrics" in retrieval_check.details


def test_run_integration_gates_blocks_on_high_severity_failure_tags(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    _write_required_reports(
        report_dir,
        failure_tags=[
            {
                "tag_id": "ft_blocker",
                "category": "hallucination",
                "severity": 5,
                "description": "Unsupported claim with fabricated citation",
                "evidence_refs": ["snippet_42"],
            }
        ],
    )

    summary = run_integration_gates(default_gate_inputs(report_dir, high_severity_threshold=4))

    assert summary.overall_status == "fail"
    failure_check = next(check for check in summary.checks if check.check == "failure_tags")
    assert failure_check.status == "fail"
    assert "ft_blocker" in failure_check.details


def test_cli_integration_gate_writes_summary_and_returns_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report_dir = tmp_path / "reports"
    summary_out = tmp_path / "out" / "integration_gate_summary.json"
    _write_required_reports(report_dir)

    code = cli.cmd_integration_gate(
        report_dir,
        schema_report=None,
        retrieval_report=None,
        robustness_report=None,
        domain_transfer_report=None,
        failure_tags_report=None,
        severity_threshold=4,
        summary_out=summary_out,
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["overall_status"] == "pass"
    assert summary_out.exists()


def test_cli_integration_gate_returns_non_zero_on_failure(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    _write_required_reports(report_dir)
    (report_dir / "schema_compatibility.json").write_text("{}", encoding="utf-8")

    code = cli.cmd_integration_gate(
        report_dir,
        schema_report=None,
        retrieval_report=None,
        robustness_report=None,
        domain_transfer_report=None,
        failure_tags_report=None,
        severity_threshold=4,
        summary_out=None,
    )

    assert code == 1
