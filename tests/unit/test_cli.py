from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from user_signal_mining_agents import cli
from user_signal_mining_agents.data.ingestion import IngestionResult, SnapshotResult
from user_signal_mining_agents.schemas import (
    DatasetSnapshotManifest,
    EvaluationSummary,
    FounderPrompt,
    JudgeResult,
    JudgeScores,
    PromptEvaluationPair,
)


def test_build_parser_supports_expected_commands() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["search", "--query", "slow service", "--top-k", "3"])
    assert args.command == "search"
    assert args.query == "slow service"
    assert args.top_k == 3


def test_build_parser_supports_domain_filters() -> None:
    parser = cli.build_parser()

    evaluate_args = parser.parse_args(["evaluate", "--domain", "restaurants,saas"])
    assert evaluate_args.command == "evaluate"
    assert evaluate_args.domain == "restaurants,saas"

    baseline_args = parser.parse_args(["run-baseline", "--domain", "saas"])
    assert baseline_args.command == "run-baseline"
    assert baseline_args.domain == "saas"



def test_build_parser_supports_integration_gate_command() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "integration-gate",
            "--reports-dir",
            "reports/research_upgrade",
            "--severity-threshold",
            "5",
            "--summary-out",
            "artifacts/research_upgrade/integration_gate_summary.json",
        ]
    )

    assert args.command == "integration-gate"
    assert args.reports_dir == Path("reports/research_upgrade")
    assert args.severity_threshold == 5
    assert args.summary_out == Path("artifacts/research_upgrade/integration_gate_summary.json")

def test_load_prompts_filters_by_id(monkeypatch: pytest.MonkeyPatch, tmp_settings) -> None:
    prompts = [
        {"id": "a", "statement": "A", "domain": "restaurants"},
        {"id": "b", "statement": "B", "domain": "restaurants"},
    ]
    tmp_settings.founder_prompts_path.write_text(json.dumps(prompts), encoding="utf-8")

    monkeypatch.setattr(cli, "get_settings", lambda: tmp_settings)
    selected = cli._load_prompts("b")

    assert [prompt.id for prompt in selected] == ["b"]


def test_load_prompts_raises_for_unknown_id(monkeypatch: pytest.MonkeyPatch, tmp_settings) -> None:
    tmp_settings.founder_prompts_path.write_text(
        json.dumps([{"id": "a", "statement": "A", "domain": "restaurants"}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "get_settings", lambda: tmp_settings)

    with pytest.raises(ValueError, match="No founder prompt found"):
        cli._load_prompts("missing")


def test_cmd_search_requires_existing_index(monkeypatch: pytest.MonkeyPatch, tmp_settings) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: tmp_settings)

    with pytest.raises(FileNotFoundError, match="Retrieval index not found"):
        cli.cmd_search("slow", 3)


def test_cmd_validate_founder_prompts_validates_file(monkeypatch: pytest.MonkeyPatch, tmp_settings, capsys) -> None:
    prompt_path = tmp_settings.founder_prompts_path
    prompt_path.write_text(
        json.dumps([{"id": "a", "statement": "A", "domain": "restaurants"}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "get_settings", lambda: tmp_settings)

    code = cli.cmd_validate_founder_prompts(prompt_path)
    output = capsys.readouterr().out

    assert code == 0
    assert "Validated 1 founder prompts" in output


def test_cmd_validate_founder_prompts_validates_domain_packs_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
    capsys,
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: tmp_settings)

    code = cli.cmd_validate_founder_prompts(None)
    output = capsys.readouterr().out

    assert code == 0
    assert "Validated 1 domain pack(s)" in output
    assert "founder prompt(s)" in output


def test_main_dispatches_to_command_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "cmd_show_config", lambda: 123)
    result = cli.main(["show-config"])
    assert result == 123


def test_build_parser_supports_annotate_human_command() -> None:
    parser = cli.build_parser()
    args = parser.parse_args([
        "annotate-human",
        "--tasks-dir",
        "artifacts/runs/_human_annotations",
        "--host",
        "127.0.0.1",
        "--port",
        "9001",
        "--annotator-id",
        "reviewer_1",
    ])
    assert args.command == "annotate-human"
    assert args.tasks_dir == Path("artifacts/runs/_human_annotations")
    assert args.host == "127.0.0.1"
    assert args.port == 9001
    assert args.annotator_id == "reviewer_1"


def test_cmd_annotate_human_dispatches_to_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
) -> None:
    from user_signal_mining_agents.evaluation import human_annotation_gui

    called: dict[str, object] = {}

    def _fake_run_annotation_server(
        tasks_dir: Path,
        *,
        host: str,
        port: int,
        default_annotator_id: str,
    ) -> None:
        called["tasks_dir"] = tasks_dir
        called["host"] = host
        called["port"] = port
        called["default_annotator_id"] = default_annotator_id

    monkeypatch.setattr(cli, "get_settings", lambda: tmp_settings)
    monkeypatch.setattr(human_annotation_gui, "run_annotation_server", _fake_run_annotation_server)

    code = cli.cmd_annotate_human(None, host="127.0.0.1", port=9001, annotator_id="reviewer_1")

    assert code == 0
    assert called == {
        "tasks_dir": tmp_settings.run_artifacts_dir / "_human_annotations",
        "host": "127.0.0.1",
        "port": 9001,
        "default_annotator_id": "reviewer_1",
    }


def test_build_parser_supports_variant_commands() -> None:
    parser = cli.build_parser()

    list_args = parser.parse_args(["list-variants"])
    assert list_args.command == "list-variants"

    run_args = parser.parse_args(["run-variant", "--variant", "full_hybrid", "--prompt-id", "p1", "--domain", "saas"])
    assert run_args.command == "run-variant"
    assert run_args.variant == "full_hybrid"
    assert run_args.prompt_id == "p1"
    assert run_args.domain == "saas"

    eval_args = parser.parse_args(["evaluate-variants", "--variants", "critic_loop,full_hybrid", "--domain", "restaurants,saas"])
    assert eval_args.command == "evaluate-variants"
    assert eval_args.variants == "critic_loop,full_hybrid"
    assert eval_args.domain == "restaurants,saas"


def test_parse_variant_ids() -> None:
    assert cli._parse_variant_ids("a,b , c") == ["a", "b", "c"]
    assert cli._parse_variant_ids(None) is None

    with pytest.raises(ValueError, match="no variant ids"):
        cli._parse_variant_ids("  , ")


def test_parse_k_values() -> None:
    assert cli._parse_k_values("10, 3,1,3") == [1, 3, 10]

    with pytest.raises(ValueError, match="did not include any integers"):
        cli._parse_k_values(" , ")

    with pytest.raises(ValueError, match="positive integers"):
        cli._parse_k_values("0,2")
def test_parse_domain_ids() -> None:
    assert cli._parse_domain_ids("restaurants, saas") == ["restaurants", "saas"]
    assert cli._parse_domain_ids(None) is None

    with pytest.raises(ValueError, match="no domain ids"):
        cli._parse_domain_ids(" , ")


def test_build_parser_supports_foundation_contract_commands() -> None:
    parser = cli.build_parser()

    ingest_args = parser.parse_args(["ingest", "--adapter", "app_reviews", "--input-path", "data/input.jsonl"])
    assert ingest_args.command == "ingest"
    assert ingest_args.adapter == "app_reviews"
    assert ingest_args.input_path == Path("data/input.jsonl")

    with pytest.raises(SystemExit):
        parser.parse_args(["ingest", "--adapter", "unknown"])

    snapshot_args = parser.parse_args(["snapshot-data", "--dataset-id", "restaurants_v1"])
    assert snapshot_args.command == "snapshot-data"
    assert snapshot_args.dataset_id == "restaurants_v1"

    retrieval_args = parser.parse_args([
        "eval-retrieval",
        "--label-set",
        "artifacts/retrieval_labels.jsonl",
        "--mode",
        "hybrid",
        "--reranker",
        "token_overlap",
        "--k-values",
        "1,5,10",
        "--top-k",
        "25",
        "--output-dir",
        "artifacts/retrieval_eval",
    ])
    assert retrieval_args.command == "eval-retrieval"
    assert retrieval_args.label_set == Path("artifacts/retrieval_labels.jsonl")
    assert retrieval_args.mode == "hybrid"
    assert retrieval_args.reranker == "token_overlap"
    assert retrieval_args.k_values == "1,5,10"
    assert retrieval_args.top_k == 25
    assert retrieval_args.output_dir == Path("artifacts/retrieval_eval")

    robustness_args = parser.parse_args(
        ["eval-robustness", "--suite", "adversarial_core", "--prompt-id", "p1", "--no-cache"]
    )
    assert robustness_args.command == "eval-robustness"
    assert robustness_args.suite == "adversarial_core"
    assert robustness_args.prompt_id == "p1"
    assert robustness_args.no_cache is True

    compare_args = parser.parse_args(["compare-runs", "--run-a", "run_001", "--run-b", "run_002"])
    assert compare_args.command == "compare-runs"
    assert compare_args.run_a == "run_001"
    assert compare_args.run_b == "run_002"


def test_ingest_and_snapshot_commands_emit_real_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
    capsys,
) -> None:
    records_path = tmp_settings.run_artifacts_dir / "datasets" / "app_reviews.jsonl"
    ingest_manifest_path = tmp_settings.run_artifacts_dir / "datasets" / "app_reviews.manifest.json"
    snapshot_path = tmp_settings.run_artifacts_dir / "datasets" / "snapshots" / "default_20260311T100000Z.json"

    def _fake_run_ingest(*, settings, adapter_id: str, input_path: Path | None) -> IngestionResult:
        assert settings == tmp_settings
        assert adapter_id == "app_reviews"
        assert input_path == Path("tickets.jsonl")
        return IngestionResult(
            adapter_id=adapter_id,
            dataset_id="app_reviews",
            record_count=2,
            records_path=records_path,
            checksum_sha256="abc123",
            source_manifests={"app_reviews": "src999"},
            manifest_path=ingest_manifest_path,
        )

    def _fake_build_snapshot(*, settings, dataset_id: str) -> SnapshotResult:
        assert settings == tmp_settings
        assert dataset_id == "default"
        return SnapshotResult(
            manifest=DatasetSnapshotManifest(
                snapshot_id="default_20260311T100000Z",
                dataset_ids=["app_reviews"],
                record_count=2,
                checksum_sha256="snap456",
                source_manifests={"app_reviews::app_reviews": "src999"},
            ),
            manifest_path=snapshot_path,
        )

    monkeypatch.setattr(cli, "get_settings", lambda: tmp_settings)
    monkeypatch.setattr(cli, "run_ingest", _fake_run_ingest)
    monkeypatch.setattr(cli, "build_snapshot", _fake_build_snapshot)

    code = cli.cmd_ingest("app_reviews", Path("tickets.jsonl"))
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["command"] == "ingest"
    assert payload["payload"]["record_count"] == 2
    assert payload["payload"]["source_manifests"] == {"app_reviews": "src999"}

    code = cli.cmd_snapshot_data("default")
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["command"] == "snapshot-data"
    assert payload["payload"]["record_count"] == 2
    assert payload["payload"]["source_manifests"] == {"app_reviews::app_reviews": "src999"}


def test_cmd_eval_retrieval_dispatches_runner_and_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
    tmp_path,
) -> None:
    from user_signal_mining_agents.evaluation import retrieval_report, retrieval_runner

    label_set = tmp_path / "labels.jsonl"
    label_set.write_text(
        json.dumps({"query": "slow service", "relevant_snippet_ids": ["s1"]}) + "\n",
        encoding="utf-8",
    )

    summary = retrieval_runner.RetrievalEvaluationSummary(
        generated_at=datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc),
        label_set_path=str(label_set),
        retrieval_mode="hybrid",
        reranker="none",
        top_k=5,
        k_values=[1, 3],
        query_count=1,
        dense_weight=1.0,
        lexical_weight=1.0,
        fusion_k=60,
        reranker_weight=0.25,
        candidate_pool=20,
        aggregates={
            "recall_at_k": {"1": 1.0, "3": 1.0},
            "mrr_at_k": {"1": 1.0, "3": 1.0},
            "ndcg_at_k": {"1": 1.0, "3": 1.0},
        },
        queries=[],
    )

    calls: dict[str, object] = {}

    def _fake_run_retrieval_evaluation(label_set_path, settings, **kwargs):
        calls["label_set_path"] = label_set_path
        calls["kwargs"] = kwargs
        assert settings is tmp_settings
        return summary

    def _fake_generate_report(summary_obj, output_dir):
        calls["summary_obj"] = summary_obj
        calls["output_dir"] = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "retrieval_eval_summary.json"
        md_path = output_dir / "retrieval_eval_report.md"
        json_path.write_text("{}", encoding="utf-8")
        md_path.write_text("# report", encoding="utf-8")
        return json_path, md_path

    monkeypatch.setattr(cli, "get_settings", lambda: tmp_settings)
    monkeypatch.setattr(retrieval_runner, "run_retrieval_evaluation", _fake_run_retrieval_evaluation)
    monkeypatch.setattr(retrieval_report, "generate_retrieval_report", _fake_generate_report)

    out_dir = tmp_path / "out"
    code = cli.cmd_eval_retrieval(
        label_set,
        mode="hybrid",
        reranker="token_overlap",
        k_values="1,3",
        top_k=5,
        output_dir=out_dir,
    )

    assert code == 0
    assert calls["label_set_path"] == label_set
    assert calls["kwargs"] == {
        "mode": "hybrid",
        "reranker": "token_overlap",
        "top_k": 5,
        "k_values": [1, 3],
    }
    assert calls["summary_obj"] is summary
    assert calls["output_dir"] == out_dir


def test_compare_runs_still_emits_foundation_placeholder(capsys) -> None:
    code = cli.cmd_compare_runs("run_a", "run_b")
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "foundation-placeholder"
    assert payload["command"] == "compare-runs"
    assert payload["payload"] == {"run_a": "run_a", "run_b": "run_b"}

def test_cmd_evaluate_generates_failure_taxonomy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
) -> None:
    from user_signal_mining_agents.evaluation import failure_taxonomy, report, runner

    prompt = FounderPrompt(id="p1", statement="Question", domain="restaurants")
    judge_baseline = JudgeResult(
        prompt_id="p1",
        system_variant="baseline",
        scores=JudgeScores(
            relevance=3.0,
            actionability=3.0,
            evidence_grounding=3.0,
            contradiction_handling=3.0,
            non_redundancy=3.0,
            rationale="baseline rationale",
        ),
    )
    judge_pipeline = JudgeResult(
        prompt_id="p1",
        system_variant="pipeline",
        scores=JudgeScores(
            relevance=4.0,
            actionability=4.0,
            evidence_grounding=4.0,
            contradiction_handling=4.0,
            non_redundancy=4.0,
            rationale="pipeline rationale",
        ),
    )
    summary = EvaluationSummary(
        pairs=[
            PromptEvaluationPair(
                prompt=prompt,
                baseline_scores=judge_baseline,
                pipeline_scores=judge_pipeline,
            )
        ]
    )

    called: dict[str, object] = {}

    monkeypatch.setattr(cli, "get_settings", lambda: tmp_settings)
    monkeypatch.setattr(runner, "run_evaluation", lambda *_args, **_kwargs: summary)
    monkeypatch.setattr(report, "generate_report", lambda *_args, **_kwargs: tmp_settings.run_artifacts_dir / "evaluation_report.md")

    def _fake_failure_taxonomy(
        run_artifacts_dir: Path,
        *,
        prompt_ids: list[str] | None = None,
        score_threshold: float = 3.5,
    ) -> tuple[list[object], Path, Path]:
        called["run_artifacts_dir"] = run_artifacts_dir
        called["prompt_ids"] = prompt_ids
        called["score_threshold"] = score_threshold
        return (
            [],
            run_artifacts_dir / "failure_tags.jsonl",
            run_artifacts_dir / "failure_taxonomy_report.md",
        )

    monkeypatch.setattr(failure_taxonomy, "generate_failure_taxonomy", _fake_failure_taxonomy)

    code = cli.cmd_evaluate("p1", no_cache=False)

    assert code == 0
    assert called["run_artifacts_dir"] == tmp_settings.run_artifacts_dir
    assert called["prompt_ids"] == ["p1"]
    assert called["score_threshold"] == 3.5

def test_cmd_eval_robustness_returns_non_zero_on_gate_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
) -> None:
    from user_signal_mining_agents.evaluation import robustness_runner
    from user_signal_mining_agents.schemas import JudgeScores

    control = JudgeScores(
        relevance=4.0,
        actionability=4.0,
        evidence_grounding=4.0,
        contradiction_handling=4.0,
        non_redundancy=4.0,
        rationale="control",
    )
    perturbed = JudgeScores(
        relevance=3.0,
        actionability=3.0,
        evidence_grounding=3.0,
        contradiction_handling=3.0,
        non_redundancy=3.0,
        rationale="perturbed",
    )

    summary = robustness_runner.RobustnessSuiteSummary(
        suite_id="adversarial_core",
        suite_description="core",
        prompt_ids=["p1"],
        thresholds=robustness_runner.RobustnessGateThresholds(
            max_overall_drop=0.5,
            max_dimension_drop=1.0,
            min_case_pass_rate=1.0,
        ),
        total_cases=1,
        passed_cases=0,
        failed_cases=1,
        pass_rate=0.0,
        gate_passed=False,
        failed_case_keys=["p1:rb_negation_flip"],
        gate_failure_reasons=["Case pass rate 0.00% is below required 100.00%."],
        outcomes=[
            robustness_runner.RobustnessCaseOutcome(
                prompt_id="p1",
                case_id="rb_negation_flip",
                family="negation",
                description="flip",
                expected_behavior="remain stable",
                perturbed_prompt_id="p1__rb_negation_flip",
                perturbed_statement="not Why are diners returning?",
                control_scores=control,
                perturbed_scores=perturbed,
                dimension_deltas={
                    "relevance": -1.0,
                    "actionability": -1.0,
                    "evidence_grounding": -1.0,
                    "contradiction_handling": -1.0,
                    "non_redundancy": -1.0,
                },
                delta_overall=-1.0,
                passed=False,
                failure_reasons=["overall drop -1.00 exceeded max -0.50"],
            )
        ],
    )

    monkeypatch.setattr(cli, "get_settings", lambda: tmp_settings)
    monkeypatch.setattr(robustness_runner, "run_robustness_suite", lambda *_args, **_kwargs: summary)

    code = cli.cmd_eval_robustness("adversarial_core", "p1", no_cache=True)

    assert code == 2
    summary_path = (
        tmp_settings.run_artifacts_dir.parent
        / "robustness_runs"
        / "adversarial_core"
        / "robustness_summary.json"
    )
    assert summary_path.exists()






