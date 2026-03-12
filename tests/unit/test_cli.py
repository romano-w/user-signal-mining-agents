from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from user_signal_mining_agents import cli


def test_build_parser_supports_expected_commands() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["search", "--query", "slow service", "--top-k", "3"])
    assert args.command == "search"
    assert args.query == "slow service"
    assert args.top_k == 3


def test_load_prompts_filters_by_id(monkeypatch: pytest.MonkeyPatch, tmp_settings) -> None:
    prompts = [
        {"id": "a", "statement": "A", "domain": "restaurants"},
        {"id": "b", "statement": "B", "domain": "restaurants"},
    ]
    tmp_settings.founder_prompts_path.write_text(json.dumps(prompts), encoding="utf-8")

    monkeypatch.setattr(cli, "get_settings", lambda: tmp_settings)
    selected = cli._load_prompts("b")

    assert [p.id for p in selected] == ["b"]


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

    code = cli.cmd_validate_founder_prompts(None)
    output = capsys.readouterr().out

    assert code == 0
    assert "Validated 1 founder prompts" in output


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

    run_args = parser.parse_args(["run-variant", "--variant", "full_hybrid", "--prompt-id", "p1"])
    assert run_args.command == "run-variant"
    assert run_args.variant == "full_hybrid"
    assert run_args.prompt_id == "p1"

    eval_args = parser.parse_args(["evaluate-variants", "--variants", "critic_loop,full_hybrid"])
    assert eval_args.command == "evaluate-variants"
    assert eval_args.variants == "critic_loop,full_hybrid"


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


def test_build_parser_supports_foundation_contract_commands() -> None:
    parser = cli.build_parser()

    ingest_args = parser.parse_args(["ingest", "--adapter", "app_reviews", "--input-path", "data/input.jsonl"])
    assert ingest_args.command == "ingest"
    assert ingest_args.adapter == "app_reviews"
    assert ingest_args.input_path == Path("data/input.jsonl")

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

    robustness_args = parser.parse_args(["eval-robustness", "--suite", "adversarial_core"])
    assert robustness_args.command == "eval-robustness"
    assert robustness_args.suite == "adversarial_core"

    compare_args = parser.parse_args(["compare-runs", "--run-a", "run_001", "--run-b", "run_002"])
    assert compare_args.command == "compare-runs"
    assert compare_args.run_a == "run_001"
    assert compare_args.run_b == "run_002"


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


def test_foundation_placeholder_commands_emit_contract_payload(capsys) -> None:
    code = cli.cmd_ingest("support_tickets", Path("tickets.jsonl"))
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "foundation-placeholder"
    assert payload["command"] == "ingest"
    assert payload["payload"]["adapter"] == "support_tickets"

    code = cli.cmd_compare_runs("run_a", "run_b")
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "compare-runs"
    assert payload["payload"] == {"run_a": "run_a", "run_b": "run_b"}
