from __future__ import annotations

import json
from pathlib import Path

import pytest

from user_signal_mining_agents import cli


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

    with pytest.raises(FileNotFoundError, match="Dense index not found"):
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

    snapshot_args = parser.parse_args(["snapshot-data", "--dataset-id", "restaurants_v1"])
    assert snapshot_args.command == "snapshot-data"
    assert snapshot_args.dataset_id == "restaurants_v1"

    retrieval_args = parser.parse_args(["eval-retrieval", "--label-set", "artifacts/retrieval_labels.jsonl"])
    assert retrieval_args.command == "eval-retrieval"
    assert retrieval_args.label_set == Path("artifacts/retrieval_labels.jsonl")

    robustness_args = parser.parse_args(["eval-robustness", "--suite", "adversarial_core"])
    assert robustness_args.command == "eval-robustness"
    assert robustness_args.suite == "adversarial_core"

    compare_args = parser.parse_args(["compare-runs", "--run-a", "run_001", "--run-b", "run_002"])
    assert compare_args.command == "compare-runs"
    assert compare_args.run_a == "run_001"
    assert compare_args.run_b == "run_002"


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
