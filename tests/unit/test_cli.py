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

    with pytest.raises(FileNotFoundError, match="Dense index not found"):
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
