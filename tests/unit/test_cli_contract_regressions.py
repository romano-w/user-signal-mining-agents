from __future__ import annotations

import json
from pathlib import Path

import pytest

from user_signal_mining_agents import cli


def test_build_parser_foundation_command_defaults() -> None:
    parser = cli.build_parser()

    ingest_args = parser.parse_args(["ingest"])
    assert ingest_args.adapter == "yelp"
    assert ingest_args.input_path is None

    snapshot_args = parser.parse_args(["snapshot-data"])
    assert snapshot_args.dataset_id == "default"

    retrieval_args = parser.parse_args(["eval-retrieval"])
    assert retrieval_args.label_set is None

    robustness_args = parser.parse_args(["eval-robustness"])
    assert robustness_args.suite == "default"


@pytest.mark.parametrize(
    ("handler", "argv", "expected_args"),
    [
        (
            "cmd_ingest",
            ["ingest", "--adapter", "support_tickets", "--input-path", "data/input.jsonl"],
            ("support_tickets", Path("data/input.jsonl")),
        ),
        (
            "cmd_snapshot_data",
            ["snapshot-data", "--dataset-id", "restaurants_v2"],
            ("restaurants_v2",),
        ),
        (
            "cmd_eval_retrieval",
            ["eval-retrieval", "--label-set", "artifacts/retrieval_labels.jsonl"],
            (Path("artifacts/retrieval_labels.jsonl"),),
        ),
        (
            "cmd_eval_robustness",
            ["eval-robustness", "--suite", "adversarial_core"],
            ("adversarial_core",),
        ),
        (
            "cmd_compare_runs",
            ["compare-runs", "--run-a", "run_001", "--run-b", "run_002"],
            ("run_001", "run_002"),
        ),
    ],
)
def test_main_dispatches_foundation_surfaces(
    monkeypatch: pytest.MonkeyPatch,
    handler: str,
    argv: list[str],
    expected_args: tuple[object, ...],
) -> None:
    called: dict[str, tuple[object, ...]] = {}

    def _fake_handler(*args: object) -> int:
        called["args"] = args
        return 211

    monkeypatch.setattr(cli, handler, _fake_handler)

    result = cli.main(argv)

    assert result == 211
    assert called["args"] == expected_args


@pytest.mark.parametrize(
    ("command", "fn_name", "kwargs", "expected_payload"),
    [
        (
            "ingest",
            "cmd_ingest",
            {"adapter": "app_reviews", "input_path": Path("incoming.jsonl")},
            {"adapter": "app_reviews", "input_path": "incoming.jsonl"},
        ),
        (
            "snapshot-data",
            "cmd_snapshot_data",
            {"dataset_id": "restaurants_v3"},
            {"dataset_id": "restaurants_v3"},
        ),
        (
            "eval-retrieval",
            "cmd_eval_retrieval",
            {"label_set": Path("labels.jsonl")},
            {"label_set": "labels.jsonl"},
        ),
        (
            "eval-robustness",
            "cmd_eval_robustness",
            {"suite": "noise_shift"},
            {"suite": "noise_shift"},
        ),
        (
            "compare-runs",
            "cmd_compare_runs",
            {"run_a": "run_a", "run_b": "run_b"},
            {"run_a": "run_a", "run_b": "run_b"},
        ),
    ],
)
def test_foundation_placeholder_payload_shape_is_stable(
    capsys: pytest.CaptureFixture[str],
    command: str,
    fn_name: str,
    kwargs: dict[str, object],
    expected_payload: dict[str, object],
) -> None:
    command_fn = getattr(cli, fn_name)

    code = command_fn(**kwargs)

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "status": "foundation-placeholder",
        "command": command,
        "notes": (
            "This command surface is intentionally scaffolded in the foundation branch. "
            "Program-specific agent branches should replace this placeholder implementation."
        ),
        "payload": expected_payload,
    }
