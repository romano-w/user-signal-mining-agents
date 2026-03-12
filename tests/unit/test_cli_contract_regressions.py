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
    assert retrieval_args.mode is None
    assert retrieval_args.reranker is None
    assert retrieval_args.k_values == "1,3,5,10"
    assert retrieval_args.top_k is None
    assert retrieval_args.output_dir is None

    robustness_args = parser.parse_args(["eval-robustness"])
    assert robustness_args.suite == "default"
    assert robustness_args.prompt_id is None
    assert robustness_args.no_cache is False


@pytest.mark.parametrize(
    ("handler", "argv", "expected_args", "expected_kwargs"),
    [
        (
            "cmd_ingest",
            ["ingest", "--adapter", "support_tickets", "--input-path", "data/input.jsonl"],
            ("support_tickets", Path("data/input.jsonl")),
            {},
        ),
        (
            "cmd_snapshot_data",
            ["snapshot-data", "--dataset-id", "restaurants_v2"],
            ("restaurants_v2",),
            {},
        ),
        (
            "cmd_eval_retrieval",
            ["eval-retrieval", "--label-set", "artifacts/retrieval_labels.jsonl"],
            (Path("artifacts/retrieval_labels.jsonl"),),
            {
                "mode": None,
                "reranker": None,
                "k_values": "1,3,5,10",
                "top_k": None,
                "output_dir": None,
            },
        ),
        (
            "cmd_eval_robustness",
            ["eval-robustness", "--suite", "adversarial_core"],
            ("adversarial_core", None),
            {"no_cache": False},
        ),
        (
            "cmd_compare_runs",
            ["compare-runs", "--run-a", "run_001", "--run-b", "run_002"],
            ("run_001", "run_002"),
            {},
        ),
    ],
)
def test_main_dispatches_foundation_surfaces(
    monkeypatch: pytest.MonkeyPatch,
    handler: str,
    argv: list[str],
    expected_args: tuple[object, ...],
    expected_kwargs: dict[str, object],
) -> None:
    called: dict[str, object] = {}

    def _fake_handler(*args: object, **kwargs: object) -> int:
        called["args"] = args
        called["kwargs"] = kwargs
        return 211

    monkeypatch.setattr(cli, handler, _fake_handler)

    result = cli.main(argv)

    assert result == 211
    assert called["args"] == expected_args
    assert called["kwargs"] == expected_kwargs


def test_compare_runs_placeholder_payload_shape_is_stable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = cli.cmd_compare_runs("run_a", "run_b")

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "status": "foundation-placeholder",
        "command": "compare-runs",
        "notes": (
            "This command surface is intentionally scaffolded in the foundation branch. "
            "Program-specific agent branches should replace this placeholder implementation."
        ),
        "payload": {"run_a": "run_a", "run_b": "run_b"},
    }
