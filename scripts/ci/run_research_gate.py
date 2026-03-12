from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GateStep:
    name: str
    command: list[str]


def _run_step(step: GateStep) -> int:
    printable = " ".join(shlex.quote(part) for part in step.command)
    print(f"[gate] Running: {step.name}")
    print(f"[gate] Command: {printable}")
    completed = subprocess.run(step.command, check=False)
    if completed.returncode != 0:
        print(f"[gate] Step failed: {step.name} (exit={completed.returncode})")
        return completed.returncode
    print(f"[gate] Step passed: {step.name}")
    return 0


def _branch_ready_steps() -> list[GateStep]:
    return [
        GateStep(
            name="contract-and-cli-regressions",
            command=[
                "uv",
                "run",
                "pytest",
                "tests/unit/test_foundation_contracts.py",
                "tests/unit/test_schema_contract_compatibility.py",
                "tests/unit/test_cli.py",
                "tests/unit/test_cli_contract_regressions.py",
                "tests/unit/test_evaluation_gates.py",
                "-q",
                "--maxfail=1",
            ],
        ),
        GateStep(
            name="unit-offline-suite",
            command=[
                "uv",
                "run",
                "pytest",
                "-m",
                "not integration and not live",
                "--cov=user_signal_mining_agents",
                "--cov-report=term-missing",
                "--cov-report=xml",
                "--cov-fail-under=80",
                "--junitxml=artifacts/test-results/branch-ready-junit.xml",
                "--maxfail=1",
            ],
        ),
    ]


def _integration_ready_steps() -> list[GateStep]:
    return [
        GateStep(
            name="integration-offline-suite",
            command=[
                "uv",
                "run",
                "pytest",
                "-m",
                "integration and not live",
                "-q",
                "--maxfail=1",
                "--junitxml=artifacts/test-results/integration-ready-junit.xml",
            ],
        )
    ]


def _main_ready_steps(
    *,
    metric_runs_dir: Path,
    max_overall_drop: float,
    max_dimension_drop: float,
    require_metric_pairs: bool,
) -> list[GateStep]:
    metric_command = [
        "uv",
        "run",
        "python",
        "scripts/ci/check_metric_guardrails.py",
        "--runs-dir",
        str(metric_runs_dir),
        "--max-overall-drop",
        str(max_overall_drop),
        "--max-dimension-drop",
        str(max_dimension_drop),
    ]
    if require_metric_pairs:
        metric_command.append("--require-pairs")

    return [
        GateStep(
            name="full-offline-suite",
            command=[
                "uv",
                "run",
                "pytest",
                "-m",
                "not live",
                "--cov=user_signal_mining_agents",
                "--cov-report=term-missing",
                "--cov-report=xml",
                "--cov-fail-under=80",
                "--junitxml=artifacts/test-results/main-ready-junit.xml",
                "--maxfail=1",
            ],
        ),
        GateStep(
            name="metric-guardrails",
            command=metric_command,
        ),
    ]


def _steps_for_level(
    level: str,
    *,
    metric_runs_dir: Path,
    max_overall_drop: float,
    max_dimension_drop: float,
    require_metric_pairs: bool,
) -> list[GateStep]:
    steps = list(_branch_ready_steps())

    if level in {"integration-ready", "main-ready"}:
        steps.extend(_integration_ready_steps())

    if level == "main-ready":
        steps.extend(
            _main_ready_steps(
                metric_runs_dir=metric_runs_dir,
                max_overall_drop=max_overall_drop,
                max_dimension_drop=max_dimension_drop,
                require_metric_pairs=require_metric_pairs,
            )
        )

    return steps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute standardized branch/integration/main research CI gates.",
    )
    parser.add_argument(
        "--level",
        choices=["branch-ready", "integration-ready", "main-ready"],
        required=True,
        help="Gate level to run.",
    )
    parser.add_argument(
        "--metric-runs-dir",
        type=Path,
        default=Path("artifacts/runs"),
        help="Directory containing judge_* artifacts for metric guardrail checks.",
    )
    parser.add_argument(
        "--max-overall-drop",
        type=float,
        default=0.30,
        help="Maximum allowed aggregate overall score drop for pipeline vs baseline.",
    )
    parser.add_argument(
        "--max-dimension-drop",
        type=float,
        default=0.40,
        help="Maximum allowed single-dimension score drop for pipeline vs baseline.",
    )
    parser.add_argument(
        "--require-metric-pairs",
        action="store_true",
        help="Require on-disk judge artifact pairs for metric checks in main-ready gate.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    steps = _steps_for_level(
        args.level,
        metric_runs_dir=args.metric_runs_dir,
        max_overall_drop=args.max_overall_drop,
        max_dimension_drop=args.max_dimension_drop,
        require_metric_pairs=args.require_metric_pairs,
    )

    print(f"[gate] Starting {args.level} gate with {len(steps)} step(s).")
    for step in steps:
        code = _run_step(step)
        if code != 0:
            return code

    print(f"[gate] {args.level} gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
