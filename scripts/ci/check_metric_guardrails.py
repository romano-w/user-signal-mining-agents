from __future__ import annotations

import argparse
from pathlib import Path

from user_signal_mining_agents.evaluation.gates import (
    find_critical_metric_regressions,
    load_judge_pairs,
    summarize_metric_deltas,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail when evaluation judge artifacts show critical metric regressions.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("artifacts/runs"),
        help="Directory containing per-prompt judge_baseline.json and judge_pipeline.json artifacts.",
    )
    parser.add_argument(
        "--max-overall-drop",
        type=float,
        default=0.30,
        help="Maximum allowed drop for aggregate overall score (pipeline minus baseline).",
    )
    parser.add_argument(
        "--max-dimension-drop",
        type=float,
        default=0.40,
        help="Maximum allowed drop for any single rubric dimension (pipeline minus baseline).",
    )
    parser.add_argument(
        "--require-pairs",
        action="store_true",
        help="Fail if no judge artifact pairs are available.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    pairs = load_judge_pairs(args.runs_dir)
    if not pairs:
        if args.require_pairs:
            print(f"[gate] No judge artifact pairs found in: {args.runs_dir}")
            return 1
        print(f"[gate] No judge artifact pairs found in: {args.runs_dir}; skipping metric guardrail check.")
        return 0

    summary = summarize_metric_deltas(pairs)
    violations = find_critical_metric_regressions(
        pairs,
        max_overall_drop=args.max_overall_drop,
        max_dimension_drop=args.max_dimension_drop,
    )

    print(f"[gate] Loaded {len(pairs)} prompt-level judge pairs from {args.runs_dir}")
    for metric in summary:
        print(
            "[gate] "
            f"{metric.metric}: baseline={metric.baseline_avg:.3f}, "
            f"pipeline={metric.pipeline_avg:.3f}, delta={metric.delta:+.3f}"
        )

    if violations:
        print("[gate] Critical metric regressions detected:")
        for violation in violations:
            print(
                "[gate] "
                f"{violation.metric}: delta={violation.delta:+.3f} "
                f"(allowed >= {-violation.max_drop:+.3f})"
            )
        return 1

    print("[gate] Metric guardrail check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
