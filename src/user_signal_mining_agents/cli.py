from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pydantic import TypeAdapter

from .config import ROOT_DIR, ensure_scaffold_directories, get_settings
from .data.fetch_yelp import EXPECTED_YELP_FILES, ensure_yelp_dataset
from .retrieval.index import search_dense_index
from .schemas import FounderPrompt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="usm",
        description="Utilities for the founder-grounded review mining project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "show-config",
        help="Print the resolved application settings.",
    )

    subparsers.add_parser(
        "bootstrap",
        help="Create the default data, prompt, and artifact directories.",
    )

    validate_parser = subparsers.add_parser(
        "validate-founder-prompts",
        help="Validate a founder prompt JSON file against the shared schema.",
    )
    validate_parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Optional path to the founder prompt JSON file.",
    )

    fetch_parser = subparsers.add_parser(
        "fetch-yelp-dataset",
        help="Download and extract the Yelp Open Dataset into the configured raw data directory.",
    )
    fetch_parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the zip download step and use an existing local tar archive instead.",
    )
    fetch_parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the Yelp zip and replace any existing cached copy.",
    )
    fetch_parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract the tar contents even if the JSON files already exist.",
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Query the dense index and print top-K snippets.",
    )
    search_parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Free-text query to search the review index.",
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to return.",
    )

    run_baseline_parser = subparsers.add_parser(
        "run-baseline",
        help="Run the zero-shot baseline for one or all founder prompts.",
    )
    run_baseline_parser.add_argument(
        "--prompt-id",
        type=str,
        default=None,
        help="Run only this prompt. Omit to run all.",
    )

    run_pipeline_parser = subparsers.add_parser(
        "run-pipeline",
        help="Run the multi-step grounded pipeline for one or all founder prompts.",
    )
    run_pipeline_parser.add_argument(
        "--prompt-id",
        type=str,
        default=None,
        help="Run only this prompt. Omit to run all.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run baseline + pipeline + judge for all prompts, then generate a report.",
    )
    evaluate_parser.add_argument(
        "--prompt-id",
        type=str,
        default=None,
        help="Evaluate only this prompt. Omit to evaluate all.",
    )
    evaluate_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Re-run both systems even if cached results exist.",
    )

    return parser


def cmd_show_config() -> int:
    settings = get_settings()
    print(json.dumps(settings.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


def cmd_bootstrap() -> int:
    settings = get_settings()
    directories = ensure_scaffold_directories(settings)
    print("Created or verified directories:")
    for directory in directories:
        try:
            display_path = directory.relative_to(ROOT_DIR)
        except ValueError:
            display_path = directory
        print(f"- {display_path}")
    return 0


def cmd_validate_founder_prompts(path: Path | None) -> int:
    settings = get_settings()
    prompt_path = path or settings.founder_prompts_path
    if not prompt_path.exists():
        raise FileNotFoundError(f"Founder prompt file not found: {prompt_path}")
    data = json.loads(prompt_path.read_text(encoding="utf-8"))
    prompts = TypeAdapter(list[FounderPrompt]).validate_python(data)
    print(f"Validated {len(prompts)} founder prompts from {prompt_path}.")
    return 0


def cmd_fetch_yelp_dataset(
    *,
    skip_download: bool,
    force_download: bool,
    force_extract: bool,
) -> int:
    settings = get_settings()
    dataset_dir = ensure_yelp_dataset(
        settings,
        skip_download=skip_download,
        force_download=force_download,
        force_extract=force_extract,
    )
    print("Yelp dataset is ready:")
    for name in EXPECTED_YELP_FILES:
        print(f"- {dataset_dir / name}")
    return 0


def cmd_search(query: str, top_k: int) -> int:
    settings = get_settings()
    index_dir = settings.index_dir
    if not (index_dir / "metadata.json").exists():
        raise FileNotFoundError(
            f"Dense index not found at {index_dir}. "
            "Run `uv run python scripts/build_index.py` first."
        )
    hits = search_dense_index(
        query,
        index_dir=index_dir,
        top_k=top_k,
    )
    print(f"Top {len(hits)} results for: {query!r}\n")
    for rank, hit in enumerate(hits, start=1):
        biz = hit.snippet.business_name or hit.snippet.business_id
        print(f"  [{rank}] score={hit.score:.4f}  {biz}")
        print(f"       {hit.snippet.text[:200]}")
        print()
    return 0


def _load_prompts(prompt_id: str | None = None) -> list[FounderPrompt]:
    settings = get_settings()
    data = json.loads(settings.founder_prompts_path.read_text(encoding="utf-8"))
    prompts = TypeAdapter(list[FounderPrompt]).validate_python(data)
    if prompt_id:
        prompts = [p for p in prompts if p.id == prompt_id]
        if not prompts:
            raise ValueError(f"No founder prompt found with id={prompt_id!r}")
    return prompts


def cmd_run_baseline(prompt_id: str | None) -> int:
    from .agents.baseline import run_baseline

    prompts = _load_prompts(prompt_id)
    for prompt in prompts:
        run_baseline(prompt)
    print(f"Baseline complete for {len(prompts)} prompt(s).")
    return 0


def cmd_run_pipeline(prompt_id: str | None) -> int:
    from .agents.pipeline import run_pipeline

    prompts = _load_prompts(prompt_id)
    for prompt in prompts:
        run_pipeline(prompt)
    print(f"Pipeline complete for {len(prompts)} prompt(s).")
    return 0


def cmd_evaluate(prompt_id: str | None, *, no_cache: bool) -> int:
    from .evaluation.runner import run_evaluation
    from .evaluation.report import generate_report

    settings = get_settings()
    prompt_ids = [prompt_id] if prompt_id else None
    summary = run_evaluation(settings, prompt_ids=prompt_ids, skip_cached=not no_cache)
    report_path = generate_report(summary, settings.run_artifacts_dir)
    print(f"\nReport saved to {report_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "show-config":
        return cmd_show_config()
    if args.command == "bootstrap":
        return cmd_bootstrap()
    if args.command == "validate-founder-prompts":
        return cmd_validate_founder_prompts(args.path)
    if args.command == "fetch-yelp-dataset":
        return cmd_fetch_yelp_dataset(
            skip_download=args.skip_download,
            force_download=args.force_download,
            force_extract=args.force_extract,
        )
    if args.command == "search":
        return cmd_search(args.query, args.top_k)
    if args.command == "run-baseline":
        return cmd_run_baseline(args.prompt_id)
    if args.command == "run-pipeline":
        return cmd_run_pipeline(args.prompt_id)
    if args.command == "evaluate":
        return cmd_evaluate(args.prompt_id, no_cache=args.no_cache)

    parser.error(f"Unknown command: {args.command}")
    return 2
