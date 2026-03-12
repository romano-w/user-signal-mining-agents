from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

load_dotenv()  # Export .env vars to os.environ for HF Hub, etc.

from pydantic import TypeAdapter

from .config import ROOT_DIR, ensure_scaffold_directories, get_settings
from .data.fetch_yelp import EXPECTED_YELP_FILES, ensure_yelp_dataset
from .domain_packs import load_domain_packs, load_founder_prompts, parse_domain_ids
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

    run_baseline_parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Comma-separated domain ids. Omit to use enabled domain packs.",
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

    run_pipeline_parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Comma-separated domain ids. Omit to use enabled domain packs.",
    )

    list_variants_parser = subparsers.add_parser(
        "list-variants",
        help="List available experimental pipeline variants.",
    )
    list_variants_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full stage graph for each variant.",
    )

    run_variant_parser = subparsers.add_parser(
        "run-variant",
        help="Run one experimental variant for one or all prompts.",
    )
    run_variant_parser.add_argument(
        "--variant",
        type=str,
        required=True,
        help="Variant id to run (use `usm list-variants`).",
    )
    run_variant_parser.add_argument(
        "--prompt-id",
        type=str,
        default=None,
        help="Run only this prompt. Omit to run all.",
    )

    run_variant_parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Comma-separated domain ids. Omit to use enabled domain packs.",
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

    evaluate_parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Comma-separated domain ids. Omit to use enabled domain packs.",
    )

    evaluate_variants_parser = subparsers.add_parser(
        "evaluate-variants",
        help="Compare experimental variants against the control pipeline.",
    )
    evaluate_variants_parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated variant ids. Omit to run the default candidate set.",
    )
    evaluate_variants_parser.add_argument(
        "--prompt-id",
        type=str,
        default=None,
        help="Evaluate only this prompt. Omit for staged default prompt subset.",
    )
    evaluate_variants_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Re-run variants and judge even if cached results exist.",
    )

    evaluate_variants_parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Comma-separated domain ids. Omit to use enabled domain packs.",
    )

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="[Foundation placeholder] Run adapter-based source ingestion.",
    )
    ingest_parser.add_argument(
        "--adapter",
        type=str,
        default="yelp",
        help="Adapter id to run (for example: yelp, app_reviews, support_tickets).",
    )
    ingest_parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Optional path consumed by the adapter.",
    )

    snapshot_parser = subparsers.add_parser(
        "snapshot-data",
        help="[Foundation placeholder] Create immutable dataset snapshot manifests.",
    )
    snapshot_parser.add_argument(
        "--dataset-id",
        type=str,
        default="default",
        help="Logical dataset id included in the snapshot manifest.",
    )

    eval_retrieval_parser = subparsers.add_parser(
        "eval-retrieval",
        help="[Foundation placeholder] Evaluate retrieval quality metrics.",
    )
    eval_retrieval_parser.add_argument(
        "--label-set",
        type=Path,
        default=None,
        help="Optional labeled query set path for retrieval metrics.",
    )

    eval_robustness_parser = subparsers.add_parser(
        "eval-robustness",
        help="[Foundation placeholder] Run robustness stress-case evaluation.",
    )
    eval_robustness_parser.add_argument(
        "--suite",
        type=str,
        default="default",
        help="Robustness suite id.",
    )

    compare_runs_parser = subparsers.add_parser(
        "compare-runs",
        help="[Foundation placeholder] Compare two experiment manifests.",
    )
    compare_runs_parser.add_argument(
        "--run-a",
        type=str,
        required=True,
        help="Run id A.",
    )
    compare_runs_parser.add_argument(
        "--run-b",
        type=str,
        required=True,
        help="Run id B.",
    )
    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Run prompt variant sweep and compare pipeline scores.",
    )
    sweep_parser.add_argument(
        "--prompt-id",
        type=str,
        default=None,
        help="Sweep only this prompt. Omit to sweep all.",
    )

    sweep_parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Comma-separated domain ids. Omit to use enabled domain packs.",
    )

    annotate_parser = subparsers.add_parser(
        "annotate-human",
        help="Launch a local web GUI for scoring blinded human-annotation tasks.",
    )
    annotate_parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=None,
        help="Optional task directory. Defaults to artifacts/runs/_human_annotations.",
    )
    annotate_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface for the local server.",
    )
    annotate_parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for the local annotation server.",
    )
    annotate_parser.add_argument(
        "--annotator-id",
        type=str,
        default="",
        help="Optional default annotator ID pre-filled in the UI.",
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

    if path is not None:
        prompt_path = path
        if not prompt_path.exists():
            raise FileNotFoundError(f"Founder prompt file not found: {prompt_path}")
        data = json.loads(prompt_path.read_text(encoding="utf-8"))
        prompts = TypeAdapter(list[FounderPrompt]).validate_python(data)
        print(f"Validated {len(prompts)} founder prompts from {prompt_path}.")
        return 0

    packs = load_domain_packs(settings)
    prompts = load_founder_prompts(settings)
    enabled_domains = [pack.domain_id for pack in packs if pack.enabled]
    print(
        "Validated "
        f"{len(packs)} domain pack(s) from {settings.domain_packs_path} and "
        f"{len(prompts)} founder prompt(s) across {len(enabled_domains)} enabled domain(s)."
    )
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


def _load_prompts(prompt_id: str | None = None, domain_arg: str | None = None) -> list[FounderPrompt]:
    settings = get_settings()
    domain_ids = _parse_domain_ids(domain_arg)
    prompts = load_founder_prompts(settings, domain_ids=domain_ids)
    if prompt_id:
        prompts = [prompt for prompt in prompts if prompt.id == prompt_id]
        if not prompts:
            raise ValueError(f"No founder prompt found with id={prompt_id!r}")
    return prompts


def _parse_domain_ids(domains_arg: str | None) -> list[str] | None:
    if domains_arg is None:
        return None

    domain_ids = parse_domain_ids(domains_arg)
    if not domain_ids:
        raise ValueError("--domain was provided but no domain ids were parsed")
    return domain_ids


def _parse_variant_ids(variants_arg: str | None) -> list[str] | None:
    if variants_arg is None:
        return None
    variant_ids = [v.strip() for v in variants_arg.split(",") if v.strip()]
    if not variant_ids:
        raise ValueError("--variants was provided but no variant ids were parsed")
    return variant_ids

def _print_placeholder_contract_response(command: str, **payload: object) -> int:
    response = {
        "status": "foundation-placeholder",
        "command": command,
        "notes": (
            "This command surface is intentionally scaffolded in the foundation branch. "
            "Program-specific agent branches should replace this placeholder implementation."
        ),
        "payload": payload,
    }
    print(json.dumps(response, indent=2, sort_keys=True, default=str))
    return 0

def cmd_run_baseline(prompt_id: str | None, domain_arg: str | None) -> int:
    from .agents.baseline import run_baseline

    prompts = _load_prompts(prompt_id, domain_arg)
    for prompt in prompts:
        run_baseline(prompt)
    print(f"Baseline complete for {len(prompts)} prompt(s).")
    return 0


def cmd_run_pipeline(prompt_id: str | None, domain_arg: str | None) -> int:
    from .agents.pipeline import run_pipeline

    prompts = _load_prompts(prompt_id, domain_arg)
    for prompt in prompts:
        run_pipeline(prompt)
    print(f"Pipeline complete for {len(prompts)} prompt(s).")
    return 0


def cmd_list_variants(*, verbose: bool = False) -> int:
    from .agents.variant_pipeline import list_variant_specs

    specs = list_variant_specs()
    print("Available variants:")
    for spec in specs:
        print(f"- {spec.name}: {spec.description}")
        if verbose:
            for stage in spec.stages:
                deps = ", ".join(stage.depends_on) if stage.depends_on else "none"
                print(f"    - {stage.stage_id} (depends on: {deps})")
    return 0


def cmd_run_variant(variant: str, prompt_id: str | None, domain_arg: str | None) -> int:
    from .agents.variant_pipeline import run_variant_pipeline

    prompts = _load_prompts(prompt_id, domain_arg)
    for prompt in prompts:
        run_variant_pipeline(prompt, variant)
    print(f"Variant {variant!r} complete for {len(prompts)} prompt(s).")
    return 0


def cmd_evaluate(prompt_id: str | None, domain_arg: str | None, *, no_cache: bool) -> int:
    from .evaluation.runner import run_evaluation
    from .evaluation.report import generate_report
    from . import console as con

    settings = get_settings()
    prompt_ids = [prompt_id] if prompt_id else None
    domain_ids = _parse_domain_ids(domain_arg)
    summary = run_evaluation(
        settings,
        prompt_ids=prompt_ids,
        domain_ids=domain_ids,
        skip_cached=not no_cache,
    )
    report_path = generate_report(summary, settings.run_artifacts_dir)

    # Show aggregate scores table
    if summary.pairs:
        dims = [
            "relevance",
            "actionability",
            "evidence_grounding",
            "contradiction_handling",
            "non_redundancy",
        ]
        dim_labels = {
            "relevance": "Relevance",
            "actionability": "Actionability",
            "evidence_grounding": "Evidence Grounding",
            "contradiction_handling": "Contradiction Handling",
            "non_redundancy": "Non Redundancy",
        }
        scores: dict[str, tuple[float, float]] = {}
        for dim in dims:
            b_avg = sum(getattr(pair.baseline_scores.scores, dim) for pair in summary.pairs) / len(summary.pairs)
            p_avg = sum(getattr(pair.pipeline_scores.scores, dim) for pair in summary.pairs) / len(summary.pairs)
            scores[dim_labels[dim]] = (b_avg, p_avg)

        b_overall = sum(value[0] for value in scores.values()) / len(scores)
        p_overall = sum(value[1] for value in scores.values()) / len(scores)

        con.console.print()
        con.results_table(scores, b_overall, p_overall)

    con.success("report", f"Saved to {report_path}")
    return 0


def cmd_evaluate_variants(
    variants_arg: str | None,
    prompt_id: str | None,
    domain_arg: str | None,
    *,
    no_cache: bool,
) -> int:
    from .evaluation.variant_report import generate_variant_report
    from .evaluation.variant_runner import run_variant_evaluation
    from . import console as con
    from rich.table import Table

    settings = get_settings()
    prompt_ids = [prompt_id] if prompt_id else None
    variant_ids = _parse_variant_ids(variants_arg)
    domain_ids = _parse_domain_ids(domain_arg)

    summary = run_variant_evaluation(
        settings,
        variant_ids=variant_ids,
        prompt_ids=prompt_ids,
        domain_ids=domain_ids,
        skip_cached=not no_cache,
    )
    report_path = generate_variant_report(summary, settings.run_artifacts_dir.parent / "variant_runs")

    table = Table(title="Variant Results", show_lines=True)
    table.add_column("Variant", style="cyan bold")
    table.add_column("Control", justify="center")
    table.add_column("Variant", justify="center")
    table.add_column("Delta", justify="center")

    for aggregate in summary.aggregates:
        table.add_row(
            aggregate.variant,
            f"{aggregate.control_overall:.2f}",
            f"{aggregate.variant_overall:.2f}",
            f"{aggregate.delta_overall:+.2f}",
        )

    con.console.print()
    con.console.print(table)
    con.success("variant-report", f"Saved to {report_path}")
    return 0

def cmd_ingest(adapter: str, input_path: Path | None) -> int:
    return _print_placeholder_contract_response(
        "ingest",
        adapter=adapter,
        input_path=input_path,
    )


def cmd_snapshot_data(dataset_id: str) -> int:
    return _print_placeholder_contract_response(
        "snapshot-data",
        dataset_id=dataset_id,
    )


def cmd_eval_retrieval(label_set: Path | None) -> int:
    return _print_placeholder_contract_response(
        "eval-retrieval",
        label_set=label_set,
    )


def cmd_eval_robustness(suite: str) -> int:
    return _print_placeholder_contract_response(
        "eval-robustness",
        suite=suite,
    )


def cmd_compare_runs(run_a: str, run_b: str) -> int:
    return _print_placeholder_contract_response(
        "compare-runs",
        run_a=run_a,
        run_b=run_b,
    )


def cmd_sweep(prompt_id: str | None, domain_arg: str | None) -> int:
    from .evaluation.prompt_sweep import run_sweep
    from . import console as con
    from rich.table import Table

    settings = get_settings()
    prompt_ids = [prompt_id] if prompt_id else None
    domain_ids = _parse_domain_ids(domain_arg)

    con.header(
        "Prompt Variant Sweep",
        f"model: {settings.llm_model} | provider: {settings.llm_provider}",
    )

    results = run_sweep(settings, prompt_ids=prompt_ids, domain_ids=domain_ids)

    # Build comparison table
    table = Table(title="Sweep Results", show_lines=True)
    table.add_column("Variant", style="cyan bold")
    table.add_column("Description", style="dim")
    table.add_column("Rel", justify="center")
    table.add_column("Act", justify="center")
    table.add_column("Evi", justify="center")
    table.add_column("Con", justify="center")
    table.add_column("NR", justify="center")
    table.add_column("Overall", justify="center", style="bold")

    for r in results:
        table.add_row(
            r.variant,
            r.description,
            f"{r.scores.get('relevance', 0):.1f}",
            f"{r.scores.get('actionability', 0):.1f}",
            f"{r.scores.get('evidence_grounding', 0):.1f}",
            f"{r.scores.get('contradiction_handling', 0):.1f}",
            f"{r.scores.get('non_redundancy', 0):.1f}",
            f"{r.overall:.2f}",
        )

    con.console.print()
    con.console.print(table)
    return 0


def cmd_annotate_human(
    tasks_dir: Path | None,
    *,
    host: str,
    port: int,
    annotator_id: str,
) -> int:
    from .evaluation.human_annotation_gui import run_annotation_server

    settings = get_settings()
    target_dir = tasks_dir or (settings.run_artifacts_dir / "_human_annotations")
    run_annotation_server(
        target_dir,
        host=host,
        port=port,
        default_annotator_id=annotator_id,
    )
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
        return cmd_run_baseline(args.prompt_id, args.domain)
    if args.command == "run-pipeline":
        return cmd_run_pipeline(args.prompt_id, args.domain)
    if args.command == "list-variants":
        return cmd_list_variants(verbose=args.verbose)
    if args.command == "run-variant":
        return cmd_run_variant(args.variant, args.prompt_id, args.domain)
    if args.command == "evaluate":
        return cmd_evaluate(args.prompt_id, args.domain, no_cache=args.no_cache)
    if args.command == "evaluate-variants":
        return cmd_evaluate_variants(args.variants, args.prompt_id, args.domain, no_cache=args.no_cache)
    if args.command == "ingest":
        return cmd_ingest(args.adapter, args.input_path)
    if args.command == "snapshot-data":
        return cmd_snapshot_data(args.dataset_id)
    if args.command == "eval-retrieval":
        return cmd_eval_retrieval(args.label_set)
    if args.command == "eval-robustness":
        return cmd_eval_robustness(args.suite)
    if args.command == "compare-runs":
        return cmd_compare_runs(args.run_a, args.run_b)

    if args.command == "sweep":
        return cmd_sweep(args.prompt_id, args.domain)

    if args.command == "annotate-human":
        return cmd_annotate_human(
            args.tasks_dir,
            host=args.host,
            port=args.port,
            annotator_id=args.annotator_id,
        )

    parser.error(f"Unknown command: {args.command}")
    return 2











