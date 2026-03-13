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
from .data.ingestion import build_snapshot, list_adapter_ids, run_ingest
from .domain_packs import load_domain_packs, load_founder_prompts, parse_domain_ids
from .retrieval.index import search_retrieval_index
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
        help="Query the retrieval stack and print top-K snippets.",
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
    evaluate_parser.add_argument(
        "--judge-panel-size",
        type=int,
        default=None,
        help="Optional deterministic judge panel size (must be >= 1).",
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
        help="Run adapter-based source ingestion and normalize into DatasetRecord JSONL.",
    )
    ingest_parser.add_argument(
        "--adapter",
        type=str,
        choices=list_adapter_ids(),
        default="yelp",
        help="Adapter id to run.",
    )
    ingest_parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Optional path consumed by the adapter.",
    )

    snapshot_parser = subparsers.add_parser(
        "snapshot-data",
        help="Create an immutable DatasetSnapshotManifest from ingested datasets.",
    )
    snapshot_parser.add_argument(
        "--dataset-id",
        type=str,
        default="default",
        help="Logical dataset id included in the snapshot manifest.",
    )

    eval_retrieval_parser = subparsers.add_parser(
        "eval-retrieval",
        help="Evaluate retrieval quality metrics and generate reports.",
    )
    eval_retrieval_parser.add_argument(
        "--label-set",
        type=Path,
        default=None,
        help="Optional labeled query-set JSONL path for retrieval metrics.",
    )
    eval_retrieval_parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=("dense", "lexical", "hybrid"),
        help="Override retrieval mode for this run.",
    )
    eval_retrieval_parser.add_argument(
        "--reranker",
        type=str,
        default=None,
        choices=("none", "token_overlap"),
        help="Override reranker stage for this run.",
    )
    eval_retrieval_parser.add_argument(
        "--k-values",
        type=str,
        default="1,3,5,10",
        help="Comma-separated K values used for Recall@K, MRR@K, and nDCG@K.",
    )
    eval_retrieval_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override retrieval top_k for this run (must be >= max K value).",
    )
    eval_retrieval_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for retrieval evaluation artifacts.",
    )

    eval_robustness_parser = subparsers.add_parser(
        "eval-robustness",
        help="Run robustness stress-case evaluation and release gates.",
    )
    eval_robustness_parser.add_argument(
        "--suite",
        type=str,
        default="default",
        help="Robustness suite id.",
    )
    eval_robustness_parser.add_argument(
        "--prompt-id",
        type=str,
        default=None,
        help="Evaluate only this prompt. Omit for staged prompt subset.",
    )
    eval_robustness_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Re-run robustness suites even if cached artifacts exist.",
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
    integration_gate_parser = subparsers.add_parser(
        "integration-gate",
        help="Run integration-ready report gates for research-upgrade branches.",
    )
    integration_gate_parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/research_upgrade"),
        help="Directory containing report JSON inputs.",
    )
    integration_gate_parser.add_argument(
        "--schema-report",
        type=Path,
        default=None,
        help="Override path for schema_compatibility.json.",
    )
    integration_gate_parser.add_argument(
        "--retrieval-report",
        type=Path,
        default=None,
        help="Override path for retrieval_eval_summary.json.",
    )
    integration_gate_parser.add_argument(
        "--robustness-report",
        type=Path,
        default=None,
        help="Override path for robustness_report.json.",
    )
    integration_gate_parser.add_argument(
        "--domain-transfer-report",
        type=Path,
        default=None,
        help="Override path for domain_transfer_report.json.",
    )
    integration_gate_parser.add_argument(
        "--failure-tags-report",
        type=Path,
        default=None,
        help="Override path for failure_tags_report.json.",
    )
    integration_gate_parser.add_argument(
        "--severity-threshold",
        type=int,
        default=4,
        help="Block when a failure tag severity is at or above this threshold.",
    )
    integration_gate_parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional file path where the gate summary JSON will also be written.",
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
            f"Retrieval index not found at {index_dir}. "
            "Run `uv run python scripts/build_index.py` first."
        )
    hits = search_retrieval_index(
        query,
        index_dir=index_dir,
        embedding_model=settings.embedding_model,
        top_k=top_k,
        mode=settings.retrieval_mode,
        dense_weight=settings.retrieval_dense_weight,
        lexical_weight=settings.retrieval_lexical_weight,
        fusion_k=settings.retrieval_fusion_k,
        candidate_pool=settings.retrieval_candidate_pool,
        reranker=settings.retrieval_reranker,
        reranker_weight=settings.retrieval_reranker_weight,
        bm25_k1=settings.retrieval_bm25_k1,
        bm25_b=settings.retrieval_bm25_b,
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

def _parse_k_values(k_values_arg: str) -> list[int]:
    parsed = sorted({int(v.strip()) for v in k_values_arg.split(",") if v.strip()})
    if not parsed:
        raise ValueError("--k-values did not include any integers")
    if any(v <= 0 for v in parsed):
        raise ValueError("--k-values must be positive integers")
    return parsed

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


def cmd_evaluate(
    prompt_id: str | None,
    domain_arg: str | None = None,
    *,
    no_cache: bool = False,
    judge_panel_size: int | None = None,
) -> int:
    from .evaluation.failure_taxonomy import generate_failure_taxonomy
    from .evaluation.report import generate_report
    from .evaluation.runner import run_evaluation
    from . import console as con

    base_settings = get_settings()
    if judge_panel_size is not None and judge_panel_size < 1:
        raise ValueError("--judge-panel-size must be >= 1")
    settings = (
        base_settings.model_copy(update={"judge_panel_size": judge_panel_size})
        if judge_panel_size is not None
        else base_settings
    )
    prompt_ids = [prompt_id] if prompt_id else None
    domain_ids = _parse_domain_ids(domain_arg)
    summary = run_evaluation(
        settings,
        prompt_ids=prompt_ids,
        domain_ids=domain_ids,
        skip_cached=not no_cache,
    )
    report_path = generate_report(summary, settings.run_artifacts_dir)
    failure_tags, failure_tags_path, failure_report_path = generate_failure_taxonomy(
        settings.run_artifacts_dir,
        prompt_ids=prompt_ids,
    )

    # Show aggregate scores table
    if summary.pairs:
        dims = [
            "relevance",
            "groundedness",
            "distinctiveness",
        ]
        dim_labels = {
            "relevance": "Relevance",
            "groundedness": "Groundedness",
            "distinctiveness": "Distinctiveness",
        }
        scores: dict[str, tuple[float, float]] = {}
        for dim in dims:
            b_avg = sum(getattr(pair.baseline_scores.scores, dim) for pair in summary.pairs) / len(summary.pairs)
            p_avg = sum(getattr(pair.pipeline_scores.scores, dim) for pair in summary.pairs) / len(summary.pairs)
            scores[dim_labels[dim]] = (b_avg, p_avg)

        b_overall = sum(pair.baseline_scores.scores.overall_preference for pair in summary.pairs) / len(summary.pairs)
        p_overall = sum(pair.pipeline_scores.scores.overall_preference for pair in summary.pairs) / len(summary.pairs)

        con.console.print()
        con.results_table(scores, b_overall, p_overall)

    con.success("report", f"Saved to {report_path}")
    con.success("failure-tags", f"Saved {len(failure_tags)} tag(s) to {failure_tags_path}")
    con.success("root-cause-report", f"Saved to {failure_report_path}")
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
    settings = get_settings()
    result = run_ingest(
        settings=settings,
        adapter_id=adapter,
        input_path=input_path,
    )
    response = {
        "status": "ok",
        "command": "ingest",
        "payload": {
            "adapter": result.adapter_id,
            "dataset_id": result.dataset_id,
            "record_count": result.record_count,
            "records_path": str(result.records_path),
            "manifest_path": str(result.manifest_path),
            "checksum_sha256": result.checksum_sha256,
            "source_manifests": result.source_manifests,
        },
    }
    print(json.dumps(response, indent=2, sort_keys=True, default=str))
    return 0


def cmd_snapshot_data(dataset_id: str) -> int:
    settings = get_settings()
    result = build_snapshot(
        settings=settings,
        dataset_id=dataset_id,
    )
    response = {
        "status": "ok",
        "command": "snapshot-data",
        "payload": {
            "snapshot_id": result.manifest.snapshot_id,
            "dataset_ids": result.manifest.dataset_ids,
            "record_count": result.manifest.record_count,
            "checksum_sha256": result.manifest.checksum_sha256,
            "source_manifests": result.manifest.source_manifests,
            "manifest_path": str(result.manifest_path),
        },
    }
    print(json.dumps(response, indent=2, sort_keys=True, default=str))
    return 0


def cmd_eval_retrieval(
    label_set: Path | None,
    *,
    mode: str | None = None,
    reranker: str | None = None,
    k_values: str = "1,3,5,10",
    top_k: int | None = None,
    output_dir: Path | None = None,
) -> int:
    from rich.table import Table

    from . import console as con
    from .evaluation.retrieval_report import generate_retrieval_report
    from .evaluation.retrieval_runner import run_retrieval_evaluation

    settings = get_settings()
    target_label_set = label_set or (settings.run_artifacts_dir.parent / "retrieval_labels.jsonl")
    if not target_label_set.exists():
        raise FileNotFoundError(
            f"Retrieval label set not found: {target_label_set}. "
            "Provide --label-set or place retrieval_labels.jsonl under artifacts/."
        )

    parsed_k_values = _parse_k_values(k_values)
    summary = run_retrieval_evaluation(
        target_label_set,
        settings,
        mode=mode,
        reranker=reranker,
        top_k=top_k,
        k_values=parsed_k_values,
    )

    destination = output_dir or (settings.run_artifacts_dir.parent / "retrieval_eval")
    json_path, markdown_path = generate_retrieval_report(summary, destination)

    table = Table(title="Retrieval Metrics", show_lines=False)
    table.add_column("Metric", style="cyan bold")
    for k in summary.k_values:
        table.add_column(f"@{k}", justify="center")

    table.add_row(
        "Recall",
        *[f"{summary.aggregates['recall_at_k'][str(k)]:.4f}" for k in summary.k_values],
    )
    table.add_row(
        "MRR",
        *[f"{summary.aggregates['mrr_at_k'][str(k)]:.4f}" for k in summary.k_values],
    )
    table.add_row(
        "nDCG",
        *[f"{summary.aggregates['ndcg_at_k'][str(k)]:.4f}" for k in summary.k_values],
    )

    con.console.print()
    con.console.print(table)
    con.success("eval-retrieval", f"JSON summary -> {json_path}")
    con.success("eval-retrieval", f"Markdown report -> {markdown_path}")
    return 0

def cmd_eval_robustness(
    suite: str,
    prompt_id: str | None = None,
    *,
    no_cache: bool = False,
) -> int:
    from .evaluation.robustness_report import generate_robustness_report
    from .evaluation.robustness_runner import run_robustness_suite, suite_output_dir
    from . import console as con
    from rich.table import Table

    settings = get_settings()
    prompt_ids = [prompt_id] if prompt_id else None

    summary = run_robustness_suite(
        settings,
        suite_id=suite,
        prompt_ids=prompt_ids,
        skip_cached=not no_cache,
    )
    output_dir = suite_output_dir(settings, summary.suite_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "robustness_summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    report_path = generate_robustness_report(summary, output_dir)

    table = Table(title=f"Robustness Results ({summary.suite_id})", show_lines=True)
    table.add_column("Prompt", style="cyan bold")
    table.add_column("Case")
    table.add_column("Family")
    table.add_column("Delta", justify="center")
    table.add_column("Status", justify="center")

    for outcome in summary.outcomes:
        status = "[green]PASS[/green]" if outcome.passed else "[red]FAIL[/red]"
        table.add_row(
            outcome.prompt_id,
            outcome.case_id,
            outcome.family,
            f"{outcome.delta_overall:+.2f}",
            status,
        )

    con.console.print()
    con.console.print(table)
    con.success("robustness", f"Summary JSON: {summary_path}")
    con.success("robustness", f"Report: {report_path}")

    if summary.gate_passed:
        con.success(
            "robustness",
            f"Gate passed ({summary.passed_cases}/{summary.total_cases}, {summary.pass_rate:.2%})",
        )
        return 0

    for reason in summary.gate_failure_reasons:
        con.error(reason)
    return 2


def cmd_compare_runs(run_a: str, run_b: str) -> int:
    return _print_placeholder_contract_response(
        "compare-runs",
        run_a=run_a,
        run_b=run_b,
    )


def cmd_integration_gate(
    reports_dir: Path,
    *,
    schema_report: Path | None,
    retrieval_report: Path | None,
    robustness_report: Path | None,
    domain_transfer_report: Path | None,
    failure_tags_report: Path | None,
    severity_threshold: int,
    summary_out: Path | None,
) -> int:
    from .integration.gates import default_gate_inputs, run_integration_gates

    inputs = default_gate_inputs(reports_dir, high_severity_threshold=severity_threshold)
    if schema_report is not None:
        inputs.schema_compatibility_path = schema_report
    if retrieval_report is not None:
        inputs.retrieval_report_path = retrieval_report
    if robustness_report is not None:
        inputs.robustness_report_path = robustness_report
    if domain_transfer_report is not None:
        inputs.domain_transfer_report_path = domain_transfer_report
    if failure_tags_report is not None:
        inputs.failure_tags_report_path = failure_tags_report

    summary = run_integration_gates(inputs)
    payload = summary.model_dump(mode="json")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if summary_out is not None:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    return 0 if summary.overall_status == "pass" else 1


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
    table.add_column("Ground", justify="center")
    table.add_column("Dist", justify="center")
    table.add_column("Overall", justify="center", style="bold")

    for r in results:
        table.add_row(
            r.variant,
            r.description,
            f"{r.scores.get('relevance', 0):.1f}",
            f"{r.scores.get('groundedness', 0):.1f}",
            f"{r.scores.get('distinctiveness', 0):.1f}",
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
        return cmd_evaluate(args.prompt_id, args.domain, no_cache=args.no_cache, judge_panel_size=args.judge_panel_size)
    if args.command == "evaluate-variants":
        return cmd_evaluate_variants(args.variants, args.prompt_id, args.domain, no_cache=args.no_cache)
    if args.command == "ingest":
        return cmd_ingest(args.adapter, args.input_path)
    if args.command == "snapshot-data":
        return cmd_snapshot_data(args.dataset_id)
    if args.command == "eval-retrieval":
        return cmd_eval_retrieval(
            args.label_set,
            mode=args.mode,
            reranker=args.reranker,
            k_values=args.k_values,
            top_k=args.top_k,
            output_dir=args.output_dir,
        )
    if args.command == "eval-robustness":
        return cmd_eval_robustness(args.suite, args.prompt_id, no_cache=args.no_cache)
    if args.command == "compare-runs":
        return cmd_compare_runs(args.run_a, args.run_b)

    if args.command == "integration-gate":
        return cmd_integration_gate(
            reports_dir=args.reports_dir,
            schema_report=args.schema_report,
            retrieval_report=args.retrieval_report,
            robustness_report=args.robustness_report,
            domain_transfer_report=args.domain_transfer_report,
            failure_tags_report=args.failure_tags_report,
            severity_threshold=args.severity_threshold,
            summary_out=args.summary_out,
        )

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

