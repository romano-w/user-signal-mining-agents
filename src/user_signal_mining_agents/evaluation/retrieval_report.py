"""Report generator for retrieval evaluation outputs."""

from __future__ import annotations

from pathlib import Path

from .retrieval_runner import RetrievalEvaluationSummary



def generate_retrieval_report(summary: RetrievalEvaluationSummary, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "retrieval_eval_summary.json"
    markdown_path = output_dir / "retrieval_eval_report.md"

    json_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    lines: list[str] = [
        "# Retrieval Evaluation Report",
        "",
        f"**Generated at:** {summary.generated_at.isoformat()}",
        f"**Label set:** `{summary.label_set_path}`",
        f"**Queries evaluated:** {summary.query_count}",
        f"**Mode:** `{summary.retrieval_mode}`",
        f"**Reranker:** `{summary.reranker}`",
        f"**Top K retrieved:** {summary.top_k}",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | " + " | ".join(f"@{k}" for k in summary.k_values) + " |",
        "|---|" + "|".join(":---:" for _ in summary.k_values) + "|",
    ]

    for metric_name in ("recall_at_k", "mrr_at_k", "ndcg_at_k"):
        row_values = [summary.aggregates[metric_name][str(k)] for k in summary.k_values]
        lines.append(
            "| " + metric_name.replace("_", " ").upper() + " | " + " | ".join(f"{v:.4f}" for v in row_values) + " |"
        )

    lines.extend([
        "",
        "## Per-Query Metrics",
        "",
    ])

    for query in summary.queries:
        lines.append(f"### `{query.query_id}`")
        lines.append(f"> {query.query}")
        lines.append("")
        lines.append("| Metric | " + " | ".join(f"@{k}" for k in summary.k_values) + " |")
        lines.append("|---|" + "|".join(":---:" for _ in summary.k_values) + "|")
        lines.append(
            "| Recall | " + " | ".join(f"{query.recall_at_k[str(k)]:.4f}" for k in summary.k_values) + " |"
        )
        lines.append(
            "| MRR | " + " | ".join(f"{query.mrr_at_k[str(k)]:.4f}" for k in summary.k_values) + " |"
        )
        lines.append(
            "| nDCG | " + " | ".join(f"{query.ndcg_at_k[str(k)]:.4f}" for k in summary.k_values) + " |"
        )
        preview = ", ".join(query.retrieved_snippet_ids[:5])
        lines.append(f"**Top retrieved snippet IDs:** {preview if preview else 'none'}")
        lines.append("")

    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path
