"""Generate publication-quality PDF figures for paper/figures/ from analysis data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SUMMARY_PATH = Path("reports/final_analysis/final_analysis_summary.json")
OUTPUT_DIR = Path("paper/figures")

COLORS = {
    "baseline": "#c06060",
    "pipeline": "#2a9d8f",
    "positive": "#2a9d8f",
    "negative": "#c06060",
    "restaurants": "#4a90c4",
    "ecommerce": "#e9a03b",
    "saas": "#7b68c4",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def load_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def fig_overall_scores(data: dict) -> None:
    metrics = ["Relevance", "Groundedness", "Distinctiveness", "Overall\nPreference"]
    keys = ["relevance", "groundedness", "distinctiveness", "overall_preference"]
    baseline = [data["baseline"][k] for k in keys]
    pipeline = [data["pipeline"][k] for k in keys]

    x = np.arange(len(metrics))
    w = 0.32

    fig, ax = plt.subplots(figsize=(7, 4))
    bars_b = ax.bar(x - w / 2, baseline, w, label="Baseline", color=COLORS["baseline"],
                    edgecolor="white", linewidth=0.5, zorder=3)
    bars_p = ax.bar(x + w / 2, pipeline, w, label="Pipeline", color=COLORS["pipeline"],
                    edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_ylabel("Mean Score (1–5)")
    ax.set_title("Average Scores Across All Prompts", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 5.5)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    for bars in (bars_b, bars_p):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "overall_scores.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_domain_deltas(data: dict) -> None:
    domains = data["domains"]
    names = [d["domain"].title() for d in domains]
    deltas = [d["delta"]["overall_preference"] for d in domains]
    colors = [COLORS.get(d["domain"], COLORS["pipeline"]) for d in domains]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(names, deltas, color=colors, edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_ylabel("Overall Preference Δ\n(Pipeline − Baseline)")
    ax.set_title("Domain-Level Pipeline Gain", fontweight="bold", pad=12)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    for bar, val in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"+{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "domain_deltas.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_prompt_deltas(data: dict) -> None:
    outcomes = sorted(data["prompt_outcomes"],
                      key=lambda r: r["delta"]["overall_preference"])

    labels = [o["prompt_id"] for o in outcomes]
    deltas = [o["delta"]["overall_preference"] for o in outcomes]
    colors = [COLORS["positive"] if d >= 0 else COLORS["negative"] for d in deltas]

    fig, ax = plt.subplots(figsize=(7, 7))
    y = np.arange(len(labels))
    bars = ax.barh(y, deltas, color=colors, edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Overall Preference Δ (Pipeline − Baseline)")
    ax.set_title("Per-Prompt Pipeline Gain (Sorted)", fontweight="bold", pad=12)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3, zorder=0)

    for bar, val in zip(bars, deltas):
        offset = 0.1 if val >= 0 else -0.1
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}", ha=ha, va="center", fontsize=8)

    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, fc=COLORS["positive"], label="Pipeline preferred"),
            plt.Rectangle((0, 0), 1, 1, fc=COLORS["negative"], label="Baseline preferred"),
        ],
        frameon=False, loc="lower right", fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "prompt_deltas.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_failure_taxonomy(data: dict) -> None:
    categories = data["failure_categories"]
    if not categories:
        return

    names = [c["category"].replace("_", " ").title() for c in categories]
    baseline_counts = [c["baseline_count"] for c in categories]
    pipeline_counts = [c["pipeline_count"] for c in categories]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(x + w / 2, baseline_counts, w, label="Baseline",
            color=COLORS["baseline"], edgecolor="white", linewidth=0.5, zorder=3)
    ax.barh(x - w / 2, pipeline_counts, w, label="Pipeline",
            color=COLORS["pipeline"], edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Failure Tag Count")
    ax.set_title("Failure Tags by Category and System", fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "failure_taxonomy.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_domain_rubric_heatmap(data: dict) -> None:
    """Per-domain deltas across all 4 rubric dimensions as a heatmap."""
    domains = data["domains"]
    if not domains:
        return

    dim_keys = ["relevance", "groundedness", "distinctiveness", "overall_preference"]
    dim_labels = ["Relevance", "Groundedness", "Distinctiveness", "Overall Pref."]
    domain_labels = [d["domain"].title() for d in domains]

    matrix = np.array([[d["delta"][k] for k in dim_keys] for d in domains])

    fig, ax = plt.subplots(figsize=(6, 3))
    vmax = max(abs(matrix.min()), abs(matrix.max()), 0.5)
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(dim_labels)))
    ax.set_xticklabels(dim_labels, fontsize=10)
    ax.set_yticks(np.arange(len(domain_labels)))
    ax.set_yticklabels(domain_labels, fontsize=10)
    ax.set_title("Pipeline Δ by Domain and Rubric Dimension", fontweight="bold", pad=12)

    for i in range(len(domain_labels)):
        for j in range(len(dim_labels)):
            val = matrix[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
    cbar.set_label("Score Δ (Pipeline − Baseline)", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "domain_rubric_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_judge_agreement(data: dict) -> None:
    """Human vs LLM judge agreement confusion matrix from annotation summary."""
    annotation_path = Path("reports/human_annotation/analysis/human_annotation_summary.json")
    if not annotation_path.exists():
        return

    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    alignments = annotation.get("judge_alignment", [])
    if not alignments:
        return

    align = alignments[0]
    n = align["sample_size"]
    exact = align["exact_agreement"]
    kappa = align["cohen_kappa"]
    h_counts = align["human_preference_counts"]
    j_counts = align["judge_preference_counts"]

    # Reconstruct the 2x2 confusion matrix (no ties in this dataset).
    # Labels: Pipeline preferred, Baseline preferred (mapped from system_a/b).
    # From the summary: human and judge each picked the same label distribution.
    # exact_agreement = 0.70 => 14 agree out of 20.
    # h_a=7, h_b=13, j_a=7, j_b=13 => agree(a,a) + agree(b,b) = 14.
    # Solving: aa + ab = 7 (human_a), aa + ba = 7 (judge_a),
    #          ba + bb = 13 (human_b), ab + bb = 13 (judge_b),
    #          aa + bb = 14 (agree).
    # => aa = 7 - ab, bb = 13 - ab, aa + bb = 20 - 2*ab = 14 => ab = 3
    # => aa = 4, ab = 3, ba = 3, bb = 10.
    h_a = h_counts.get("system_a", 0)
    h_b = h_counts.get("system_b", 0)
    j_a = j_counts.get("system_a", 0)

    agree_count = round(exact * n)
    # ab = human picked A but judge picked B
    ab = h_a - (agree_count - (n - h_a - (h_a - j_a)))  # simplify:
    # aa + bb = agree_count, aa + ab = h_a, aa + ba = j_a
    # => aa = h_a - ab, ba = j_a - aa = j_a - h_a + ab
    # => bb = n - aa - ab - ba = n - (h_a - ab) - ab - (j_a - h_a + ab) = n - j_a - ab
    # => aa + bb = (h_a - ab) + (n - j_a - ab) = h_a + n - j_a - 2*ab = agree_count
    # => ab = (h_a + n - j_a - agree_count) / 2
    ab = (h_a + n - j_a - agree_count) / 2
    aa = h_a - ab
    ba = j_a - aa
    bb = n - aa - ab - ba

    cm = np.array([[aa, ab], [ba, bb]])
    labels_display = ["Pipeline\npreferred", "Baseline\npreferred"]

    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    vmax = max(cm.max(), 1)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)

    for i in range(2):
        for j in range(2):
            val = int(cm[i, j])
            color = "white" if val > vmax * 0.6 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=16, fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels_display, fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels_display, fontsize=10)
    ax.set_xlabel("LLM Judge", fontweight="bold", labelpad=10)
    ax.set_ylabel("Human Annotator", fontweight="bold")
    ax.set_title("Human vs. Judge Agreement", fontweight="bold", pad=12)

    summary_text = f"n={n}  |  Exact={exact:.0%}  |  κ={kappa:.2f}"
    fig.text(0.5, 0.02, summary_text, ha="center", va="bottom",
             fontsize=9, style="italic", color="#555555")

    fig.subplots_adjust(bottom=0.22)
    fig.savefig(OUTPUT_DIR / "judge_agreement.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_summary()

    fig_overall_scores(data)
    fig_domain_deltas(data)
    fig_prompt_deltas(data)
    fig_failure_taxonomy(data)
    fig_domain_rubric_heatmap(data)
    fig_judge_agreement(data)

    count = len(list(OUTPUT_DIR.glob("*.pdf")))
    print(f"Generated {count} figures in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
