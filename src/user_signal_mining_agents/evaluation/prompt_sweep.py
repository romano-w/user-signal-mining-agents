"""Prompt sweep: run evaluation across multiple prompt variants and compare."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings, get_settings
from .. import console as con
from .runner import run_evaluation


@dataclass
class SweepVariant:
    """A named variant with prompt file overrides."""

    name: str
    description: str
    overrides: dict[str, str | None]  # filename -> content (None = patched later)


@dataclass
class SweepResult:
    """Scores for a single variant."""

    variant: str
    description: str
    scores: dict[str, float]  # dimension -> avg score
    overall: float = 0.0


# Default sweep variants - override specific prompt files
SWEEP_VARIANTS: list[SweepVariant] = [
    SweepVariant(
        name="control",
        description="Current prompts (no changes)",
        overrides={},
    ),
    SweepVariant(
        name="evidence-budget",
        description="Each snippet may only appear in one focus point",
        overrides={
            "synthesis.md": None,  # Marker - patched dynamically below
        },
    ),
    SweepVariant(
        name="strict-quoting",
        description="Only direct quotes, never paraphrase",
        overrides={
            "synthesis.md": None,
        },
    ),
    SweepVariant(
        name="fewer-points",
        description="Exactly 3 focus points instead of 3-5",
        overrides={
            "synthesis.md": None,
        },
    ),
]


def _build_variant_prompts(base_dir: Path) -> dict[str, list[SweepVariant]]:
    """Build actual prompt content for each variant by patching the base prompts."""

    base_synthesis = (base_dir / "synthesis.md").read_text(encoding="utf-8")

    # evidence-budget: add one-snippet-per-focus-point rule
    budget_synthesis = base_synthesis.replace(
        "- Prefer distinct, non-overlapping focus points.",
        "- Prefer distinct, non-overlapping focus points.\n"
        "- **EVIDENCE BUDGET**: Each evidence snippet may be cited in at most ONE focus point. "
        "Do NOT reuse the same quote or restaurant example across multiple focus points.",
    )

    # strict-quoting: require exact quotes only
    strict_synthesis = base_synthesis.replace(
        "directly quote or closely paraphrase 2-3 specific customer statements",
        "use EXACT DIRECT QUOTES (not paraphrases) of 2-3 specific customer statements",
    )

    # fewer-points: cap at exactly 3
    fewer_synthesis = base_synthesis.replace(
        "- Produce exactly 3-5 focus points.",
        "- Produce exactly 3 focus points. No more, no less.",
    )

    for variant in SWEEP_VARIANTS:
        if variant.name == "evidence-budget":
            variant.overrides["synthesis.md"] = budget_synthesis
        elif variant.name == "strict-quoting":
            variant.overrides["synthesis.md"] = strict_synthesis
        elif variant.name == "fewer-points":
            variant.overrides["synthesis.md"] = fewer_synthesis

    return {}


def run_sweep(
    settings: Settings | None = None,
    *,
    variants: list[SweepVariant] | None = None,
    prompt_ids: list[str] | None = None,
    domain_ids: list[str] | None = None,
) -> list[SweepResult]:
    """Run evaluation across multiple prompt variants."""

    s = settings or get_settings()
    variants = variants or SWEEP_VARIANTS
    prompts_dir = s.prompts_dir
    runs_dir = s.run_artifacts_dir

    # Build variant prompt content
    _build_variant_prompts(prompts_dir)

    # Back up original prompts
    backup_dir = prompts_dir.parent / "prompts_backup"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(prompts_dir, backup_dir)

    results: list[SweepResult] = []

    try:
        for vi, variant in enumerate(variants, start=1):
            con.console.rule(f"[bold cyan]Sweep {vi}/{len(variants)}: {variant.name}[/]")
            con.step("sweep", variant.description)

            # Apply overrides
            for filename, content in variant.overrides.items():
                if content is not None:
                    (prompts_dir / filename).write_text(content, encoding="utf-8")

            # Use variant-specific run dir
            variant_runs = runs_dir.parent / "sweep_runs" / variant.name
            variant_runs.mkdir(parents=True, exist_ok=True)
            variant_settings = s.model_copy(update={"run_artifacts_dir": variant_runs})

            # Run evaluation (no cache - we want fresh results for each variant)
            summary = run_evaluation(
                variant_settings,
                prompt_ids=prompt_ids,
                domain_ids=domain_ids,
                skip_cached=False,
            )

            # Collect scores
            dims = [
                "relevance",
                "groundedness",
                "distinctiveness",
            ]
            dim_scores: dict[str, float] = {}
            for dim in dims:
                avg = sum(getattr(pair.pipeline_scores.scores, dim) for pair in summary.pairs) / max(
                    len(summary.pairs),
                    1,
                )
                dim_scores[dim] = avg

            overall = sum(pair.pipeline_scores.scores.overall_preference for pair in summary.pairs) / max(
                len(summary.pairs),
                1,
            )
            results.append(SweepResult(
                variant=variant.name,
                description=variant.description,
                scores=dim_scores,
                overall=overall,
            ))

            # Restore original prompts for next variant
            for filename in variant.overrides:
                original = backup_dir / filename
                if original.exists():
                    shutil.copy2(original, prompts_dir / filename)

    finally:
        # Always restore original prompts
        for source in backup_dir.iterdir():
            shutil.copy2(source, prompts_dir / source.name)
        shutil.rmtree(backup_dir)

    return results

