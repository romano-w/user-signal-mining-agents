"""Pipeline orchestrator: intent → evidence filter → synthesis → verify."""

from __future__ import annotations

from ..config import Settings, get_settings
from ..schemas import FounderPrompt, SynthesisResult
from .. import console as con
from .evidence_filter import retrieve_and_filter
from .evidence_verifier import verify_evidence
from .intent import decompose_intent
from .synthesis import run_synthesis


def run_pipeline(
    prompt: FounderPrompt,
    settings: Settings | None = None,
) -> SynthesisResult:
    """Run the full multi-step grounded pipeline for a single founder prompt."""

    s = settings or get_settings()

    # Step 1 — Intent decomposition
    intent = decompose_intent(prompt, s)
    con.step("pipeline", f"Intent: {len(intent.retrieval_queries)} queries, "
          f"{len(intent.counter_hypotheses)} counter-hypotheses")

    # Step 2 — Multi-query retrieval + dedup + re-rank
    evidence = retrieve_and_filter(prompt, intent, s)

    # Step 3 — Grounded synthesis
    result = run_synthesis(prompt, intent, evidence, s)

    # Step 4 — Evidence-grounding verification
    result = verify_evidence(result, evidence, s)

    # Persist
    run_dir = s.run_artifacts_dir / prompt.id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "pipeline.json"
    output_path.write_text(
        result.model_dump_json(indent=2, exclude_none=True),
        encoding="utf-8",
    )
    con.success("pipeline", f"Saved -> {output_path}")
    return result
