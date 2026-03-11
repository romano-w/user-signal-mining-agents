"""Configurable variant runner for multi-agent pipeline experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..config import Settings, get_settings
from ..schemas import EvidenceSnippet, FounderPrompt, IntentBundle, SynthesisResult
from .. import console as con
from .counterevidence_miner import mine_counterevidence_queries
from .critic import critique_focus_points
from .evidence_filter import retrieve_for_queries
from .evidence_verifier import verify_evidence
from .intent import decompose_intent
from .query_planner import plan_retrieval_queries
from .refiner import refine_focus_points
from .synthesis import run_synthesis


StageHandler = Callable[["PipelineContext", Settings], None]


@dataclass
class PipelineContext:
    prompt: FounderPrompt
    intent_bundle: IntentBundle | None = None
    retrieval_queries: list[str] = field(default_factory=list)
    retrieved_evidence: list[EvidenceSnippet] = field(default_factory=list)
    result: SynthesisResult | None = None
    critic_feedback: list[str] = field(default_factory=list)
    stage_trace: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PipelineStage:
    stage_id: str
    handler: StageHandler
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineVariantSpec:
    name: str
    description: str
    stages: tuple[PipelineStage, ...]


def _dedupe_queries(queries: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = query.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _merge_evidence(primary: list[EvidenceSnippet], secondary: list[EvidenceSnippet], *, top_k: int) -> list[EvidenceSnippet]:
    best_by_id: dict[str, EvidenceSnippet] = {}

    for snippet in primary + secondary:
        existing = best_by_id.get(snippet.snippet_id)
        if existing is None:
            best_by_id[snippet.snippet_id] = snippet
            continue

        existing_score = existing.relevance_score if existing.relevance_score is not None else float("-inf")
        incoming_score = snippet.relevance_score if snippet.relevance_score is not None else float("-inf")
        if incoming_score > existing_score:
            best_by_id[snippet.snippet_id] = snippet

    ranked = sorted(
        best_by_id.values(),
        key=lambda s: s.relevance_score if s.relevance_score is not None else float("-inf"),
        reverse=True,
    )
    return ranked[:top_k]


def _stage_intent(ctx: PipelineContext, settings: Settings) -> None:
    intent = decompose_intent(ctx.prompt, settings)
    ctx.intent_bundle = intent
    ctx.retrieval_queries = _dedupe_queries([ctx.prompt.statement] + list(intent.retrieval_queries))
    con.step("variant", f"Intent stage produced {len(ctx.retrieval_queries)} retrieval queries")


def _stage_query_planner(ctx: PipelineContext, settings: Settings) -> None:
    if ctx.intent_bundle is None:
        raise ValueError("query_planner stage requires intent_bundle")
    extra_queries = plan_retrieval_queries(ctx.prompt, ctx.intent_bundle, settings)
    ctx.retrieval_queries = _dedupe_queries(ctx.retrieval_queries + extra_queries)
    con.step("variant", f"Query planner expanded to {len(ctx.retrieval_queries)} total queries")


def _stage_evidence_filter(ctx: PipelineContext, settings: Settings) -> None:
    queries = ctx.retrieval_queries or [ctx.prompt.statement]
    ctx.retrieved_evidence = retrieve_for_queries(ctx.prompt, queries, settings)


def _stage_counterevidence_miner(ctx: PipelineContext, settings: Settings) -> None:
    if ctx.intent_bundle is None:
        raise ValueError("counterevidence_miner stage requires intent_bundle")
    if not ctx.retrieved_evidence:
        return

    counter_queries = mine_counterevidence_queries(
        ctx.prompt,
        ctx.intent_bundle,
        ctx.retrieved_evidence,
        settings,
    )
    if not counter_queries:
        return

    counter_evidence = retrieve_for_queries(ctx.prompt, counter_queries, settings)
    ctx.retrieved_evidence = _merge_evidence(
        ctx.retrieved_evidence,
        counter_evidence,
        top_k=settings.synthesis_evidence_k,
    )
    con.step("variant", f"Counter-evidence merge retained {len(ctx.retrieved_evidence)} snippets")


def _stage_synthesis(ctx: PipelineContext, settings: Settings) -> None:
    if ctx.intent_bundle is None:
        raise ValueError("synthesis stage requires intent_bundle")

    evidence = ctx.retrieved_evidence
    if not evidence:
        evidence = retrieve_for_queries(ctx.prompt, [ctx.prompt.statement], settings)
        ctx.retrieved_evidence = evidence

    ctx.result = run_synthesis(
        ctx.prompt,
        ctx.intent_bundle,
        ctx.retrieved_evidence,
        settings,
    )


def _stage_critic(ctx: PipelineContext, settings: Settings) -> None:
    if ctx.intent_bundle is None or ctx.result is None:
        raise ValueError("critic stage requires synthesis output")

    ctx.critic_feedback = critique_focus_points(
        ctx.prompt,
        ctx.intent_bundle,
        ctx.retrieved_evidence,
        ctx.result.focus_points,
        settings,
    )


def _stage_refiner(ctx: PipelineContext, settings: Settings) -> None:
    if ctx.intent_bundle is None or ctx.result is None:
        raise ValueError("refiner stage requires synthesis output")
    if not ctx.critic_feedback:
        return

    refined_points = refine_focus_points(
        ctx.prompt,
        ctx.intent_bundle,
        ctx.retrieved_evidence,
        ctx.result.focus_points,
        ctx.critic_feedback,
        settings,
    )
    ctx.result = ctx.result.model_copy(update={"focus_points": refined_points})


def _stage_verifier(ctx: PipelineContext, settings: Settings) -> None:
    if ctx.result is None:
        raise ValueError("verifier stage requires synthesis output")
    ctx.result = verify_evidence(ctx.result, ctx.retrieved_evidence, settings)


_VARIANT_REGISTRY: dict[str, PipelineVariantSpec] = {
    "control": PipelineVariantSpec(
        name="control",
        description="Current pipeline: intent -> evidence -> synthesis -> verifier.",
        stages=(
            PipelineStage("intent", _stage_intent),
            PipelineStage("evidence", _stage_evidence_filter, ("intent",)),
            PipelineStage("synthesis", _stage_synthesis, ("evidence",)),
            PipelineStage("verifier", _stage_verifier, ("synthesis",)),
        ),
    ),
    "retrieval_hybrid": PipelineVariantSpec(
        name="retrieval_hybrid",
        description="Adds query planning and contradiction-focused evidence mining before synthesis.",
        stages=(
            PipelineStage("intent", _stage_intent),
            PipelineStage("query_planner", _stage_query_planner, ("intent",)),
            PipelineStage("evidence", _stage_evidence_filter, ("query_planner",)),
            PipelineStage("counterevidence", _stage_counterevidence_miner, ("evidence",)),
            PipelineStage("synthesis", _stage_synthesis, ("counterevidence",)),
            PipelineStage("verifier", _stage_verifier, ("synthesis",)),
        ),
    ),
    "critic_loop": PipelineVariantSpec(
        name="critic_loop",
        description="Adds a critic/refiner loop after synthesis.",
        stages=(
            PipelineStage("intent", _stage_intent),
            PipelineStage("evidence", _stage_evidence_filter, ("intent",)),
            PipelineStage("synthesis", _stage_synthesis, ("evidence",)),
            PipelineStage("critic", _stage_critic, ("synthesis",)),
            PipelineStage("refiner", _stage_refiner, ("critic",)),
            PipelineStage("verifier", _stage_verifier, ("refiner",)),
        ),
    ),
    "full_hybrid": PipelineVariantSpec(
        name="full_hybrid",
        description="Combines retrieval expansion and critic/refiner loop.",
        stages=(
            PipelineStage("intent", _stage_intent),
            PipelineStage("query_planner", _stage_query_planner, ("intent",)),
            PipelineStage("evidence", _stage_evidence_filter, ("query_planner",)),
            PipelineStage("counterevidence", _stage_counterevidence_miner, ("evidence",)),
            PipelineStage("synthesis", _stage_synthesis, ("counterevidence",)),
            PipelineStage("critic", _stage_critic, ("synthesis",)),
            PipelineStage("refiner", _stage_refiner, ("critic",)),
            PipelineStage("verifier", _stage_verifier, ("refiner",)),
        ),
    ),
}


def _validate_variant_spec(spec: PipelineVariantSpec) -> None:
    stage_ids = [stage.stage_id for stage in spec.stages]
    if len(stage_ids) != len(set(stage_ids)):
        raise ValueError(f"Duplicate stage ids in variant {spec.name!r}")

    known = set(stage_ids)
    for stage in spec.stages:
        for dep in stage.depends_on:
            if dep not in known:
                raise ValueError(f"Variant {spec.name!r} stage {stage.stage_id!r} depends on unknown stage {dep!r}")


def get_variant_spec(variant_name: str) -> PipelineVariantSpec:
    spec = _VARIANT_REGISTRY.get(variant_name)
    if spec is None:
        valid = ", ".join(sorted(_VARIANT_REGISTRY))
        raise ValueError(f"Unknown variant {variant_name!r}. Choose one of: {valid}")
    return spec


def list_variant_specs() -> list[PipelineVariantSpec]:
    return list(_VARIANT_REGISTRY.values())


def default_candidate_variants() -> list[str]:
    return [v.name for v in list_variant_specs() if v.name != "control"]


def _execute_variant(spec: PipelineVariantSpec, ctx: PipelineContext, settings: Settings) -> None:
    pending: dict[str, PipelineStage] = {stage.stage_id: stage for stage in spec.stages}
    completed: set[str] = set()

    while pending:
        progressed = False
        for stage_id in list(pending):
            stage = pending[stage_id]
            if not set(stage.depends_on).issubset(completed):
                continue
            stage.handler(ctx, settings)
            ctx.stage_trace.append(stage.stage_id)
            completed.add(stage.stage_id)
            del pending[stage_id]
            progressed = True

        if not progressed:
            blocked = ", ".join(sorted(pending))
            raise RuntimeError(f"Could not resolve runnable stage for variant {spec.name!r}. Blocked: {blocked}")


def run_variant_pipeline(
    prompt: FounderPrompt,
    variant_name: str,
    settings: Settings | None = None,
    *,
    output_root: Path | None = None,
    persist: bool = True,
) -> SynthesisResult:
    """Run a named pipeline variant and optionally persist its output."""

    s = settings or get_settings()
    spec = get_variant_spec(variant_name)
    _validate_variant_spec(spec)

    ctx = PipelineContext(prompt=prompt)
    con.step("variant", f"Running variant {spec.name!r} for prompt {prompt.id!r}...")
    _execute_variant(spec, ctx, s)

    if ctx.result is None:
        raise RuntimeError(f"Variant {spec.name!r} produced no synthesis result")

    result = ctx.result.model_copy(update={"system_variant": spec.name})

    if persist:
        root = output_root or (s.run_artifacts_dir.parent / "variant_runs")
        run_dir = root / spec.name / prompt.id
        run_dir.mkdir(parents=True, exist_ok=True)
        output_path = run_dir / "synthesis.json"
        output_path.write_text(
            result.model_dump_json(indent=2, exclude_none=True),
            encoding="utf-8",
        )
        con.success("variant", f"[{spec.name}] Saved -> {output_path}")

    return result
