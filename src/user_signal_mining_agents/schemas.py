"""Pydantic data models shared across the agent pipeline and evaluation framework."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FounderPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    statement: str
    domain: str = "restaurants"
    notes: str | None = None


class IntentBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem_keywords: list[str] = Field(default_factory=list)
    target_user: str | None = None
    usage_context: str | None = None
    counter_hypotheses: list[str] = Field(default_factory=list)
    retrieval_queries: list[str] = Field(default_factory=list)


class EvidenceSnippet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snippet_id: str
    review_id: str
    business_id: str
    source: str = "yelp_review"
    business_name: str | None = None
    categories: list[str] = Field(default_factory=list)
    text: str
    stars: float | None = None
    city: str | None = None
    state: str | None = None
    review_date: str | None = None
    chunk_index: int | None = None
    relevance_score: float | None = None
    intensity_score: float | None = None
    diversity_group: str | None = None


class FocusPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    why_it_matters: str
    supporting_snippets: list[str] = Field(min_length=1, max_length=5)
    counter_signal: str
    next_validation_question: str


class SynthesisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_variant: str
    prompt: FounderPrompt
    intent_bundle: IntentBundle | None = None
    retrieved_evidence: list[EvidenceSnippet] = Field(default_factory=list)
    focus_points: list[FocusPoint] = Field(min_length=3, max_length=5)


class JudgeScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevance: float = Field(ge=1, le=5)
    groundedness: float = Field(ge=1, le=5)
    distinctiveness: float = Field(ge=1, le=5)
    overall_preference: float = Field(ge=1, le=5)
    rationale: str

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_groundedness_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        if "groundedness" not in normalized:
            legacy_scores = [
                float(value)
                for key in ("coverage", "contradiction")
                if isinstance((value := normalized.get(key)), int | float)
            ]
            if legacy_scores:
                normalized["groundedness"] = sum(legacy_scores) / len(legacy_scores)

        normalized.pop("coverage", None)
        normalized.pop("contradiction", None)
        return normalized


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_id: str
    system_variant: str
    scores: JudgeScores


class PromptEvaluationPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: FounderPrompt
    baseline_scores: JudgeResult
    pipeline_scores: JudgeResult
    baseline_panel: JudgePanelResult | None = None
    pipeline_panel: JudgePanelResult | None = None


class EvaluationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pairs: list[PromptEvaluationPair] = Field(default_factory=list)


class HumanAnnotationTask(BaseModel):
    """The data payload presented to the human annotator (completely blinded)."""
    model_config = ConfigDict(extra="forbid")

    task_id: str
    prompt: FounderPrompt
    retrieved_evidence: list[EvidenceSnippet]
    system_a_focus_points: list[FocusPoint]
    system_b_focus_points: list[FocusPoint]
    
    # Hidden metadata not shown in the annotation UI
    ground_truth_mapping: dict[str, str]


class HumanAnnotationScores(BaseModel):
    """The scores provided by the human for a single system's output."""
    model_config = ConfigDict(extra="forbid")

    relevance: int = Field(ge=1, le=5)
    groundedness: int = Field(ge=1, le=5)
    distinctiveness: int = Field(ge=1, le=5)
    rationale: str | None = None


class HumanAnnotationResult(BaseModel):
    """The final completed annotation."""
    model_config = ConfigDict(extra="forbid")

    task_id: str
    annotator_id: str
    system_a_scores: HumanAnnotationScores
    system_b_scores: HumanAnnotationScores
    overall_preference: Literal["system_a", "system_b", "tie"]
    difficulty_rating: int = Field(ge=1, le=5, description="How subjective or difficult was this to grade?")
    annotated_at: datetime = Field(default_factory=_utcnow)




class SnippetProvenance(BaseModel):
    """Traceability metadata for evidence snippets and cited quote spans."""
    model_config = ConfigDict(extra="forbid")

    source_dataset_id: str
    source_record_id: str
    source_type: str
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)
    extracted_at: datetime | None = None


class DatasetRecord(BaseModel):
    """Normalized ingest record produced by any source adapter."""
    model_config = ConfigDict(extra="forbid")

    record_id: str
    dataset_id: str
    source_type: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: SnippetProvenance | None = None


class DatasetSnapshotManifest(BaseModel):
    """Immutable manifest tying an ingest snapshot to source checksums."""
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    dataset_ids: list[str] = Field(default_factory=list)
    record_count: int = Field(ge=0)
    checksum_sha256: str
    source_manifests: dict[str, str] = Field(default_factory=dict)


class ExperimentManifest(BaseModel):
    """Run-level manifest for reproducibility and cross-run comparison."""
    model_config = ConfigDict(extra="forbid")

    run_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    dataset_snapshot_id: str
    prompt_bundle_id: str
    embedding_index_id: str
    system_variants: list[str] = Field(default_factory=list)
    llm_provider: str
    llm_model: str
    git_commit: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class MetricWithCI(BaseModel):
    """Metric summary including 95% confidence interval."""
    model_config = ConfigDict(extra="forbid")

    metric: str
    mean: float
    ci95_lower: float
    ci95_upper: float
    sample_size: int = Field(ge=1)


class SignificanceResult(BaseModel):
    """Statistical comparison result between two systems."""
    model_config = ConfigDict(extra="forbid")

    metric: str
    p_value: float = Field(ge=0.0, le=1.0)
    is_significant: bool
    effect_size: float | None = None
    notes: str | None = None


class JudgePanelResult(BaseModel):
    """Aggregated output from a multi-judge evaluation panel."""
    model_config = ConfigDict(extra="forbid")

    prompt_id: str
    system_variant: str
    panel_size: int = Field(ge=1)
    per_judge_scores: list[JudgeScores] = Field(default_factory=list)
    aggregate_scores: JudgeScores
    metrics_with_ci: list[MetricWithCI] = Field(default_factory=list)
    significance: list[SignificanceResult] = Field(default_factory=list)


class FailureTag(BaseModel):
    """Taxonomy tag applied to low-quality generations."""
    model_config = ConfigDict(extra="forbid")

    tag_id: str
    category: str
    severity: int = Field(ge=1, le=5)
    prompt_id: str | None = None
    description: str
    evidence_refs: list[str] = Field(default_factory=list)


class RobustnessCase(BaseModel):
    """One robustness stress case used during adversarial evaluation."""
    model_config = ConfigDict(extra="forbid")

    case_id: str
    family: str
    description: str
    transform_spec: dict[str, Any] = Field(default_factory=dict)
    expected_behavior: str


class DomainPack(BaseModel):
    """Declarative domain package for transfer evaluations."""
    model_config = ConfigDict(extra="forbid")

    domain_id: str
    title: str
    founder_prompts_path: str
    evaluation_notes: str | None = None
    enabled: bool = True


