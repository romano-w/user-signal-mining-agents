from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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

    system_variant: Literal["baseline", "pipeline"]
    prompt: FounderPrompt
    intent_bundle: IntentBundle | None = None
    retrieved_evidence: list[EvidenceSnippet] = Field(default_factory=list)
    focus_points: list[FocusPoint] = Field(min_length=3, max_length=5)


class JudgeScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevance: float = Field(ge=1, le=5)
    actionability: float = Field(ge=1, le=5)
    evidence_grounding: float = Field(ge=1, le=5)
    contradiction_handling: float = Field(ge=1, le=5)
    non_redundancy: float = Field(ge=1, le=5)
    rationale: str

    @property
    def overall_avg(self) -> float:
        return (
            self.relevance
            + self.actionability
            + self.evidence_grounding
            + self.contradiction_handling
            + self.non_redundancy
        ) / 5


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_id: str
    system_variant: Literal["baseline", "pipeline"]
    scores: JudgeScores


class PromptEvaluationPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: FounderPrompt
    baseline_scores: JudgeResult
    pipeline_scores: JudgeResult


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
    actionability: int = Field(ge=1, le=5)
    evidence_grounding: int = Field(ge=1, le=5)
    contradiction_handling: int = Field(ge=1, le=5)
    non_redundancy: int = Field(ge=1, le=5)
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
    annotated_at: datetime = Field(default_factory=datetime.utcnow)

