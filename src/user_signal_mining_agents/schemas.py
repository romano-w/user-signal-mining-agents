from __future__ import annotations

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
    business_name: str | None = None
    categories: list[str] = Field(default_factory=list)
    text: str
    stars: float | None = None
    city: str | None = None
    relevance_score: float | None = None
    intensity_score: float | None = None
    diversity_group: str | None = None


class FocusPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    why_it_matters: str
    supporting_snippets: list[str] = Field(min_length=2, max_length=3)
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


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_id: str
    system_variant: Literal["baseline", "pipeline"]
    scores: JudgeScores
