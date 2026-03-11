from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from user_signal_mining_agents import config as config_module
from user_signal_mining_agents import llm_client
from user_signal_mining_agents.retrieval import index as retrieval_index
from user_signal_mining_agents.config import Settings
from user_signal_mining_agents.schemas import (
    EvidenceSnippet,
    FocusPoint,
    FounderPrompt,
    IntentBundle,
    SynthesisResult,
)


@pytest.fixture(autouse=True)
def reset_singletons() -> None:
    """Reset module-level caches so tests stay isolated and deterministic."""
    config_module.get_settings.cache_clear()
    llm_client._gemini_key_cycle = None
    retrieval_index._MODEL_CACHE.clear()
    retrieval_index._INDEX_CACHE.clear()


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    for name in ("baseline.md", "intent.md", "synthesis.md", "judge.md", "evidence_verifier.md", "query_planner.md", "counterevidence_miner.md", "critic.md", "refiner.md"):
        (prompts_dir / name).write_text(f"template::{name}", encoding="utf-8")

    founder_prompts_path = tmp_path / "founder_prompts.json"
    founder_prompts_path.write_text(
        json.dumps(
            [
                {
                    "id": "p1",
                    "statement": "Why are first-time diners not returning?",
                    "domain": "restaurants",
                },
                {
                    "id": "p2",
                    "statement": "Why do takeout orders feel unreliable?",
                    "domain": "restaurants",
                },
            ]
        ),
        encoding="utf-8",
    )

    yelp_dataset_dir = tmp_path / "data" / "raw" / "Yelp-JSON"
    settings = Settings(
        llm_provider="openai",
        llm_model="test-model",
        llm_temperature=0.1,
        openai_api_key="test-openai",
        gemini_api_key_1="g1",
        gemini_api_key_2="g2",
        openrouter_api_key="r1",
        founder_prompts_path=founder_prompts_path,
        prompts_dir=prompts_dir,
        index_dir=tmp_path / "artifacts" / "index",
        run_artifacts_dir=tmp_path / "artifacts" / "runs",
        yelp_dataset_dir=yelp_dataset_dir,
        yelp_download_zip_path=yelp_dataset_dir / "Yelp-JSON.zip",
        yelp_tar_path=yelp_dataset_dir / "yelp_dataset.tar",
        yelp_businesses_path=yelp_dataset_dir / "yelp_academic_dataset_business.json",
        yelp_reviews_path=yelp_dataset_dir / "yelp_academic_dataset_review.json",
        working_subset_path=tmp_path / "data" / "processed" / "restaurant_reviews.jsonl",
        retrieval_top_k=5,
        synthesis_evidence_k=3,
    )
    return settings


@pytest.fixture
def founder_prompt() -> FounderPrompt:
    return FounderPrompt(
        id="prompt-1",
        statement="I need to understand slow service complaints.",
        domain="restaurants",
    )


@pytest.fixture
def intent_bundle() -> IntentBundle:
    return IntentBundle(
        problem_keywords=["slow service", "wait time"],
        target_user="dine-in guests",
        usage_context="weekend dinner",
        counter_hypotheses=["kitchen bottleneck"],
        retrieval_queries=["slow service", "long wait"],
    )


@pytest.fixture
def evidence_factory() -> Callable[..., EvidenceSnippet]:
    def _make(
        idx: int,
        *,
        text: str | None = None,
        score: float | None = None,
        business_id: str = "biz-1",
    ) -> EvidenceSnippet:
        return EvidenceSnippet(
            snippet_id=f"s-{idx}",
            review_id=f"r-{idx}",
            business_id=business_id,
            business_name=f"Restaurant {idx}",
            text=text or f"Snippet text {idx}",
            stars=4.0,
            relevance_score=score,
        )

    return _make


@pytest.fixture
def focus_point_factory() -> Callable[..., FocusPoint]:
    def _make(idx: int, *, label: str | None = None) -> FocusPoint:
        return FocusPoint(
            label=label or f"Focus {idx}",
            why_it_matters=f"Reason {idx}",
            supporting_snippets=[f"Quote {idx}"],
            counter_signal=f"Counter {idx}",
            next_validation_question=f"Question {idx}",
        )

    return _make


@pytest.fixture
def synthesis_result(
    founder_prompt: FounderPrompt,
    intent_bundle: IntentBundle,
    evidence_factory: Callable[..., EvidenceSnippet],
    focus_point_factory: Callable[..., FocusPoint],
) -> SynthesisResult:
    return SynthesisResult(
        system_variant="pipeline",
        prompt=founder_prompt,
        intent_bundle=intent_bundle,
        retrieved_evidence=[evidence_factory(1), evidence_factory(2)],
        focus_points=[focus_point_factory(1), focus_point_factory(2), focus_point_factory(3)],
    )

