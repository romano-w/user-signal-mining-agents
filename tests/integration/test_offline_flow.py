from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from user_signal_mining_agents.data.chunking import iter_chunked_reviews, write_snippets_jsonl
from user_signal_mining_agents.data.yelp_loader import iter_restaurant_reviews, load_restaurant_business_lookup
from user_signal_mining_agents.evaluation import report, runner
from user_signal_mining_agents.retrieval import index as retrieval_index
from user_signal_mining_agents.schemas import FocusPoint, FounderPrompt, JudgeResult, JudgeScores, SynthesisResult


@pytest.mark.integration
def test_offline_chunk_to_index_to_search(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    businesses_src = Path("tests/fixtures/businesses.jsonl")
    reviews_src = Path("tests/fixtures/reviews.jsonl")

    businesses = load_restaurant_business_lookup(businesses_src)
    reviews = iter_restaurant_reviews(
        reviews_src,
        businesses,
        review_limit=10,
        min_review_characters=20,
        max_reviews_per_business=2,
    )
    snippets = list(
        iter_chunked_reviews(
            reviews,
            businesses,
            sentence_window=1,
            sentence_stride=1,
            max_chunks=2,
            min_chunk_characters=20,
        )
    )
    assert snippets

    snippets_path = tmp_path / "snippets.jsonl"
    index_dir = tmp_path / "index"
    write_snippets_jsonl(snippets, snippets_path)

    class _FakeModel:
        def encode(self, texts, **_kwargs):
            vectors = []
            for text in texts:
                t = text.lower()
                vectors.append(
                    [
                        1.0 if ("slow" in t or "wait" in t) else 0.0,
                        1.0 if ("pickup" in t or "wrong" in t) else 0.0,
                    ]
                )
            return np.array(vectors, dtype=np.float32)

    monkeypatch.setattr(retrieval_index, "load_embedding_model", lambda *_args, **_kwargs: _FakeModel())

    retrieval_index.build_dense_index_from_jsonl(
        snippets_path,
        index_dir=index_dir,
        embedding_model="fake-model",
        batch_size=4,
        device="cpu",
    )

    hits = retrieval_index.search_dense_index(
        "slow service",
        index_dir=index_dir,
        top_k=3,
        device="cpu",
    )

    assert hits
    assert "slow" in hits[0].snippet.text.lower() or "wait" in hits[0].snippet.text.lower()


@pytest.mark.integration
def test_offline_evaluation_and_report_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
) -> None:
    prompts = [
        FounderPrompt(id="fp-1", statement="Why do guests stop returning?", domain="restaurants"),
        FounderPrompt(id="fp-2", statement="Why is pickup unreliable?", domain="restaurants"),
    ]
    tmp_settings.founder_prompts_path.write_text(json.dumps([p.model_dump() for p in prompts]), encoding="utf-8")

    def _synth(prompt: FounderPrompt, variant: str) -> SynthesisResult:
        return SynthesisResult(
            system_variant=variant,
            prompt=prompt,
            retrieved_evidence=[],
            focus_points=[
                FocusPoint(
                    label="L1",
                    why_it_matters="W1",
                    supporting_snippets=["Q1"],
                    counter_signal="C1",
                    next_validation_question="N1",
                ),
                FocusPoint(
                    label="L2",
                    why_it_matters="W2",
                    supporting_snippets=["Q2"],
                    counter_signal="C2",
                    next_validation_question="N2",
                ),
                FocusPoint(
                    label="L3",
                    why_it_matters="W3",
                    supporting_snippets=["Q3"],
                    counter_signal="C3",
                    next_validation_question="N3",
                ),
            ],
        )

    def _scores(prompt_id: str, variant: str, value: float) -> JudgeResult:
        return JudgeResult(
            prompt_id=prompt_id,
            system_variant=variant,
            scores=JudgeScores(
                relevance=value,
                overall_preference=value,
                groundedness=value,
                distinctiveness=value,
                rationale=f"rationale {variant}",
            ),
        )

    monkeypatch.setattr(runner, "run_baseline", lambda prompt, _settings: _synth(prompt, "baseline"))
    monkeypatch.setattr(runner, "run_pipeline", lambda prompt, _settings: _synth(prompt, "pipeline"))
    monkeypatch.setattr(
        runner,
        "judge_pair",
        lambda prompt, *_args: (_scores(prompt.id, "baseline", 3.0), _scores(prompt.id, "pipeline", 4.0)),
    )

    summary = runner.run_evaluation(tmp_settings, skip_cached=False)
    report_path = report.generate_report(summary, tmp_settings.run_artifacts_dir)

    assert len(summary.pairs) == 2
    assert report_path.exists()
    assert "Aggregate Scores" in report_path.read_text(encoding="utf-8")
    assert (tmp_settings.run_artifacts_dir / "fp-1" / "judge_baseline.json").exists()
    assert (tmp_settings.run_artifacts_dir / "fp-2" / "judge_pipeline.json").exists()

