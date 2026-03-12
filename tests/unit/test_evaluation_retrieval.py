from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from user_signal_mining_agents.evaluation import retrieval_report, retrieval_runner
from user_signal_mining_agents.retrieval.index import DenseRetrievalHit
from user_signal_mining_agents.schemas import EvidenceSnippet


def _hit(snippet_id: str) -> DenseRetrievalHit:
    snippet = EvidenceSnippet(
        snippet_id=snippet_id,
        review_id=f"r-{snippet_id}",
        business_id="b1",
        text=f"snippet text {snippet_id}",
    )
    return DenseRetrievalHit(snippet=snippet, score=1.0)


def test_run_retrieval_evaluation_computes_recall_mrr_ndcg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    tmp_settings,
) -> None:
    label_set = tmp_path / "labels.jsonl"
    label_set.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "slow service",
                "relevant_snippet_ids": ["s1", "s3"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        retrieval_runner,
        "search_retrieval_index",
        lambda *_args, **_kwargs: [_hit("s2"), _hit("s3"), _hit("s1")],
    )

    summary = retrieval_runner.run_retrieval_evaluation(
        label_set,
        tmp_settings,
        mode="hybrid",
        reranker="none",
        top_k=3,
        k_values=[1, 3],
    )

    assert summary.query_count == 1
    assert summary.aggregates["recall_at_k"]["1"] == 0.0
    assert summary.aggregates["recall_at_k"]["3"] == 1.0
    assert summary.aggregates["mrr_at_k"]["1"] == 0.0
    assert summary.aggregates["mrr_at_k"]["3"] == pytest.approx(0.5)
    assert summary.aggregates["ndcg_at_k"]["1"] == 0.0
    assert summary.aggregates["ndcg_at_k"]["3"] == pytest.approx(0.693426, rel=1e-5)


def test_run_retrieval_evaluation_supports_graded_relevance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    tmp_settings,
) -> None:
    label_set = tmp_path / "graded_labels.jsonl"
    label_set.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "slow service",
                "graded_relevance": {"s1": 3.0, "s2": 1.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        retrieval_runner,
        "search_retrieval_index",
        lambda *_args, **_kwargs: [_hit("s2"), _hit("s1")],
    )

    summary = retrieval_runner.run_retrieval_evaluation(
        label_set,
        tmp_settings,
        mode="hybrid",
        reranker="none",
        top_k=2,
        k_values=[2],
    )

    assert summary.aggregates["ndcg_at_k"]["2"] == pytest.approx(0.709810, rel=1e-5)


def test_generate_retrieval_report_writes_json_and_markdown(tmp_path) -> None:
    summary = retrieval_runner.RetrievalEvaluationSummary(
        generated_at=datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc),
        label_set_path="artifacts/retrieval_labels.jsonl",
        retrieval_mode="hybrid",
        reranker="none",
        top_k=10,
        k_values=[1, 3],
        query_count=1,
        dense_weight=1.0,
        lexical_weight=1.0,
        fusion_k=60,
        reranker_weight=0.25,
        candidate_pool=50,
        aggregates={
            "recall_at_k": {"1": 0.5, "3": 1.0},
            "mrr_at_k": {"1": 0.5, "3": 0.5},
            "ndcg_at_k": {"1": 0.5, "3": 0.8},
        },
        queries=[
            retrieval_runner.RetrievalQueryMetrics(
                query_id="q1",
                query="slow service",
                relevant_count=2,
                retrieved_snippet_ids=["s1", "s2", "s3"],
                recall_at_k={"1": 0.5, "3": 1.0},
                mrr_at_k={"1": 0.5, "3": 0.5},
                ndcg_at_k={"1": 0.5, "3": 0.8},
            )
        ],
    )

    json_path, markdown_path = retrieval_report.generate_retrieval_report(summary, tmp_path)

    assert json_path.exists()
    assert markdown_path.exists()
    assert "Retrieval Evaluation Report" in markdown_path.read_text(encoding="utf-8")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["query_count"] == 1
    assert payload["aggregates"]["recall_at_k"]["3"] == 1.0
