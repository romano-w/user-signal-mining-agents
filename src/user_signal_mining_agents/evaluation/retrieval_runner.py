"""Runner for retrieval metrics over labeled query sets."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, Field, model_validator

from ..config import Settings, get_settings
from ..retrieval.index import search_retrieval_index

DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)


class RetrievalLabel(BaseModel):
    query_id: str | None = None
    query: str
    relevant_snippet_ids: list[str] = Field(default_factory=list)
    graded_relevance: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize(self) -> "RetrievalLabel":
        if not self.relevant_snippet_ids and not self.graded_relevance:
            raise ValueError("Each label must include relevant_snippet_ids or graded_relevance")

        deduped = list(dict.fromkeys(self.relevant_snippet_ids))
        if not deduped:
            deduped = list(self.graded_relevance.keys())

        for snippet_id in deduped:
            self.graded_relevance.setdefault(snippet_id, 1.0)

        self.relevant_snippet_ids = deduped
        if not self.query_id or not self.query_id.strip():
            self.query_id = self.query
        return self


class RetrievalQueryMetrics(BaseModel):
    query_id: str
    query: str
    relevant_count: int
    retrieved_snippet_ids: list[str]
    recall_at_k: dict[str, float]
    mrr_at_k: dict[str, float]
    ndcg_at_k: dict[str, float]


class RetrievalEvaluationSummary(BaseModel):
    generated_at: datetime
    label_set_path: str
    retrieval_mode: str
    reranker: str
    top_k: int
    k_values: list[int]
    query_count: int
    dense_weight: float
    lexical_weight: float
    fusion_k: int
    reranker_weight: float
    candidate_pool: int
    aggregates: dict[str, dict[str, float]]
    queries: list[RetrievalQueryMetrics] = Field(default_factory=list)



def load_retrieval_labels(path: Path) -> list[RetrievalLabel]:
    labels: list[RetrievalLabel] = []

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw_line.strip()
        if not text:
            continue

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc

        try:
            labels.append(RetrievalLabel.model_validate(payload))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid retrieval label on line {line_no} in {path}: {exc}") from exc

    if not labels:
        raise ValueError(f"No retrieval labels found in {path}")

    return labels



def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0



def _recall_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)



def _mrr_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    for rank, snippet_id in enumerate(retrieved_ids[:k], start=1):
        if snippet_id in relevant_ids:
            return 1.0 / rank
    return 0.0



def _dcg(relevances: Sequence[float]) -> float:
    return sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(relevances))



def _ndcg_at_k(retrieved_ids: Sequence[str], graded_relevance: dict[str, float], k: int) -> float:
    if not graded_relevance:
        return 0.0

    gains = [float(graded_relevance.get(snippet_id, 0.0)) for snippet_id in retrieved_ids[:k]]
    dcg = _dcg(gains)

    ideal = sorted((float(score) for score in graded_relevance.values()), reverse=True)[:k]
    idcg = _dcg(ideal)
    if idcg <= 0:
        return 0.0
    return dcg / idcg



def run_retrieval_evaluation(
    label_set_path: Path,
    settings: Settings | None = None,
    *,
    mode: str | None = None,
    reranker: str | None = None,
    top_k: int | None = None,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> RetrievalEvaluationSummary:
    s = settings or get_settings()
    labels = load_retrieval_labels(label_set_path)

    normalized_k_values = sorted({int(k) for k in k_values if int(k) > 0})
    if not normalized_k_values:
        raise ValueError("k_values must contain at least one positive integer")

    effective_mode = (mode or s.retrieval_mode).strip().lower()
    effective_reranker = (reranker or s.retrieval_reranker).strip().lower()
    effective_top_k = max(top_k or s.retrieval_top_k, max(normalized_k_values))
    candidate_pool = max(s.retrieval_candidate_pool, effective_top_k)

    per_query: list[RetrievalQueryMetrics] = []

    for label in labels:
        hits = search_retrieval_index(
            label.query,
            index_dir=s.index_dir,
            embedding_model=s.embedding_model,
            top_k=effective_top_k,
            mode=effective_mode,
            dense_weight=s.retrieval_dense_weight,
            lexical_weight=s.retrieval_lexical_weight,
            fusion_k=s.retrieval_fusion_k,
            candidate_pool=candidate_pool,
            reranker=effective_reranker,
            reranker_weight=s.retrieval_reranker_weight,
            bm25_k1=s.retrieval_bm25_k1,
            bm25_b=s.retrieval_bm25_b,
        )
        retrieved_ids = [hit.snippet.snippet_id for hit in hits]
        relevant_ids = set(label.relevant_snippet_ids)

        recall = {
            str(k): _recall_at_k(retrieved_ids, relevant_ids, k)
            for k in normalized_k_values
        }
        mrr = {
            str(k): _mrr_at_k(retrieved_ids, relevant_ids, k)
            for k in normalized_k_values
        }
        ndcg = {
            str(k): _ndcg_at_k(retrieved_ids, label.graded_relevance, k)
            for k in normalized_k_values
        }

        per_query.append(
            RetrievalQueryMetrics(
                query_id=label.query_id or label.query,
                query=label.query,
                relevant_count=len(relevant_ids),
                retrieved_snippet_ids=retrieved_ids,
                recall_at_k=recall,
                mrr_at_k=mrr,
                ndcg_at_k=ndcg,
            )
        )

    aggregates = {
        "recall_at_k": {
            str(k): _mean([result.recall_at_k[str(k)] for result in per_query])
            for k in normalized_k_values
        },
        "mrr_at_k": {
            str(k): _mean([result.mrr_at_k[str(k)] for result in per_query])
            for k in normalized_k_values
        },
        "ndcg_at_k": {
            str(k): _mean([result.ndcg_at_k[str(k)] for result in per_query])
            for k in normalized_k_values
        },
    }

    return RetrievalEvaluationSummary(
        generated_at=datetime.now(timezone.utc),
        label_set_path=str(label_set_path),
        retrieval_mode=effective_mode,
        reranker=effective_reranker,
        top_k=effective_top_k,
        k_values=normalized_k_values,
        query_count=len(per_query),
        dense_weight=s.retrieval_dense_weight,
        lexical_weight=s.retrieval_lexical_weight,
        fusion_k=s.retrieval_fusion_k,
        reranker_weight=s.retrieval_reranker_weight,
        candidate_pool=candidate_pool,
        aggregates=aggregates,
        queries=per_query,
    )
