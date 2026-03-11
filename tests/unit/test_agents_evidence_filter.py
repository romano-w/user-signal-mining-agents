from __future__ import annotations

import pytest

from user_signal_mining_agents.agents import evidence_filter
from user_signal_mining_agents.retrieval.index import DenseRetrievalHit


def test_retrieve_and_filter_dedupes_and_keeps_best_score(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
    founder_prompt,
    intent_bundle,
    evidence_factory,
) -> None:
    query_hits = {
        founder_prompt.statement: [
            DenseRetrievalHit(snippet=evidence_factory(1), score=0.50),
            DenseRetrievalHit(snippet=evidence_factory(2), score=0.40),
        ],
        "slow service": [
            DenseRetrievalHit(snippet=evidence_factory(1), score=0.91),
            DenseRetrievalHit(snippet=evidence_factory(3), score=0.65),
        ],
        "long wait": [
            DenseRetrievalHit(snippet=evidence_factory(4), score=0.80),
        ],
    }

    seen_queries: list[str] = []

    def _search(query: str, **_kwargs):
        seen_queries.append(query)
        return query_hits[query]

    monkeypatch.setattr(evidence_filter, "search_dense_index", _search)
    tmp_settings.synthesis_evidence_k = 3

    evidence = evidence_filter.retrieve_and_filter(founder_prompt, intent_bundle, tmp_settings)

    assert seen_queries == [founder_prompt.statement, "slow service", "long wait"]
    assert [snippet.snippet_id for snippet in evidence] == ["s-1", "s-4", "s-3"]
    assert evidence[0].relevance_score == pytest.approx(0.91)
