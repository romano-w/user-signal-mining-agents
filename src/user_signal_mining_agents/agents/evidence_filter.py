"""Evidence filter: multi-query retrieval, dedup, and top-K selection."""

from __future__ import annotations

from .. import console as con
from ..config import Settings, get_settings
from ..retrieval.index import search_retrieval_index
from ..schemas import EvidenceSnippet, FounderPrompt, IntentBundle


def _dedupe_queries(queries: list[str]) -> list[str]:
    """Preserve order while dropping empty/duplicate queries."""

    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = query.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def retrieve_for_queries(
    prompt: FounderPrompt,
    queries: list[str],
    settings: Settings | None = None,
) -> list[EvidenceSnippet]:
    """Retrieve snippets for a list of queries, deduplicate, and keep top-K."""

    s = settings or get_settings()
    deduped_queries = _dedupe_queries(queries)
    if not deduped_queries:
        deduped_queries = [prompt.statement]

    con.step("evidence", f"Searching with {len(deduped_queries)} queries...")

    # Gather all hits, tracking best score per snippet.
    best_by_id: dict[str, tuple[float, EvidenceSnippet]] = {}
    for query in deduped_queries:
        hits = search_retrieval_index(
            query,
            index_dir=s.index_dir,
            embedding_model=s.embedding_model,
            top_k=s.retrieval_top_k,
            mode=s.retrieval_mode,
            dense_weight=s.retrieval_dense_weight,
            lexical_weight=s.retrieval_lexical_weight,
            fusion_k=s.retrieval_fusion_k,
            candidate_pool=s.retrieval_candidate_pool,
            reranker=s.retrieval_reranker,
            reranker_weight=s.retrieval_reranker_weight,
            bm25_k1=s.retrieval_bm25_k1,
            bm25_b=s.retrieval_bm25_b,
        )
        for hit in hits:
            sid = hit.snippet.snippet_id
            existing = best_by_id.get(sid)
            if existing is None or hit.score > existing[0]:
                updated = hit.snippet.model_copy(update={"relevance_score": hit.score})
                best_by_id[sid] = (hit.score, updated)

    # Sort by best score descending.
    ranked = sorted(best_by_id.values(), key=lambda pair: pair[0], reverse=True)

    # Take top synthesis_evidence_k.
    top_k = s.synthesis_evidence_k
    evidence = [snippet for _score, snippet in ranked[:top_k]]
    con.step("evidence", f"Selected {len(evidence)} snippets from {len(best_by_id)} unique candidates")
    return evidence


def retrieve_and_filter(
    prompt: FounderPrompt,
    intent: IntentBundle,
    settings: Settings | None = None,
) -> list[EvidenceSnippet]:
    """Retrieve snippets for each query in the intent bundle, deduplicate, re-rank."""

    queries = [prompt.statement] + list(intent.retrieval_queries)
    return retrieve_for_queries(prompt, queries, settings)
