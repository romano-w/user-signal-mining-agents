"""Evidence filter: multi-query retrieval, dedup, and top-K selection."""

from __future__ import annotations

from ..config import Settings, get_settings
from ..retrieval.index import search_dense_index
from ..schemas import EvidenceSnippet, FounderPrompt, IntentBundle
from .. import console as con


def retrieve_and_filter(
    prompt: FounderPrompt,
    intent: IntentBundle,
    settings: Settings | None = None,
) -> list[EvidenceSnippet]:
    """Retrieve snippets for each query in the intent bundle, deduplicate, re-rank."""

    s = settings or get_settings()

    # Collect queries: original statement + intent-derived queries
    queries = [prompt.statement] + list(intent.retrieval_queries)
    con.step("evidence", f"Searching with {len(queries)} queries...")

    # Gather all hits, tracking best score per snippet
    best_by_id: dict[str, tuple[float, EvidenceSnippet]] = {}
    for query in queries:
        hits = search_dense_index(
            query,
            index_dir=s.index_dir,
            top_k=s.retrieval_top_k,
        )
        for hit in hits:
            sid = hit.snippet.snippet_id
            existing = best_by_id.get(sid)
            if existing is None or hit.score > existing[0]:
                updated = hit.snippet.model_copy(update={"relevance_score": hit.score})
                best_by_id[sid] = (hit.score, updated)

    # Sort by best score descending
    ranked = sorted(best_by_id.values(), key=lambda pair: pair[0], reverse=True)

    # Take top synthesis_evidence_k
    top_k = s.synthesis_evidence_k
    evidence = [snippet for _score, snippet in ranked[:top_k]]
    con.step("evidence", f"Selected {len(evidence)} snippets from {len(best_by_id)} unique candidates")
    return evidence
