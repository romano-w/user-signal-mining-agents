from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import torch
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

import transformers

transformers.logging.set_verbosity_error()

from ..data.chunking import load_snippets_jsonl, write_snippets_jsonl
from ..schemas import EvidenceSnippet

# Suppress noisy model-loading logs from sentence-transformers
for _logger_name in ("sentence_transformers", "transformers.modeling_utils"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)


EMBEDDINGS_FILENAME = "embeddings.npy"
SNIPPETS_FILENAME = "snippets.jsonl"
METADATA_FILENAME = "metadata.json"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class DenseIndexMetadata(BaseModel):
    embedding_model: str
    device: str
    snippet_count: int
    vector_dimension: int
    normalized: bool = True


@dataclass(slots=True, frozen=True)
class DenseRetrievalHit:
    snippet: EvidenceSnippet
    score: float


@dataclass(slots=True, frozen=True)
class _LexicalIndex:
    postings: dict[str, list[tuple[int, int]]]
    idf: dict[str, float]
    doc_lengths: list[int]
    avg_doc_length: float
    doc_count: int


def infer_embedding_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_embedding_device(device: str | None) -> str:
    if device is None:
        return infer_embedding_device()

    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"

    return device


# Singleton cache: (model_name, device) -> SentenceTransformer
_MODEL_CACHE: dict[tuple[str, str], SentenceTransformer] = {}


def load_embedding_model(
    model_name: str,
    *,
    device: str | None = None,
) -> SentenceTransformer:
    resolved_device = resolve_embedding_device(device)
    cache_key = (model_name, resolved_device)
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = SentenceTransformer(model_name, device=resolved_device)
    return _MODEL_CACHE[cache_key]


def build_dense_index(
    snippets: Sequence[EvidenceSnippet],
    *,
    index_dir: Path,
    embedding_model: str,
    batch_size: int = 128,
    device: str | None = None,
) -> DenseIndexMetadata:
    if not snippets:
        raise ValueError("Cannot build an index without snippets.")

    resolved_device = resolve_embedding_device(device)
    model = load_embedding_model(embedding_model, device=resolved_device)
    texts = [snippet.text for snippet in snippets]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / EMBEDDINGS_FILENAME, embeddings)
    write_snippets_jsonl(snippets, index_dir / SNIPPETS_FILENAME)

    metadata = DenseIndexMetadata(
        embedding_model=embedding_model,
        device=resolved_device,
        snippet_count=len(snippets),
        vector_dimension=int(embeddings.shape[1]),
    )
    (index_dir / METADATA_FILENAME).write_text(
        metadata.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return metadata


def build_dense_index_from_jsonl(
    snippets_path: Path,
    *,
    index_dir: Path,
    embedding_model: str,
    batch_size: int = 128,
    device: str | None = None,
) -> DenseIndexMetadata:
    snippets = load_snippets_jsonl(snippets_path)
    return build_dense_index(
        snippets,
        index_dir=index_dir,
        embedding_model=embedding_model,
        batch_size=batch_size,
        device=device,
    )


# Cache loaded index data to avoid re-reading from disk on every search
_INDEX_CACHE: dict[str, tuple[DenseIndexMetadata, np.ndarray, list[EvidenceSnippet]]] = {}
_LEXICAL_CACHE: dict[str, _LexicalIndex] = {}


def load_dense_index(index_dir: Path) -> tuple[DenseIndexMetadata, np.ndarray, list[EvidenceSnippet]]:
    cache_key = str(index_dir)
    if cache_key not in _INDEX_CACHE:
        metadata = DenseIndexMetadata.model_validate_json(
            (index_dir / METADATA_FILENAME).read_text(encoding="utf-8")
        )
        embeddings = np.load(index_dir / EMBEDDINGS_FILENAME)
        snippets = load_snippets_jsonl(index_dir / SNIPPETS_FILENAME)
        _INDEX_CACHE[cache_key] = (metadata, embeddings, snippets)
    return _INDEX_CACHE[cache_key]


def _encode_query(model: SentenceTransformer, query: str) -> np.ndarray:
    return model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)[0]


def _top_indices(scores: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0 or scores.size == 0:
        return np.array([], dtype=np.int64)
    limit = min(limit, int(scores.size))
    return np.argsort(-scores, kind="stable")[:limit]


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _build_lexical_index(snippets: Sequence[EvidenceSnippet]) -> _LexicalIndex:
    postings_by_term: dict[str, list[tuple[int, int]]] = defaultdict(list)
    doc_lengths: list[int] = []
    doc_count = len(snippets)

    for idx, snippet in enumerate(snippets):
        token_counts = Counter(_tokenize(snippet.text))
        doc_lengths.append(sum(token_counts.values()))
        for term, tf in token_counts.items():
            postings_by_term[term].append((idx, tf))

    avg_doc_length = sum(doc_lengths) / doc_count if doc_count else 1.0
    if avg_doc_length <= 0:
        avg_doc_length = 1.0

    idf: dict[str, float] = {}
    for term, postings in postings_by_term.items():
        doc_freq = len(postings)
        # BM25-style IDF with +1 smoothing to keep scores stable.
        idf[term] = math.log1p((doc_count - doc_freq + 0.5) / (doc_freq + 0.5))

    return _LexicalIndex(
        postings=dict(postings_by_term),
        idf=idf,
        doc_lengths=doc_lengths,
        avg_doc_length=avg_doc_length,
        doc_count=doc_count,
    )


def _load_lexical_index(index_dir: Path, snippets: Sequence[EvidenceSnippet]) -> _LexicalIndex:
    cache_key = str(index_dir)
    if cache_key not in _LEXICAL_CACHE:
        _LEXICAL_CACHE[cache_key] = _build_lexical_index(snippets)
    return _LEXICAL_CACHE[cache_key]


def _score_lexical_query(
    query_tokens: list[str],
    lexical_index: _LexicalIndex,
    *,
    bm25_k1: float,
    bm25_b: float,
) -> np.ndarray:
    scores = np.zeros(lexical_index.doc_count, dtype=np.float32)
    if not query_tokens:
        return scores

    term_counts = Counter(query_tokens)
    avg_doc_length = lexical_index.avg_doc_length

    for term, query_tf in term_counts.items():
        postings = lexical_index.postings.get(term)
        if not postings:
            continue
        idf = lexical_index.idf.get(term, 0.0)
        for doc_idx, doc_tf in postings:
            doc_len = lexical_index.doc_lengths[doc_idx]
            denom = doc_tf + bm25_k1 * (1 - bm25_b + bm25_b * (doc_len / avg_doc_length))
            if denom <= 0:
                continue
            scores[doc_idx] += float(idf * query_tf * (doc_tf * (bm25_k1 + 1) / denom))

    return scores


def _normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}

    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)
    if max_score - min_score <= 1e-12:
        return {idx: 1.0 for idx in scores}

    return {
        idx: (score - min_score) / (max_score - min_score)
        for idx, score in scores.items()
    }


def _token_overlap_score(query_tokens: list[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    q_terms = set(query_tokens)
    doc_terms = set(_tokenize(text))
    if not doc_terms:
        return 0.0

    overlap = len(q_terms & doc_terms) / len(q_terms)
    query_phrase = " ".join(query_tokens)
    phrase_bonus = 0.15 if query_phrase and query_phrase in text.lower() else 0.0
    return overlap + phrase_bonus


def _apply_token_overlap_reranker(
    *,
    query_tokens: list[str],
    ranked_indices: list[int],
    snippets: Sequence[EvidenceSnippet],
    base_scores: dict[int, float],
    reranker_weight: float,
) -> tuple[list[int], dict[int, float]]:
    if not ranked_indices:
        return ranked_indices, base_scores

    clipped_weight = min(1.0, max(0.0, reranker_weight))
    if clipped_weight <= 0:
        return ranked_indices, base_scores

    reranker_raw = {
        idx: _token_overlap_score(query_tokens, snippets[idx].text)
        for idx in ranked_indices
    }
    base_norm = _normalize_scores(base_scores)
    rerank_norm = _normalize_scores(reranker_raw)
    final_scores = {
        idx: (1 - clipped_weight) * base_norm.get(idx, 0.0) + clipped_weight * rerank_norm.get(idx, 0.0)
        for idx in ranked_indices
    }
    reranked = sorted(
        ranked_indices,
        key=lambda idx: (
            -final_scores[idx],
            -base_scores.get(idx, 0.0),
            idx,
        ),
    )
    return reranked, final_scores


def _validate_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in {"dense", "lexical", "hybrid"}:
        raise ValueError(f"Unsupported retrieval mode: {mode!r}")
    return normalized


def _validate_reranker(reranker: str) -> str:
    normalized = reranker.strip().lower()
    if normalized not in {"none", "token_overlap"}:
        raise ValueError(f"Unsupported reranker: {reranker!r}")
    return normalized


def _hits_from_ranked_indices(
    indices: Sequence[int],
    snippets: Sequence[EvidenceSnippet],
    scores: dict[int, float],
) -> list[DenseRetrievalHit]:
    hits: list[DenseRetrievalHit] = []
    for index in indices:
        score = float(scores.get(index, 0.0))
        snippet = snippets[index].model_copy(update={"relevance_score": score})
        hits.append(DenseRetrievalHit(snippet=snippet, score=score))
    return hits


def search_dense_index(
    query: str,
    *,
    index_dir: Path,
    embedding_model: str | None = None,
    top_k: int = 10,
    device: str | None = None,
) -> list[DenseRetrievalHit]:
    if top_k <= 0:
        return []

    metadata, embeddings, snippets = load_dense_index(index_dir)
    model_name = embedding_model or metadata.embedding_model
    model = load_embedding_model(model_name, device=device or metadata.device)
    query_embedding = _encode_query(model, query)

    scores = embeddings @ query_embedding
    top_indices = _top_indices(scores, top_k)

    return _hits_from_ranked_indices(
        [int(i) for i in top_indices],
        snippets,
        {int(i): float(scores[i]) for i in top_indices},
    )


def search_lexical_index(
    query: str,
    *,
    index_dir: Path,
    top_k: int = 10,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> list[DenseRetrievalHit]:
    if top_k <= 0:
        return []

    _metadata, _embeddings, snippets = load_dense_index(index_dir)
    lexical_index = _load_lexical_index(index_dir, snippets)
    lexical_scores = _score_lexical_query(
        _tokenize(query),
        lexical_index,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
    )
    top_indices = _top_indices(lexical_scores, top_k)

    return _hits_from_ranked_indices(
        [int(i) for i in top_indices],
        snippets,
        {int(i): float(lexical_scores[i]) for i in top_indices},
    )


def search_retrieval_index(
    query: str,
    *,
    index_dir: Path,
    embedding_model: str | None = None,
    top_k: int = 10,
    device: str | None = None,
    mode: str = "hybrid",
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
    fusion_k: int = 60,
    candidate_pool: int | None = None,
    reranker: str = "none",
    reranker_weight: float = 0.25,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> list[DenseRetrievalHit]:
    if top_k <= 0:
        return []

    normalized_mode = _validate_mode(mode)
    normalized_reranker = _validate_reranker(reranker)
    candidate_limit = max(top_k, candidate_pool or (top_k * 4), 1)

    metadata, embeddings, snippets = load_dense_index(index_dir)
    query_tokens = _tokenize(query)

    dense_scores: np.ndarray | None = None
    dense_indices: np.ndarray = np.array([], dtype=np.int64)
    if normalized_mode in {"dense", "hybrid"}:
        model_name = embedding_model or metadata.embedding_model
        model = load_embedding_model(model_name, device=device or metadata.device)
        query_embedding = _encode_query(model, query)
        dense_scores = embeddings @ query_embedding
        dense_indices = _top_indices(dense_scores, candidate_limit)

    lexical_scores: np.ndarray | None = None
    lexical_indices: np.ndarray = np.array([], dtype=np.int64)
    if normalized_mode in {"lexical", "hybrid"}:
        lexical_index = _load_lexical_index(index_dir, snippets)
        lexical_scores = _score_lexical_query(
            query_tokens,
            lexical_index,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
        )
        lexical_indices = _top_indices(lexical_scores, candidate_limit)

    base_scores: dict[int, float]
    ranked_indices: list[int]

    if normalized_mode == "dense":
        assert dense_scores is not None
        ranked_indices = [int(i) for i in dense_indices]
        base_scores = {int(i): float(dense_scores[i]) for i in dense_indices}
    elif normalized_mode == "lexical":
        assert lexical_scores is not None
        ranked_indices = [int(i) for i in lexical_indices]
        base_scores = {int(i): float(lexical_scores[i]) for i in lexical_indices}
    else:
        fused_scores: dict[int, float] = {}
        for rank, index in enumerate(dense_indices.tolist(), start=1):
            fused_scores[index] = fused_scores.get(index, 0.0) + dense_weight / (fusion_k + rank)
        for rank, index in enumerate(lexical_indices.tolist(), start=1):
            fused_scores[index] = fused_scores.get(index, 0.0) + lexical_weight / (fusion_k + rank)

        ranked_indices = sorted(
            fused_scores,
            key=lambda idx: (
                -fused_scores[idx],
                idx,
            ),
        )
        base_scores = fused_scores

    if normalized_reranker == "token_overlap":
        reranked_indices, final_scores = _apply_token_overlap_reranker(
            query_tokens=query_tokens,
            ranked_indices=ranked_indices,
            snippets=snippets,
            base_scores=base_scores,
            reranker_weight=reranker_weight,
        )
    else:
        reranked_indices = ranked_indices
        final_scores = base_scores

    return _hits_from_ranked_indices(reranked_indices[:top_k], snippets, final_scores)


def dump_search_results(path: Path, hits: Sequence[DenseRetrievalHit]) -> None:
    payload = [
        {
            "score": hit.score,
            "snippet": hit.snippet.model_dump(mode="json", exclude_none=True),
        }
        for hit in hits
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
