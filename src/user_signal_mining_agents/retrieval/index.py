from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import os
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


def search_dense_index(
    query: str,
    *,
    index_dir: Path,
    embedding_model: str | None = None,
    top_k: int = 10,
    device: str | None = None,
) -> list[DenseRetrievalHit]:
    metadata, embeddings, snippets = load_dense_index(index_dir)
    model_name = embedding_model or metadata.embedding_model
    model = load_embedding_model(model_name, device=device or metadata.device)

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)[0]

    scores = embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]

    hits: list[DenseRetrievalHit] = []
    for index in top_indices:
        snippet = snippets[int(index)].model_copy(
            update={"relevance_score": float(scores[index])}
        )
        hits.append(DenseRetrievalHit(snippet=snippet, score=float(scores[index])))

    return hits


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
