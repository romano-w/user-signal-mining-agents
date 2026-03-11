from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from user_signal_mining_agents.retrieval import index as retrieval_index
from user_signal_mining_agents.retrieval.index import DenseRetrievalHit
from user_signal_mining_agents.schemas import EvidenceSnippet


def _snippet(idx: int, text: str = "text") -> EvidenceSnippet:
    return EvidenceSnippet(
        snippet_id=f"s{idx}",
        review_id=f"r{idx}",
        business_id="b1",
        text=f"{text}-{idx}",
    )


def test_resolve_embedding_device_handles_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retrieval_index.torch.cuda, "is_available", lambda: False)
    assert retrieval_index.resolve_embedding_device(None) == "cpu"
    assert retrieval_index.resolve_embedding_device("cuda") == "cpu"

    monkeypatch.setattr(retrieval_index.torch.cuda, "is_available", lambda: True)
    assert retrieval_index.resolve_embedding_device("cuda") == "cuda"


def test_build_dense_index_rejects_empty_snippets(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Cannot build an index without snippets"):
        retrieval_index.build_dense_index(
            [],
            index_dir=tmp_path,
            embedding_model="stub-model",
        )


def test_build_dense_index_writes_metadata_and_arrays(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snippets = [_snippet(1), _snippet(2)]

    class _FakeModel:
        def encode(self, texts, **_kwargs):
            assert texts == ["text-1", "text-2"]
            return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    monkeypatch.setattr(retrieval_index, "load_embedding_model", lambda *_args, **_kwargs: _FakeModel())

    metadata = retrieval_index.build_dense_index(
        snippets,
        index_dir=tmp_path,
        embedding_model="fake-model",
        batch_size=2,
        device="cpu",
    )

    assert metadata.embedding_model == "fake-model"
    assert metadata.snippet_count == 2
    assert (tmp_path / retrieval_index.EMBEDDINGS_FILENAME).exists()
    assert (tmp_path / retrieval_index.SNIPPETS_FILENAME).exists()
    assert (tmp_path / retrieval_index.METADATA_FILENAME).exists()


def test_load_dense_index_uses_cache(tmp_path: Path) -> None:
    snippets = [_snippet(1)]
    embeddings = np.array([[0.5, 0.5]], dtype=np.float32)
    metadata = retrieval_index.DenseIndexMetadata(
        embedding_model="m",
        device="cpu",
        snippet_count=1,
        vector_dimension=2,
    )

    (tmp_path / retrieval_index.METADATA_FILENAME).write_text(metadata.model_dump_json(), encoding="utf-8")
    np.save(tmp_path / retrieval_index.EMBEDDINGS_FILENAME, embeddings)
    (tmp_path / retrieval_index.SNIPPETS_FILENAME).write_text(snippets[0].model_dump_json() + "\n", encoding="utf-8")

    first = retrieval_index.load_dense_index(tmp_path)
    # Mutate on disk; cache should still return original object references.
    (tmp_path / retrieval_index.METADATA_FILENAME).write_text(
        retrieval_index.DenseIndexMetadata(
            embedding_model="changed",
            device="cpu",
            snippet_count=1,
            vector_dimension=2,
        ).model_dump_json(),
        encoding="utf-8",
    )
    second = retrieval_index.load_dense_index(tmp_path)

    assert first is second
    assert second[0].embedding_model == "m"


def test_search_dense_index_returns_top_hits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snippets = [_snippet(1), _snippet(2), _snippet(3)]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
        ],
        dtype=np.float32,
    )
    metadata = retrieval_index.DenseIndexMetadata(
        embedding_model="m",
        device="cpu",
        snippet_count=3,
        vector_dimension=2,
    )

    monkeypatch.setattr(retrieval_index, "load_dense_index", lambda _d: (metadata, embeddings, snippets))

    class _FakeModel:
        def encode(self, queries, **_kwargs):
            assert queries == ["slow service"]
            return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(retrieval_index, "load_embedding_model", lambda *_args, **_kwargs: _FakeModel())

    hits = retrieval_index.search_dense_index(
        "slow service",
        index_dir=tmp_path,
        top_k=2,
    )

    assert [hit.snippet.snippet_id for hit in hits] == ["s1", "s3"]
    assert hits[0].score >= hits[1].score


def test_dump_search_results_writes_expected_payload(tmp_path: Path) -> None:
    hits = [DenseRetrievalHit(snippet=_snippet(1), score=0.91)]
    output = tmp_path / "out" / "results.json"

    retrieval_index.dump_search_results(output, hits)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload[0]["score"] == 0.91
    assert payload[0]["snippet"]["snippet_id"] == "s1"
