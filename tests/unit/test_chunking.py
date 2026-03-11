from __future__ import annotations

from pathlib import Path

import pytest

from user_signal_mining_agents.data.chunking import (
    build_sentence_windows,
    chunk_review,
    iter_chunked_reviews,
    iter_snippets_jsonl,
    load_snippets_jsonl,
    split_sentences,
    write_snippets_jsonl,
)
from user_signal_mining_agents.data.yelp_loader import YelpBusiness, YelpReview


def test_split_sentences_handles_punctuation() -> None:
    text = "One. Two? Three!"
    assert split_sentences(text) == ["One.", "Two?", "Three!"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sentence_window": 0}, "sentence_window must be positive."),
        ({"sentence_stride": 0}, "sentence_stride must be positive."),
        ({"max_chunks": 0}, "max_chunks must be positive."),
    ],
)
def test_build_sentence_windows_rejects_invalid_args(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_sentence_windows("A valid sentence. Another valid sentence.", **kwargs)


def test_build_sentence_windows_returns_empty_for_short_text() -> None:
    assert build_sentence_windows("short", min_chunk_characters=20) == []


def test_build_sentence_windows_dedupes_and_caps() -> None:
    text = "A long sentence one. A long sentence two. A long sentence three. A long sentence four."
    windows = build_sentence_windows(
        text,
        sentence_window=2,
        sentence_stride=1,
        max_chunks=2,
        min_chunk_characters=10,
    )
    assert len(windows) == 2
    assert windows[0] != windows[1]


def test_chunk_review_builds_snippets() -> None:
    review = YelpReview(
        review_id="r1",
        business_id="b1",
        stars=4.0,
        text="The service was very slow tonight. Food quality stayed great though.",
        date="2024-01-01",
    )
    business = YelpBusiness(
        business_id="b1",
        name="Diner",
        city="Austin",
        state="TX",
        stars=4.2,
        review_count=10,
        categories=("Restaurants",),
    )

    snippets = chunk_review(review, business, sentence_window=1, max_chunks=3, min_chunk_characters=10)
    assert len(snippets) == 2
    assert snippets[0].snippet_id == "r1::chunk::0"
    assert snippets[0].business_name == "Diner"


def test_iter_chunked_reviews_skips_unknown_business() -> None:
    known_business = YelpBusiness(
        business_id="known",
        name="Known",
        city="Austin",
        state="TX",
        stars=4.0,
        review_count=5,
        categories=("Restaurants",),
    )
    reviews = [
        YelpReview(review_id="r1", business_id="known", stars=3.0, text="Long enough review text for chunking.", date=None),
        YelpReview(review_id="r2", business_id="unknown", stars=3.0, text="Long enough review text for chunking.", date=None),
    ]

    snippets = list(
        iter_chunked_reviews(
            reviews,
            {"known": known_business},
            sentence_window=1,
            min_chunk_characters=10,
        )
    )
    assert len(snippets) >= 1
    assert all(snippet.business_id == "known" for snippet in snippets)


def test_write_and_load_snippets_round_trip(tmp_path: Path) -> None:
    review = YelpReview(review_id="r1", business_id="b1", stars=4.0, text="Sentence one. Sentence two.", date=None)
    business = YelpBusiness(
        business_id="b1",
        name="Diner",
        city="Austin",
        state="TX",
        stars=4.2,
        review_count=10,
        categories=("Restaurants",),
    )
    snippets = chunk_review(review, business, sentence_window=1, min_chunk_characters=5)

    output = tmp_path / "snippets.jsonl"
    written = write_snippets_jsonl(snippets, output)
    loaded = load_snippets_jsonl(output)

    assert written == len(snippets)
    assert [s.snippet_id for s in loaded] == [s.snippet_id for s in snippets]


def test_iter_snippets_jsonl_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"snippet_id":"s1","review_id":"r1","business_id":"b1","text":"hello"}\n{bad json}\n',
        encoding="utf-8",
    )

    iterator = iter_snippets_jsonl(path)
    with pytest.raises(ValueError, match="line 2"):
        list(iterator)
