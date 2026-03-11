from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from ..schemas import EvidenceSnippet
from .yelp_loader import YelpBusiness, YelpReview


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(text) if sentence.strip()]
    return sentences or [text.strip()]


def build_sentence_windows(
    text: str,
    *,
    sentence_window: int = 2,
    sentence_stride: int = 1,
    max_chunks: int = 3,
    min_chunk_characters: int = 60,
) -> list[str]:
    if sentence_window <= 0:
        raise ValueError("sentence_window must be positive.")
    if sentence_stride <= 0:
        raise ValueError("sentence_stride must be positive.")
    if max_chunks <= 0:
        raise ValueError("max_chunks must be positive.")

    cleaned_text = " ".join(text.split())
    if len(cleaned_text) < min_chunk_characters:
        return []

    sentences = split_sentences(cleaned_text)
    if len(sentences) <= sentence_window:
        return [cleaned_text]

    windows: list[str] = []
    seen_text: set[str] = set()

    for start_idx in range(0, len(sentences), sentence_stride):
        window = sentences[start_idx : start_idx + sentence_window]
        if not window:
            continue

        window_text = " ".join(window).strip()
        if len(window_text) < min_chunk_characters or window_text in seen_text:
            continue

        windows.append(window_text)
        seen_text.add(window_text)

        if len(window_text) == len(cleaned_text) or len(windows) >= max_chunks:
            break

    if not windows:
        windows.append(cleaned_text)

    return windows


def chunk_review(
    review: YelpReview,
    business: YelpBusiness,
    *,
    sentence_window: int = 2,
    sentence_stride: int = 1,
    max_chunks: int = 3,
    min_chunk_characters: int = 60,
) -> list[EvidenceSnippet]:
    windows = build_sentence_windows(
        review.text,
        sentence_window=sentence_window,
        sentence_stride=sentence_stride,
        max_chunks=max_chunks,
        min_chunk_characters=min_chunk_characters,
    )

    return [
        EvidenceSnippet(
            snippet_id=f"{review.review_id}::chunk::{chunk_index}",
            review_id=review.review_id,
            business_id=review.business_id,
            business_name=business.name,
            categories=list(business.categories),
            text=window,
            stars=review.stars,
            city=business.city,
            state=business.state,
            review_date=review.date,
            chunk_index=chunk_index,
        )
        for chunk_index, window in enumerate(windows)
    ]


def iter_chunked_reviews(
    reviews: Iterable[YelpReview],
    restaurant_businesses: dict[str, YelpBusiness],
    *,
    sentence_window: int = 2,
    sentence_stride: int = 1,
    max_chunks: int = 3,
    min_chunk_characters: int = 60,
) -> Iterator[EvidenceSnippet]:
    for review in reviews:
        business = restaurant_businesses.get(review.business_id)
        if business is None:
            continue

        yield from chunk_review(
            review,
            business,
            sentence_window=sentence_window,
            sentence_stride=sentence_stride,
            max_chunks=max_chunks,
            min_chunk_characters=min_chunk_characters,
        )


def write_snippets_jsonl(snippets: Iterable[EvidenceSnippet], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for snippet in snippets:
            handle.write(snippet.model_dump_json(exclude_none=True) + "\n")
            count += 1

    return count


def iter_snippets_jsonl(path: Path) -> Iterator[EvidenceSnippet]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue

            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc

            yield EvidenceSnippet.model_validate(payload)


def load_snippets_jsonl(path: Path) -> list[EvidenceSnippet]:
    return list(iter_snippets_jsonl(path))
