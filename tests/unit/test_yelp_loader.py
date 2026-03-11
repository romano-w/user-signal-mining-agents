from __future__ import annotations

from pathlib import Path

import pytest

from user_signal_mining_agents.data.yelp_loader import (
    _coerce_float,
    _coerce_int,
    _iter_json_lines,
    is_restaurant_business,
    iter_restaurant_businesses,
    iter_restaurant_reviews,
    load_restaurant_business_lookup,
    parse_categories,
)


def _write_jsonl(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_categories_handles_non_string() -> None:
    assert parse_categories(None) == ()
    assert parse_categories("Restaurants, Italian") == ("Restaurants", "Italian")


def test_is_restaurant_business_casefold() -> None:
    assert is_restaurant_business(("RESTAURANTS", "Italian")) is True
    assert is_restaurant_business(("Shopping",)) is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1.0),
        (1.5, 1.5),
        (True, None),
        ("1", None),
    ],
)
def test_coerce_float(value: object, expected: float | None) -> None:
    assert _coerce_float(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1),
        (1.5, None),
        (True, None),
        ("1", None),
    ],
)
def test_coerce_int(value: object, expected: int | None) -> None:
    assert _coerce_int(value) == expected


def test_iter_json_lines_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    _write_jsonl(path, ['{"a": 1}', '{bad-json}'])

    it = _iter_json_lines(path)
    assert next(it) == {"a": 1}
    with pytest.raises(ValueError, match="line 2"):
        next(it)


def test_iter_restaurant_businesses_filters_and_state(tmp_path: Path) -> None:
    businesses = tmp_path / "businesses.jsonl"
    _write_jsonl(
        businesses,
        [
            '{"business_id":"b1","name":"One","city":"Austin","state":"TX","stars":4.0,"review_count":10,"categories":"Restaurants, Pizza"}',
            '{"business_id":"b2","name":"Two","city":"Boston","state":"MA","stars":4.0,"review_count":10,"categories":"Restaurants"}',
            '{"business_id":"b3","name":"Three","city":"Austin","state":"TX","stars":4.0,"review_count":10,"categories":"Shopping"}',
        ],
    )

    tx = list(iter_restaurant_businesses(businesses, allowed_states={"tx"}))
    assert [b.business_id for b in tx] == ["b1"]


def test_load_restaurant_business_lookup(tmp_path: Path) -> None:
    businesses = tmp_path / "businesses.jsonl"
    _write_jsonl(
        businesses,
        ['{"business_id":"b1","name":"One","city":"Austin","state":"TX","stars":4.0,"review_count":10,"categories":"Restaurants"}'],
    )

    lookup = load_restaurant_business_lookup(businesses)
    assert set(lookup) == {"b1"}


def test_iter_restaurant_reviews_applies_limits(tmp_path: Path) -> None:
    businesses = tmp_path / "businesses.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    _write_jsonl(
        businesses,
        ['{"business_id":"b1","name":"One","city":"Austin","state":"TX","stars":4.0,"review_count":10,"categories":"Restaurants"}'],
    )
    _write_jsonl(
        reviews,
        [
            '{"review_id":"r1","business_id":"b1","stars":1,"text":"This review is definitely long enough to be included in output.","date":"2024-01-01"}',
            '{"review_id":"r2","business_id":"b1","stars":2,"text":"This second review is also long enough and valid for output.","date":"2024-01-02"}',
            '{"review_id":"r3","business_id":"b2","stars":2,"text":"Unknown business should be skipped though long enough text.","date":"2024-01-03"}',
        ],
    )

    lookup = load_restaurant_business_lookup(businesses)
    limited = list(
        iter_restaurant_reviews(
            reviews,
            lookup,
            review_limit=1,
            min_review_characters=20,
            max_reviews_per_business=1,
        )
    )

    assert len(limited) == 1
    assert limited[0].review_id == "r1"


def test_iter_restaurant_reviews_skips_short_or_non_text(tmp_path: Path) -> None:
    businesses = tmp_path / "businesses.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    _write_jsonl(
        businesses,
        ['{"business_id":"b1","name":"One","city":"Austin","state":"TX","stars":4.0,"review_count":10,"categories":"Restaurants"}'],
    )
    _write_jsonl(
        reviews,
        [
            '{"review_id":"r1","business_id":"b1","stars":1,"text":"short","date":"2024-01-01"}',
            '{"review_id":"r2","business_id":"b1","stars":2,"text":12,"date":"2024-01-02"}',
            '{"review_id":"r3","business_id":"b1","stars":3,"text":"This one is long enough to pass the filter for sure.","date":"2024-01-03"}',
        ],
    )

    lookup = load_restaurant_business_lookup(businesses)
    reviews_out = list(iter_restaurant_reviews(reviews, lookup, min_review_characters=20))

    assert [r.review_id for r in reviews_out] == ["r3"]
