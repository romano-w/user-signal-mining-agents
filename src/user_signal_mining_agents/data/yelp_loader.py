from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


RESTAURANT_CATEGORY = "restaurants"


@dataclass(slots=True, frozen=True)
class YelpBusiness:
    business_id: str
    name: str | None
    city: str | None
    state: str | None
    stars: float | None
    review_count: int | None
    categories: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class YelpReview:
    review_id: str
    business_id: str
    stars: float | None
    text: str
    date: str | None


def _iter_json_lines(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue

            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc

            if isinstance(payload, dict):
                yield payload


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def parse_categories(raw_categories: object) -> tuple[str, ...]:
    if not isinstance(raw_categories, str):
        return ()
    return tuple(
        category.strip()
        for category in raw_categories.split(",")
        if category.strip()
    )


def is_restaurant_business(categories: tuple[str, ...]) -> bool:
    normalized = {category.casefold() for category in categories}
    return RESTAURANT_CATEGORY in normalized


def iter_restaurant_businesses(
    businesses_path: Path,
    *,
    allowed_states: set[str] | None = None,
) -> Iterator[YelpBusiness]:
    normalized_states = {state.upper() for state in allowed_states or set()}

    for payload in _iter_json_lines(businesses_path):
        categories = parse_categories(payload.get("categories"))
        if not is_restaurant_business(categories):
            continue

        state = payload.get("state")
        if normalized_states:
            if not isinstance(state, str) or state.upper() not in normalized_states:
                continue

        yield YelpBusiness(
            business_id=str(payload["business_id"]),
            name=payload.get("name") if isinstance(payload.get("name"), str) else None,
            city=payload.get("city") if isinstance(payload.get("city"), str) else None,
            state=state if isinstance(state, str) else None,
            stars=_coerce_float(payload.get("stars")),
            review_count=_coerce_int(payload.get("review_count")),
            categories=categories,
        )


def load_restaurant_business_lookup(
    businesses_path: Path,
    *,
    allowed_states: set[str] | None = None,
) -> dict[str, YelpBusiness]:
    return {
        business.business_id: business
        for business in iter_restaurant_businesses(
            businesses_path,
            allowed_states=allowed_states,
        )
    }


def iter_restaurant_reviews(
    reviews_path: Path,
    restaurant_businesses: dict[str, YelpBusiness],
    *,
    review_limit: int | None = None,
    min_review_characters: int = 60,
    max_reviews_per_business: int | None = None,
) -> Iterator[YelpReview]:
    yielded_reviews = 0
    business_review_counts: dict[str, int] = {}

    for payload in _iter_json_lines(reviews_path):
        business_id = payload.get("business_id")
        if not isinstance(business_id, str) or business_id not in restaurant_businesses:
            continue

        if max_reviews_per_business is not None:
            review_count = business_review_counts.get(business_id, 0)
            if review_count >= max_reviews_per_business:
                continue

        text = payload.get("text")
        if not isinstance(text, str):
            continue

        clean_text = " ".join(text.split())
        if len(clean_text) < min_review_characters:
            continue

        yield YelpReview(
            review_id=str(payload["review_id"]),
            business_id=business_id,
            stars=_coerce_float(payload.get("stars")),
            text=clean_text,
            date=payload.get("date") if isinstance(payload.get("date"), str) else None,
        )

        business_review_counts[business_id] = business_review_counts.get(business_id, 0) + 1
        yielded_reviews += 1

        if review_limit is not None and yielded_reviews >= review_limit:
            return
