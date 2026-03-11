"""Dataset loading and preprocessing utilities."""

from .chunking import chunk_review, iter_snippets_jsonl, write_snippets_jsonl
from .fetch_yelp import EXPECTED_YELP_FILES, ensure_yelp_dataset
from .yelp_loader import (
    YelpBusiness,
    YelpReview,
    iter_restaurant_reviews,
    load_restaurant_business_lookup,
)

__all__ = [
    "YelpBusiness",
    "YelpReview",
    "EXPECTED_YELP_FILES",
    "chunk_review",
    "ensure_yelp_dataset",
    "iter_restaurant_reviews",
    "iter_snippets_jsonl",
    "load_restaurant_business_lookup",
    "write_snippets_jsonl",
]
