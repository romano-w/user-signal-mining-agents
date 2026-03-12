"""Dataset loading and preprocessing utilities."""

from .chunking import chunk_review, iter_snippets_jsonl, write_snippets_jsonl
from .fetch_yelp import EXPECTED_YELP_FILES, ensure_yelp_dataset
from .ingestion import build_snapshot, list_adapter_ids, run_ingest
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
    "build_snapshot",
    "chunk_review",
    "ensure_yelp_dataset",
    "iter_restaurant_reviews",
    "iter_snippets_jsonl",
    "list_adapter_ids",
    "load_restaurant_business_lookup",
    "run_ingest",
    "write_snippets_jsonl",
]
