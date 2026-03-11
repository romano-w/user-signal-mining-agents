from __future__ import annotations

import argparse
import json
from pathlib import Path

from user_signal_mining_agents.config import ensure_scaffold_directories, get_settings
from user_signal_mining_agents.data.chunking import chunk_review
from user_signal_mining_agents.data.yelp_loader import (
    load_restaurant_business_lookup,
    iter_restaurant_reviews,
)


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Build a chunked Yelp restaurant subset for retrieval experiments.",
    )
    parser.add_argument(
        "--businesses-path",
        type=Path,
        default=settings.yelp_businesses_path,
        help="Path to the Yelp business JSON lines file.",
    )
    parser.add_argument(
        "--reviews-path",
        type=Path,
        default=settings.yelp_reviews_path,
        help="Path to the Yelp review JSON lines file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=settings.working_subset_path,
        help="Where to write the chunked JSONL snippet file.",
    )
    parser.add_argument(
        "--review-limit",
        type=int,
        default=settings.restaurant_review_limit,
        help="Optional global cap on restaurant reviews to process.",
    )
    parser.add_argument(
        "--max-reviews-per-business",
        type=int,
        default=settings.max_reviews_per_business,
        help="Limit reviews per restaurant so chains do not dominate the subset.",
    )
    parser.add_argument(
        "--min-review-characters",
        type=int,
        default=settings.min_review_characters,
        help="Skip very short reviews before chunking.",
    )
    parser.add_argument(
        "--sentence-window",
        type=int,
        default=settings.chunk_sentence_window,
        help="How many sentences to place in each snippet window.",
    )
    parser.add_argument(
        "--sentence-stride",
        type=int,
        default=settings.chunk_sentence_stride,
        help="Sentence stride between windows from the same review.",
    )
    parser.add_argument(
        "--max-chunks-per-review",
        type=int,
        default=settings.max_chunks_per_review,
        help="Maximum number of snippet windows to keep per review.",
    )
    return parser


def main() -> int:
    settings = get_settings()
    ensure_scaffold_directories(settings)
    args = build_parser().parse_args()

    for path in (args.businesses_path, args.reviews_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required Yelp file not found: {path}\n"
                "Run `uv run usm fetch-yelp-dataset` first, or place the extracted Yelp JSON files "
                "under the configured dataset directory."
            )

    print("Loading restaurant businesses...")
    restaurant_businesses = load_restaurant_business_lookup(args.businesses_path)
    print(f"Loaded {len(restaurant_businesses):,} restaurant businesses.")

    output_path: Path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reviews_written = 0
    snippets_written = 0
    touched_businesses: set[str] = set()

    with output_path.open("w", encoding="utf-8") as handle:
        for review in iter_restaurant_reviews(
            args.reviews_path,
            restaurant_businesses,
            review_limit=args.review_limit,
            min_review_characters=args.min_review_characters,
            max_reviews_per_business=args.max_reviews_per_business,
        ):
            snippets = chunk_review(
                review,
                restaurant_businesses[review.business_id],
                sentence_window=args.sentence_window,
                sentence_stride=args.sentence_stride,
                max_chunks=args.max_chunks_per_review,
                min_chunk_characters=args.min_review_characters,
            )
            if not snippets:
                continue

            reviews_written += 1
            touched_businesses.add(review.business_id)

            for snippet in snippets:
                handle.write(snippet.model_dump_json(exclude_none=True) + "\n")
                snippets_written += 1

            if reviews_written % 5000 == 0:
                print(
                    f"Processed {reviews_written:,} reviews into {snippets_written:,} snippets..."
                )

    stats = {
        "restaurant_businesses_loaded": len(restaurant_businesses),
        "restaurant_businesses_with_output": len(touched_businesses),
        "reviews_written": reviews_written,
        "snippets_written": snippets_written,
        "output_path": str(output_path),
        "review_limit": args.review_limit,
        "max_reviews_per_business": args.max_reviews_per_business,
        "sentence_window": args.sentence_window,
        "sentence_stride": args.sentence_stride,
        "max_chunks_per_review": args.max_chunks_per_review,
        "min_review_characters": args.min_review_characters,
    }
    stats_path = output_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Wrote {snippets_written:,} snippets from {reviews_written:,} reviews.")
    print(f"Saved subset to {output_path}")
    print(f"Saved build stats to {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
