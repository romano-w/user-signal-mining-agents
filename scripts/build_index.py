from __future__ import annotations

import argparse
from pathlib import Path

from user_signal_mining_agents.config import ensure_scaffold_directories, get_settings
from user_signal_mining_agents.retrieval.index import build_dense_index_from_jsonl


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Build a dense retrieval index from chunked review snippets.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=settings.working_subset_path,
        help="Path to the chunked snippet JSONL file.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=settings.index_dir,
        help="Directory where index artifacts should be saved.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=settings.embedding_model,
        help="Sentence Transformers model name to use for indexing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.embedding_batch_size,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional override for the embedding device, for example 'cuda' or 'cpu'.",
    )
    return parser


def main() -> int:
    settings = get_settings()
    ensure_scaffold_directories(settings)
    args = build_parser().parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(f"Snippet file not found: {args.input_path}")

    metadata = build_dense_index_from_jsonl(
        args.input_path,
        index_dir=args.index_dir,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        device=args.device,
    )

    print(
        "Built dense index with "
        f"{metadata.snippet_count:,} snippets "
        f"at dimension {metadata.vector_dimension} "
        f"using {metadata.embedding_model} on {metadata.device}."
    )
    print(f"Index artifacts saved to {args.index_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
