# user-signal-mining-agents

Founder-grounded review mining experiments for the AI Agents W26 final project.

The current repo is focused on getting the data and retrieval layer in place for one narrow experiment:

- Compare a grounded multi-step pipeline against a zero-shot baseline.
- Use the Yelp Open Dataset as the starting corpus.
- Work from a fixed restaurant founder prompt benchmark.
- Build a local dense retrieval index over grounded review snippets.

## Current status

Implemented now:

- Python package scaffold under `src/user_signal_mining_agents/`
- Shared typed config and schema models
- Yelp dataset fetch/extract command
- Restaurant founder prompt benchmark in `founder_prompts/restaurants.json`
- Yelp subset builder that filters restaurants and chunks reviews into snippets
- Dense local embedding index builder using Sentence Transformers

Not implemented yet:

- Baseline summarizer
- Intent/retrieval/filter/synthesis pipeline
- Evaluation and judging loop

## Prerequisites

You will need:

- `uv`
- Python `3.14`
- Enough disk space for the Yelp dataset:
  - about `4.35 GB` for the download zip
  - about `8.65 GB` once extracted
  - additional room for processed snippets and embedding artifacts
- Optional NVIDIA GPU for faster embedding

Important environment note:

- This repo currently pins `torch==2.10.0+cu130` and `torchvision==0.25.0+cu130`.
- If you are using a different CUDA version or want a CPU-only PyTorch build, update `pyproject.toml` before running `uv sync`.
- If you want to force CPU for indexing, you can still run `uv run python scripts/build_index.py --device cpu`.

## Repo Setup

The commands below assume you are already in the repository root.

### 1. Install Python and sync dependencies

```powershell
uv python install 3.14
uv sync
```

### 2. Create a local environment file

PowerShell:

```powershell
Copy-Item .env.example .env
```

Bash:

```bash
cp .env.example .env
```

### 3. Bootstrap the local directories

```powershell
uv run usm bootstrap
```

### 4. Confirm the resolved config

```powershell
uv run usm show-config
```

### 5. Validate the default founder prompt file

```powershell
uv run usm validate-founder-prompts
```

## Fast Start

If you already have the Yelp tar file in `data/raw/Yelp-JSON/yelp_dataset.tar`, the shortest path to a working local setup is:

```powershell
uv sync
Copy-Item .env.example .env
uv run usm bootstrap
uv run usm fetch-yelp-dataset --skip-download
uv run python scripts/build_yelp_subset.py
uv run python scripts/build_index.py
```

For a smaller smoke test run:

```powershell
uv run usm fetch-yelp-dataset --skip-download
uv run python scripts/build_yelp_subset.py --review-limit 5000 --max-reviews-per-business 50
uv run python scripts/build_index.py --device cpu
```

## Dataset Setup

Official source:

- [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/)

This project expects the Yelp JSON dataset in:

- `data/raw/Yelp-JSON/`

### Option A: download and extract through the CLI

```powershell
uv run usm fetch-yelp-dataset
```

What this does:

- downloads `Yelp-JSON.zip`
- extracts `yelp_dataset.tar`
- extracts the dataset JSON files into `data/raw/Yelp-JSON/`

### Option B: use an existing local tar file

If you already placed the tar file at `data/raw/Yelp-JSON/yelp_dataset.tar`, run:

```powershell
uv run usm fetch-yelp-dataset --skip-download
```

### Force a fresh download or re-extraction

```powershell
uv run usm fetch-yelp-dataset --force-download --force-extract
```

### Expected files after extraction

- `data/raw/Yelp-JSON/yelp_academic_dataset_business.json`
- `data/raw/Yelp-JSON/yelp_academic_dataset_review.json`
- `data/raw/Yelp-JSON/yelp_academic_dataset_tip.json`
- `data/raw/Yelp-JSON/yelp_academic_dataset_user.json`
- `data/raw/Yelp-JSON/yelp_academic_dataset_checkin.json`

## Founder Prompt Benchmark

The default founder prompt file is:

- `founder_prompts/restaurants.json`

To validate it after edits:

```powershell
uv run usm validate-founder-prompts --path founder_prompts/restaurants.json
```

## Build the Yelp Working Subset

The subset builder:

- loads the Yelp business and review JSON files
- keeps only restaurant businesses
- filters short reviews
- caps the number of reviews per business
- chunks each review into small sentence windows, using 2-sentence windows by default
- writes the result as JSONL for later retrieval/indexing

### Default run

```powershell
uv run python scripts/build_yelp_subset.py
```

### Useful example

```powershell
uv run python scripts/build_yelp_subset.py --review-limit 20000 --max-reviews-per-business 100 --min-review-characters 80
```

### See all options

```powershell
uv run python scripts/build_yelp_subset.py --help
```

### Outputs

- `data/processed/restaurant_reviews.jsonl`
- `data/processed/restaurant_reviews.stats.json`

## Build the Dense Retrieval Index

The index builder:

- reads the chunked snippet JSONL file
- embeds snippet text with Sentence Transformers
- normalizes and saves dense vectors locally
- writes snippet metadata alongside the vectors

### Default run

```powershell
uv run python scripts/build_index.py
```

### Force CPU

```powershell
uv run python scripts/build_index.py --device cpu
```

### See all options

```powershell
uv run python scripts/build_index.py --help
```

### Outputs

- `artifacts/index/embeddings.npy`
- `artifacts/index/snippets.jsonl`
- `artifacts/index/metadata.json`

## Command Reference

### CLI commands

- `uv run usm bootstrap`
  - create the default local directories used by the project
- `uv run usm show-config`
  - print the resolved application settings
- `uv run usm validate-founder-prompts`
  - validate the default founder prompt file
- `uv run usm validate-founder-prompts --path <path>`
  - validate a specific founder prompt file
- `uv run usm fetch-yelp-dataset`
  - download and extract the official Yelp JSON dataset
- `uv run usm fetch-yelp-dataset --skip-download`
  - use an existing local `yelp_dataset.tar` and only extract it
- `uv run usm fetch-yelp-dataset --force-download --force-extract`
  - refresh the local Yelp dataset cache and extracted files

### Data preparation scripts

- `uv run python scripts/build_yelp_subset.py`
  - build the processed review snippet file
- `uv run python scripts/build_index.py`
  - build the dense embedding index

## Configuration

Core settings live in:

- `.env`
- `.env.example`
- `src/user_signal_mining_agents/config.py`

Useful environment variables:

- `USM_FOUNDER_PROMPTS_PATH`
- `USM_YELP_DOWNLOAD_URL`
- `USM_YELP_DATASET_DIR`
- `USM_YELP_DOWNLOAD_ZIP_PATH`
- `USM_YELP_TAR_PATH`
- `USM_YELP_BUSINESSES_PATH`
- `USM_YELP_REVIEWS_PATH`
- `USM_WORKING_SUBSET_PATH`
- `USM_INDEX_DIR`
- `USM_RUN_ARTIFACTS_DIR`
- `USM_RESTAURANT_REVIEW_LIMIT`
- `USM_MAX_REVIEWS_PER_BUSINESS`
- `USM_MIN_REVIEW_CHARACTERS`
- `USM_CHUNK_SENTENCE_WINDOW`
- `USM_CHUNK_SENTENCE_STRIDE`
- `USM_MAX_CHUNKS_PER_REVIEW`
- `USM_EMBEDDING_MODEL`
- `USM_EMBEDDING_BATCH_SIZE`

To inspect the active resolved values:

```powershell
uv run usm show-config
```

## Repository Layout

```text
.
|-- artifacts/                  # generated indexes and future run outputs
|-- data/
|   |-- processed/              # chunked review subset outputs
|   `-- raw/
|       `-- Yelp-JSON/          # downloaded and extracted Yelp dataset files
|-- founder_prompts/            # benchmark founder prompt sets
|-- prompts/                    # prompt templates for later pipeline stages
|-- scripts/                    # runnable data and indexing entrypoints
`-- src/user_signal_mining_agents/
    |-- data/                   # dataset loading, extraction, and chunking
    `-- retrieval/              # embedding and dense retrieval helpers
```

## Troubleshooting

### Missing Yelp JSON files

If `build_yelp_subset.py` complains that the business or review files do not exist, run:

```powershell
uv run usm fetch-yelp-dataset
```

Or, if the tar is already present:

```powershell
uv run usm fetch-yelp-dataset --skip-download
```

### First import feels slow

The first import of `sentence-transformers`, `transformers`, and PyTorch can take a while. This is normal, especially on the first run after `uv sync`.

### Need a smaller local test

Start with a lower `--review-limit` for the subset builder:

```powershell
uv run python scripts/build_yelp_subset.py --review-limit 5000 --max-reviews-per-business 50
```

### Using a different PyTorch build

This repo currently targets CUDA 13.0. If that does not match your machine, update the PyTorch dependency pins in `pyproject.toml` before re-running `uv sync`.

## Data Policy Notes

- The Yelp dataset is intended for educational use; review the official terms before using it.
- Raw dataset files, processed subset outputs, and generated artifacts are intentionally kept out of git.

## Next Planned Steps

- Retrieval reranking and evidence filtering
- Baseline summarizer
- Multi-step founder-grounded pipeline
- Evaluation and judging loop
