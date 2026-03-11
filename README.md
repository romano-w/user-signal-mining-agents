# User Signal Mining Agents

Founder-grounded review mining framework for the AI Agents W26 final project.

This system compares a **zero-shot baseline** against a **multi-step grounded pipeline** for extracting actionable focus points from customer reviews, scored by an **LLM judge** across 5 rubric dimensions.

## Architecture

```
Founder Prompt ─────────────────────────────────────────────────────────┐
│                                                                       │
├─ Baseline (zero-shot)                                                 │
│   └─ retrieve top-K snippets → LLM → focus points                    │
│                                                                       │
├─ Pipeline (multi-step)                                                │
│   ├─ Intent Agent → decompose into keywords, queries, hypotheses      │
│   ├─ Evidence Filter → multi-query retrieval → deduplicate + rerank   │
│   ├─ Synthesis Agent → grounded focus points with inline quotes       │
│   └─ Evidence Verifier → check for strict quotes and hallucination    │
│                                                                       │
└─ Judge (LLM-as-judge)                                                 │
    ├─ A/B position debiasing (random system assignment)                │
    └─ 5-dimension rubric scoring                                       │
```

### Evaluation Rubric

| Dimension | What it measures |
|---|---|
| **Relevance** | Does the output address the founder's specific intent? |
| **Actionability** | Would the founder know what to investigate next? |
| **Evidence Grounding** | Are claims tied to real customer quotes? |
| **Contradiction Handling** | Does the output acknowledge counter-signals? |
| **Non Redundancy** | Are focus points distinct rather than repetitive? |

### Latest Results

| Dimension | Baseline | Pipeline | Δ |
|---|:---:|:---:|:---:|
| Relevance | 4.70 | 4.50 | -0.20 |
| Actionability | 4.50 | 4.50 | +0.00 |
| Evidence Grounding | 4.70 | 4.70 | +0.00 |
| Contradiction Handling | 4.30 | 4.60 | **+0.30** |
| Non Redundancy | 4.30 | 4.50 | **+0.20** |
| **Overall** | **4.50** | **4.56** | **+0.06** |

> Pipeline edges out baseline primarily through better contradiction handling and non-redundancy, driven by multi-query retrieval producing more diverse evidence.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) package manager
- Python `3.14`
- API keys (at least one):
  - Gemini API key(s) — set `GEMINI_API_KEY_1` (and optionally `GEMINI_API_KEY_2`) in `.env`
  - Or OpenAI API key — set `OPENAI_API_KEY` in `.env`
- HuggingFace token (optional, avoids rate limits) — set `HF_TOKEN` in `.env`
- Disk space for the Yelp dataset (~4.35 GB download, ~8.65 GB extracted)
- Optional: NVIDIA GPU for faster embedding (CUDA 13.0 targeted)

## Quick Start

```powershell
# 1. Install and sync
uv python install 3.14
uv sync

# 2. Configure
Copy-Item .env.example .env
# Edit .env with your API keys

# 3. Bootstrap directories
uv run usm bootstrap

# 4. Prepare data (assumes Yelp tar already at data/raw/Yelp-JSON/yelp_dataset.tar)
uv run usm fetch-yelp-dataset --skip-download
uv run python scripts/build_yelp_subset.py --review-limit 5000 --max-reviews-per-business 50
uv run python scripts/build_index.py

# 5. Run evaluation
uv run usm evaluate
```

## CLI Commands

### Core workflow

| Command | Description |
|---|---|
| `uv run usm evaluate` | Run full evaluation: baseline + pipeline + judge for all prompts |
| `uv run usm evaluate --no-cache` | Re-run everything from scratch (ignores cached results) |
| `uv run usm evaluate --prompt-id <id>` | Evaluate a single prompt |
| `uv run usm sweep` | Run sweep of prompt variants (A/B testing) |
| `uv run usm sweep --prompt-id <id>` | Run sweep on a single prompt |
| `uv run usm run-baseline --prompt-id <id>` | Run baseline only for one prompt |
| `uv run usm run-pipeline --prompt-id <id>` | Run pipeline only for one prompt |

### Setup & utilities

| Command | Description |
|---|---|
| `uv run usm bootstrap` | Create required local directories |
| `uv run usm show-config` | Print resolved settings |
| `uv run usm validate-founder-prompts` | Validate the founder prompt benchmark file |
| `uv run usm search --query "slow service"` | Search the dense index and print top-K results |
| `uv run usm fetch-yelp-dataset` | Download and extract the Yelp dataset |

### Data preparation scripts

| Command | Description |
|---|---|
| `uv run python scripts/build_yelp_subset.py` | Build the processed review snippet JSONL |
| `uv run python scripts/build_index.py` | Build the dense embedding index |

## Configuration

All settings use the `USM_` prefix and can be set via `.env` or environment variables.

### LLM Settings

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY_1` | — | Primary Gemini API key |
| `GEMINI_API_KEY_2` | — | Secondary key (auto-rotated on rate limits) |
| `OPENAI_API_KEY` | — | OpenAI API key (alternative provider) |
| `HF_TOKEN` | — | HuggingFace token for model downloads |
| `USM_LLM_MODEL` | `gemini-3.1-flash-lite-preview` | LLM model for all agents |
| `USM_LLM_PROVIDER` | `gemini` | Provider: `gemini` or `openai` |
| `USM_LLM_TEMPERATURE` | `0.3` | Temperature for LLM generations |

### Retrieval Settings

| Variable | Default | Description |
|---|---|---|
| `USM_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformer model |
| `USM_RETRIEVAL_TOP_K` | `10` | Top-K results per query |
| `USM_SYNTHESIS_EVIDENCE_K` | `15` | Evidence snippets for synthesis |

### Data Settings

| Variable | Default | Description |
|---|---|---|
| `USM_RESTAURANT_REVIEW_LIMIT` | `75000` | Max reviews to process |
| `USM_MAX_REVIEWS_PER_BUSINESS` | `200` | Cap reviews per business |
| `USM_MIN_REVIEW_CHARACTERS` | `60` | Minimum review length |
| `USM_CHUNK_SENTENCE_WINDOW` | `2` | Sentences per snippet chunk |

## Repository Layout

```text
.
├── artifacts/
│   ├── index/                  # Dense embedding index (embeddings.npy, snippets.jsonl)
│   └── runs/                   # Cached evaluation results and reports
│       ├── <prompt-id>/        # Per-prompt baseline.json, pipeline.json, judge.json
│       └── evaluation_report.md
├── data/
│   ├── processed/              # Chunked review subsets
│   └── raw/Yelp-JSON/          # Raw Yelp dataset files
├── founder_prompts/
│   └── restaurants.json        # 10 restaurant founder prompt benchmark
├── prompts/                    # LLM prompt templates
│   ├── baseline.md             # Zero-shot baseline prompt
│   ├── intent.md               # Intent decomposition prompt
│   ├── synthesis.md            # Multi-step synthesis prompt
│   └── judge.md                # LLM judge rubric prompt
├── scripts/                    # Data preparation scripts
├── src/user_signal_mining_agents/
│   ├── agents/                 # Agent implementations
│   │   ├── baseline.py         # Zero-shot baseline agent
│   │   ├── intent.py           # Intent decomposition agent
│   │   ├── evidence_filter.py  # Multi-query retrieval + dedup
│   │   ├── synthesis.py        # Grounded synthesis agent
│   │   ├── pipeline.py         # Pipeline orchestrator
│   │   └── judge.py            # LLM judge (A/B debiased)
│   ├── evaluation/             # Evaluation runner and report generator
│   ├── retrieval/              # Dense embedding index and search
│   ├── data/                   # Dataset loading and chunking
│   ├── cli.py                  # CLI entry point
│   ├── config.py               # Pydantic settings
│   ├── console.py              # Rich terminal output helpers
│   ├── llm_client.py           # LLM client with retry/repair
│   └── schemas.py              # Shared data models
├── pyproject.toml
└── .env.example
```

## Key Design Decisions

- **A/B Position Debiasing**: The judge randomly assigns systems to "System A" and "System B" for each prompt, then maps scores back. This prevents positional bias.
- **JSON Repair + Retry**: The LLM client attempts automatic JSON repair on malformed responses, then retries the call once before failing.
- **Singleton Model Cache**: The embedding model and dense index are loaded once and reused across all search calls, avoiding redundant IO and model initialization.
- **Gemini Key Rotation**: Multiple API keys are automatically rotated when rate limits are hit.
- **Rich CLI**: All terminal output uses the `rich` library with consistent theming and status indicators.

## Troubleshooting

### Missing Yelp JSON files

```powershell
uv run usm fetch-yelp-dataset --skip-download  # if tar exists
uv run usm fetch-yelp-dataset                   # full download
```

### Rate limit errors

Add a second Gemini key to `.env` as `GEMINI_API_KEY_2`. The client auto-rotates between available keys.

### Using CPU-only PyTorch

Update the `torch` and `torchvision` pins in `pyproject.toml` to remove the `+cu130` suffix, or force CPU for indexing:

```powershell
uv run python scripts/build_index.py --device cpu
```

## Data Policy

The Yelp dataset is intended for educational use. Review the [official terms](https://business.yelp.com/data/resources/open-dataset/) before use. Raw data, processed outputs, and generated artifacts are excluded from git.
