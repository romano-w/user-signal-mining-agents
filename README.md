# User Signal Mining Agents

Founder-grounded review mining framework for the AI Agents W26 final project.

This system compares a **zero-shot baseline** against a **multi-step grounded pipeline** for extracting actionable focus points from customer reviews, scored by an **LLM judge** across 4 rubric dimensions.

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
    └─ 4-dimension rubric scoring                                       │
```

### Evaluation Rubric

| Dimension | What it measures |
|---|---|
| **Relevance** | Does the output address the founder's specific intent? |
| **Groundedness** | Are focus points traceable to specific evidence snippets rather than generic claims? |
| **Distinctiveness** | Are focus points distinct rather than repetitive? |
| **Overall Preference** | All-things-considered, how preferable is this output for founder decision-making? |

### Headline Metric

`overall_preference` is the top-line comparison metric. `relevance` is the floor, `groundedness` is the key differentiator, and `distinctiveness` guards against repetitive focus points. For run-specific numbers, see the generated evaluation artifacts under `artifacts/runs/`.

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
| `uv run usm evaluate --domain <id,id>` | Evaluate selected domains from the domain-pack registry |
| `uv run usm evaluate --judge-panel-size 3` | Run evaluation with deterministic 3-judge panel aggregation + confidence stats |
| `uv run usm sweep` | Run sweep of prompt variants (A/B testing) |
| `uv run usm sweep --prompt-id <id>` | Run sweep on a single prompt |
| `uv run usm run-baseline --prompt-id <id>` | Run baseline only for one prompt |
| `uv run usm run-baseline --domain <id,id>` | Run baseline for selected domains |
| `uv run usm run-pipeline --prompt-id <id>` | Run pipeline only for one prompt |
| `uv run usm run-pipeline --domain <id,id>` | Run pipeline for selected domains |
| `uv run usm list-variants` | List available experimental pipeline variants |
| `uv run usm run-variant --variant <id> --prompt-id <id>` | Run one experimental variant for one prompt |
| `uv run usm run-variant --variant <id> --domain <id,id>` | Run one variant across selected domains |
| `uv run usm evaluate-variants --variants <id,id>` | Compare selected variants against control pipeline |
| `uv run usm evaluate-variants --variants <id,id> --domain <id,id>` | Variant comparison on selected domains |
| `uv run usm ingest --adapter yelp` | Ingest one source adapter and emit normalized records |
| `uv run usm snapshot-data --dataset-id default` | Create an immutable dataset snapshot manifest from ingested records |
| `uv run usm eval-retrieval --label-set <path>` | Run retrieval metrics and produce JSON/Markdown retrieval reports |
| `uv run usm eval-robustness --suite adversarial_core --prompt-id <id>` | Run perturbation robustness gates for a prompt subset |
| `uv run usm compare-runs --run-a <id> --run-b <id>` | Compare two experiment manifests (foundation contract surface) |

### Setup & utilities

| Command | Description |
|---|---|
| `uv run usm bootstrap` | Create required local directories |
| `uv run usm show-config` | Print resolved settings |
| `uv run usm validate-founder-prompts` | Validate domain packs + enabled founder prompt files (or one file with `--path`) |
| `uv run usm search --query "slow service"` | Search the dense index and print top-K results |
| `uv run usm fetch-yelp-dataset` | Download and extract the Yelp dataset |
| `uv run usm annotate-human` | Launch the local human-annotation GUI for blinded A/B scoring |
| `uv run usm integration-gate --reports-dir reports/research_upgrade` | Evaluate integration readiness report artifacts and return pass/fail exit code |

### Data preparation scripts

| Command | Description |
|---|---|
| `uv run python scripts/build_yelp_subset.py` | Build the processed review snippet JSONL |
| `uv run python scripts/build_index.py` | Build the dense embedding index |

## Human Annotation GUI

Use the built-in local web app to score sampled tasks from `artifacts/runs/_human_annotations`.

```powershell
# Default directory: artifacts/runs/_human_annotations
uv run usm annotate-human --annotator-id reviewer_01

# Optional: custom folder / host / port
uv run usm annotate-human --tasks-dir artifacts/runs/_human_annotations --host 127.0.0.1 --port 8765
```

What it provides:
- Blinded side-by-side System A vs System B focus points
- 1-5 scoring on relevance, groundedness, and distinctiveness
- Overall preference + difficulty rating
- Per-annotator autosave to `artifacts/runs/_human_annotations/_results/<annotator_id>/<task_id>.json`
- One-click export of your saved annotations as a single JSON file

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
| `USM_RETRIEVAL_TOP_K` | `50` | Top-K results per query |
| `USM_RETRIEVAL_MODE` | `hybrid` | Retrieval mode: `dense`, `lexical`, or `hybrid` |
| `USM_RETRIEVAL_DENSE_WEIGHT` | `1.0` | Dense reciprocal-rank-fusion weight |
| `USM_RETRIEVAL_LEXICAL_WEIGHT` | `1.0` | Lexical reciprocal-rank-fusion weight |
| `USM_RETRIEVAL_FUSION_K` | `60` | RRF smoothing constant |
| `USM_RETRIEVAL_CANDIDATE_POOL` | `200` | Candidate pool size before final top-k cut |
| `USM_RETRIEVAL_RERANKER` | `none` | Optional reranker stage (`none`, `token_overlap`) |
| `USM_RETRIEVAL_RERANKER_WEIGHT` | `0.25` | Blend weight for reranker contribution |
| `USM_RETRIEVAL_BM25_K1` | `1.5` | BM25 term-frequency saturation |
| `USM_RETRIEVAL_BM25_B` | `0.75` | BM25 length normalization factor |
| `USM_SYNTHESIS_EVIDENCE_K` | `20` | Evidence snippets for synthesis |

### Domain Settings

| Variable | Default | Description |
|---|---|---|
| `USM_DOMAIN_PACKS_PATH` | `founder_prompts/domain_packs.json` | Domain pack registry file (list of `DomainPack`) |
| `USM_ACTIVE_DOMAINS` | `` (empty) | Optional comma-separated default domains (`restaurants,saas`) |

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
│   ├── domain_packs.json       # DomainPack registry for transfer evaluation
│   ├── restaurants.json        # Restaurant founder prompts
│   ├── saas.json               # SaaS founder prompts
│   └── ecommerce.json          # Ecommerce founder prompts
├── prompts/                    # LLM prompt templates
│   ├── baseline.md             # Zero-shot baseline prompt
│   ├── intent.md               # Intent decomposition prompt
│   ├── synthesis.md            # Multi-step synthesis prompt
│   ├── query_planner.md        # Retrieval query expansion prompt
│   ├── counterevidence_miner.md # Contradiction query generation prompt
│   ├── critic.md               # Draft critique prompt
│   ├── refiner.md              # Critique-driven rewrite prompt
│   └── judge.md                # LLM judge rubric prompt
├── scripts/                    # Data preparation scripts
├── src/user_signal_mining_agents/
│   ├── agents/                 # Agent implementations
│   │   ├── baseline.py         # Zero-shot baseline agent
│   │   ├── intent.py           # Intent decomposition agent
│   │   ├── evidence_filter.py  # Multi-query retrieval + dedup
│   │   ├── synthesis.py        # Grounded synthesis agent
│   │   ├── query_planner.py    # Query expansion agent
│   │   ├── counterevidence_miner.py # Contradiction mining agent
│   │   ├── critic.py           # Post-synthesis critic agent
│   │   ├── refiner.py          # Critique-driven refinement agent
│   │   ├── variant_pipeline.py # Configurable variant DAG runner
│   │   ├── pipeline.py         # Pipeline orchestrator
│   │   └── judge.py            # LLM judge (A/B debiased)
│   ├── evaluation/             # Evaluation runners and report generation
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

## Stability Notes

- Default run path remains `uv run usm evaluate` (baseline + main pipeline). Variant and concurrent-agent flows are optional.
- Retrieval default is `USM_RETRIEVAL_MODE=hybrid`; use `dense` only as a debugging fallback, not as a release default.
- Concurrent-agent process docs under `docs/concurrent_agents/` are integration playbooks and are treated as experimental operations guidance.
- Running limitations and first-iteration follow-ups are tracked in `docs/KNOWN_LIMITATIONS.md`.

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

