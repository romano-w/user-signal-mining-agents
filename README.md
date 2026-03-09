# user-signal-mining-agents

Founder-grounded review mining experiments for the AI Agents W26 final project.

## Current scope

This repository is scaffolded around one narrow experiment:

- Compare a grounded multi-step pipeline against a zero-shot baseline.
- Use a static review corpus, starting with Yelp restaurant data.
- Output 3-5 evidence-backed focus points for each founder prompt.

## Scaffolded structure

```text
.
|-- artifacts/
|-- data/
|   |-- processed/
|   `-- raw/
|-- founder_prompts/
|-- prompts/
`-- src/user_signal_mining_agents/
```

## Quick start

1. Install dependencies with `uv sync`.
2. Copy `.env.example` to `.env` and update values if needed.
3. Run `uv run usm bootstrap` to create the working directories.
4. Run `uv run usm show-config` to inspect resolved settings.
5. Run `uv run usm validate-founder-prompts` to validate the sample founder prompt file.

## What is included so far

- Typed project settings via `pydantic-settings`
- Shared schemas for prompts, evidence, synthesis output, and judge scores
- A small CLI for bootstrapping directories and validating prompt files
- Placeholder prompt templates for intent, baseline, synthesis, and judging

## What comes next

- Yelp filtering and subset-building scripts
- Embedding + FAISS indexing
- Baseline and pipeline orchestration
- Rubric-based evaluation
