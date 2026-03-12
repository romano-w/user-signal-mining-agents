# Agent-2 Handoff: Program 3 Retrieval Stack v2

## Branch
`codex/p3-retrieval-stack-v2`

## Mission
Ship hybrid retrieval + reranking with retrieval evaluation metrics.

## Required Outputs
- Lexical retrieval stage (BM25 or equivalent).
- Dense + lexical fusion ranking.
- Optional reranker stage.
- Metrics runner for Recall@K, MRR, nDCG.
- Wire `usm eval-retrieval` to report generator.

## Exit Criteria
- Retrieval pipeline is configurable by settings/CLI.
- Evaluation report artifacts are machine-readable and human-readable.
- Regression tests protect ranking behavior and metrics math.
