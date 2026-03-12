# Known Limitations and Follow-Ups

This file tracks first-iteration gaps discovered during integration hardening. Keep this list current as issues are closed or new blindspots are found.

## Current Limitations

| Area | Limitation | Impact | Suggested Follow-Up |
|---|---|---|---|
| Dataset coverage | Primary validation still leans on Yelp-derived restaurant data. | Retrieval and scoring behavior may not generalize across domains. | Add additional public datasets (app reviews, support tickets, ecommerce Q/A) and rerun retrieval/robustness benchmarks. |
| Retrieval evaluation | Retrieval labels are still sparse and mostly synthetic in tests. | Hybrid mode quality may be overestimated outside curated cases. | Expand human-validated relevance labels and track trendlines for Recall@K/MRR/nDCG per domain. |
| Robustness suite breadth | Perturbation suite focuses on a core adversarial set. | Some realistic failure patterns may bypass current gates. | Add spelling-noise, multilingual, negation-heavy, and long-context perturbation families. |
| Judge reliability | LLM-as-judge remains sensitive to prompt framing and model version drift. | Absolute score values can shift even when system quality is constant. | Add periodic human adjudication sampling and calibrate judge outputs against reviewer consensus. |
| Domain transfer | Non-restaurant domain packs are present but lightly exercised. | Cross-domain confidence is lower than restaurant baseline. | Add larger prompt sets for SaaS/ecommerce and compare zero-shot vs adapted prompts. |
| Operational complexity | Concurrent branch integration introduces more contract coordination overhead. | Higher merge/conflict risk when contracts move mid-stream. | Keep contract freeze strict and require schema compatibility checks in every branch-ready gate. |

## Stability-First Rules (Current)

1. Keep `usm evaluate` as the default execution path for regular runs.
2. Keep retrieval mode default as `hybrid` unless a blocking regression is observed.
3. Treat concurrent-agent procedures as experimental until additional release cycles complete.
4. Require `pytest -m "not live"` to pass before main promotion.

## Next Research Candidates

1. Add new datasets beyond Yelp (mobile app reviews, support tickets, product feedback forums).
2. Add reranker alternatives beyond token overlap and benchmark against current hybrid baseline.
3. Add longitudinal metric dashboards for retrieval and robustness regression trends.
