# Agent-3 Handoff: Program 6 Multi-Judge Panel

## Branch
`codex/p6-multi-judge-panel`

## Mission
Evolve judge from single score to panel-based aggregated scoring with confidence and significance outputs.

## Required Outputs
- Multi-judge runner with deterministic mapping to system variants.
- `JudgePanelResult` generation.
- Confidence interval computation in `MetricWithCI`.
- Statistical comparison output in `SignificanceResult`.
- Reporting integration for panel outputs.

## Exit Criteria
- Existing evaluation still works with panel mode on/off.
- Reports include panel size and confidence context.
- Tests cover aggregation, CI, and significance calculations.
