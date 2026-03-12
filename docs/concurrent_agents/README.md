# Concurrent Research Upgrade Handoffs

This directory contains handoff packets for concurrent agent execution across the research-upgrade branches.

> **Status**: Experimental integration workflow. Default day-to-day development and release flow remains `uv run usm evaluate` + standard CI gates.

## Branches

- `codex/foundation-contracts-gates`
- `codex/p1-multi-source-ingestion`
- `codex/p3-retrieval-stack-v2`
- `codex/p6-multi-judge-panel`
- `codex/p7-failure-taxonomy`
- `codex/p9-robustness-suite`
- `codex/p10-domain-transfer`
- `codex/reco-interfaces-tests-ci-gates`
- `codex/integration-research-upgrades`

## Operating Rules

1. Do not edit GUI workstream files unless explicitly required by your packet.
2. Do not rework in-flight agent-pipeline experiments unless required by your packet.
3. Keep changes branch-local and behind tests.
4. If shared contracts change after freeze, open a small PR back to `codex/foundation-contracts-gates` first.

## Acceptance Gates

1. Branch-ready: unit + integration tests pass, schema contracts validate, and a branch acceptance note is produced.
2. Integration-ready: branch merges cleanly into `codex/integration-research-upgrades` with no contract regressions.
3. Main-ready: full offline suite passes and quality guardrails hold.

Gate policy and command details:

- `docs/concurrent_agents/ci-gate-policy.md`

Standard acceptance note template:

- `docs/concurrent_agents/acceptance-report-template.md`
