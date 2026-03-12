# Integration Handoff: Event-Driven Merge Branch

## Branch

`codex/integration-research-upgrades`

## Purpose

Aggregate branch-ready program outputs and enforce integration gates.

## Merge Order Policy

1. Merge branches that only add standalone capabilities first.
2. Merge branches that consume shared interfaces next.
3. Resolve contract conflicts by backporting fixes to foundation, then rebase remaining branches.

## Readiness Workflow

1. Branch owner runs `branch-ready` gate and fills the acceptance report template.
2. Integrator merges into `codex/integration-research-upgrades` and runs `integration-ready` gate.
3. Before main promotion, integrator runs `main-ready` gate and confirms metric guardrails.

## Required Checks Before Main

- Full unit and integration suite passes.
- Schema compatibility checks pass.
- Retrieval, robustness, and domain transfer reports generated.
- No blocked high-severity failure tags in release gate summary.

## Standard Commands

```powershell
uv run python scripts/ci/run_research_gate.py --level branch-ready
uv run python scripts/ci/run_research_gate.py --level integration-ready
uv run python scripts/ci/run_research_gate.py --level main-ready
```

Metric guardrail check (optional standalone):

```powershell
uv run python scripts/ci/check_metric_guardrails.py --runs-dir artifacts/runs
```

## Reporting

Use `docs/concurrent_agents/acceptance-report-template.md` for all branch acceptance notes.
