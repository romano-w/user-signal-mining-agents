# CI Gate Policy

This policy defines the default checks for branch promotion in the concurrent research-upgrade workflow.

## Gate Levels

### Branch-ready

Purpose: validate local branch quality before merge requests.

Command:

```powershell
uv run python scripts/ci/run_research_gate.py --level branch-ready
```

Required outcomes:

- Contract and CLI regression tests pass.
- Unit offline suite passes.
- Coverage gate (`--cov-fail-under=80`) passes.
- JUnit report written to `artifacts/test-results/branch-ready-junit.xml`.

### Integration-ready

Purpose: validate branch behavior in integration merge flow.

Command:

```powershell
uv run python scripts/ci/run_research_gate.py --level integration-ready
```

Required outcomes:

- All `branch-ready` outcomes pass.
- Offline integration tests pass.
- JUnit report written to `artifacts/test-results/integration-ready-junit.xml`.

### Main-ready

Purpose: enforce final offline release checks.

Command:

```powershell
uv run python scripts/ci/run_research_gate.py --level main-ready
```

Required outcomes:

- All `integration-ready` outcomes pass.
- Full offline suite (`-m "not live"`) passes.
- Metric guardrail check passes.
- JUnit report written to `artifacts/test-results/main-ready-junit.xml`.

## Metric Guardrails

Use `scripts/ci/check_metric_guardrails.py` to detect critical regressions in `judge_baseline.json` vs `judge_pipeline.json` artifacts.

Default thresholds:

- `overall` delta must be >= `-0.30`
- each rubric dimension delta must be >= `-0.40`

Run directly:

```powershell
uv run python scripts/ci/check_metric_guardrails.py --runs-dir artifacts/runs
```

To fail when no judge artifacts exist:

```powershell
uv run python scripts/ci/check_metric_guardrails.py --runs-dir artifacts/runs --require-pairs
```

## Workflow Mapping

- `.github/workflows/pr-tests.yml` runs `branch-ready` on PRs and pushes to `main`.
- `.github/workflows/integration-gates.yml` runs `integration-ready` for `codex/integration-research-upgrades`.
- `.github/workflows/nightly-tests.yml` runs `main-ready` on nightly schedule and manual dispatch.

## Reporting Requirement

Every concurrent branch should publish an acceptance note using:

- `docs/concurrent_agents/acceptance-report-template.md`

The note must include command evidence and whether each gate passed or failed.
