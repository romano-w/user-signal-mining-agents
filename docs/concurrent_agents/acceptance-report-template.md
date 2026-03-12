# Branch Acceptance Report Template

Use this template for all concurrent-agent branches before requesting merge into `codex/integration-research-upgrades`.

## 1. Branch Metadata

- Branch: `<branch-name>`
- Agent packet: `<docs/concurrent_agents/agent-*.md>`
- Commit under review: `<git-sha>`
- Report author: `<name>`
- Report date (YYYY-MM-DD): `<date>`

## 2. Scope Delivered

- Implemented outputs:
  - `<item 1>`
  - `<item 2>`
  - `<item 3>`
- Out-of-scope or deferred items:
  - `<item>`

## 3. Gate Results

| Gate | Command | Result | Evidence |
|---|---|---|---|
| Branch-ready | `uv run python scripts/ci/run_research_gate.py --level branch-ready` | `<pass/fail>` | `<artifact path or link>` |
| Integration-ready | `uv run python scripts/ci/run_research_gate.py --level integration-ready` | `<pass/fail/na>` | `<artifact path or link>` |
| Main-ready | `uv run python scripts/ci/run_research_gate.py --level main-ready` | `<pass/fail/na>` | `<artifact path or link>` |

## 4. Contract Compatibility

- Schema contracts validated:
  - `<test references>`
- CLI contract surfaces validated:
  - `<test references>`
- Contract changes introduced after foundation freeze:
  - `<none>` or `<brief description + follow-up PR link>`

## 5. Metric Guardrails

- Metric guardrail command:
  - `uv run python scripts/ci/check_metric_guardrails.py --runs-dir <path>`
- Judge pairs evaluated: `<count>`
- Violations detected: `<none>` or `<list>`

## 6. Risks and Follow-Ups

- Open risks:
  - `<risk>`
- Required follow-up tasks:
  - `<task>`

## 7. Merge Recommendation

- Recommendation: `<ready / blocked / conditional>`
- Conditions (if any):
  - `<condition>`
