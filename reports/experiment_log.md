# Experiment Log

## 2026-03-12 - Evaluation Framework Migration (Full Cutover)

- Decision: move to the new evaluation framework with no backward-compatibility layer.
- Active dimensions:
  - `relevance`
  - `contradiction`
  - `coverage`
  - `distinctiveness`
  - `overall_preference`
- Scope updated:
  - judge scoring schema and prompt contract
  - evaluation runners/reports/gates
  - CLI score tables
  - human-annotation rubric dimensions
  - failure taxonomy categories
  - project docs and rubric references
- Validation status at migration point:
  - `uv run pytest -q` passing (`178 passed`)
