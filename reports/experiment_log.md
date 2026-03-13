# Experiment Log

## 2026-03-13 - Rubric Simplification

- Decision: simplify the evaluation design to four final signals.
- Active dimensions:
  - `relevance`
  - `groundedness`
  - `distinctiveness`
  - `overall_preference`
- Human annotation remains split into per-system rubric scores plus a pairwise `overall_preference` choice.

## 2026-03-12 - Evaluation Framework Migration (Full Cutover)

- Decision: move to the new evaluation framework with no backward-compatibility layer.
- Active dimensions at that point:
  - `relevance`
  - `groundedness`
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

