# Experiment Log

## 2026-03-14 - Final Analysis & Paper

- Generated final analysis report from 20 current-schema evaluation runs.
- Pipeline wins 17/20 prompts; overall preference 3.50 → 4.65 (+1.15).
- Human annotation (n=20, 1 annotator): 70% exact agreement with LLM judge (κ = 0.34).
- Created conference-style paper at `paper/final_report.md`.
- All 192 tests passing; deprecation warnings resolved.

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

