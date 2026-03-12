# Design + Progress Handoff (Nikil)

Date: 2026-03-12 (night handoff for 2026-03-13 morning)

## 1) Objective
We migrated the evaluation framework to the new rubric and removed backward-compatibility assumptions for old rubric fields.

Active evaluation framework:
- `relevance`
- `contradiction`
- `coverage`
- `distinctiveness`
- `overall_preference`

Interpretation used:
- Model/LLM judge: `overall_preference` is a numeric 1-5 score.
- Human annotation: rubric dimensions are scored per-system; overall preference remains pairwise choice (`system_a/system_b/tie`).

## 2) Design Decisions
- Full cutover (no legacy field aliases).
- Keep `overall_preference` as explicit first-class metric in judge outputs and aggregate reporting.
- Use rubric-dimension deltas for diagnostics, but treat `overall_preference` as the top-line quality signal in gates/reports where overall had previously been derived.
- Update all user-facing rubric text to match the new vocabulary.

## 3) Scope Implemented
Core model + judge contract:
- `src/user_signal_mining_agents/schemas.py`
- `src/user_signal_mining_agents/agents/judge.py`
- `prompts/judge.md`

Evaluation pipeline + reporting + gates:
- `src/user_signal_mining_agents/evaluation/gates.py`
- `src/user_signal_mining_agents/evaluation/report.py`
- `src/user_signal_mining_agents/evaluation/runner.py`
- `src/user_signal_mining_agents/evaluation/robustness_runner.py`
- `src/user_signal_mining_agents/evaluation/robustness_report.py`
- `src/user_signal_mining_agents/evaluation/variant_runner.py`
- `src/user_signal_mining_agents/evaluation/prompt_sweep.py`
- `src/user_signal_mining_agents/evaluation/failure_taxonomy.py`
- `src/user_signal_mining_agents/evaluation/sample_annotations.py`

CLI + human annotation surface:
- `src/user_signal_mining_agents/cli.py`
- `src/user_signal_mining_agents/evaluation/human_annotation_gui.py`

Docs/prompts:
- `README.md`
- `prompts/critic.md`
- `reports/experiment_log.md` (running log entry for migration)

Tests updated across:
- judge, evaluation runners/reports/gates, CLI, schema contracts, failure taxonomy, human annotation, integration flow.

## 4) Validation Status
- Full suite run after migration:
  - `uv run pytest -q`
  - Result: `178 passed`

## 5) Current Risk / Notes
- This is a hard cutover. Any external scripts or old artifacts still using legacy keys (`actionability`, `evidence_grounding`, `contradiction_handling`, `non_redundancy`) will fail validation and must be regenerated.
- Existing historical run artifacts may need regeneration before comparative analysis.

## 6) Tomorrow Morning Checklist
1. Pull latest `main`.
2. Regenerate fresh evaluation artifacts:
   - `uv run usm evaluate --no-cache`
3. Regenerate retrieval benchmark artifact (if needed for integration gate):
   - `uv run usm eval-retrieval --label-set <labels.jsonl> --output-dir reports/research_upgrade`
4. Regenerate remaining integration reports and run:
   - `uv run usm integration-gate --reports-dir reports/research_upgrade`
5. Run gates:
   - `uv run python scripts/ci/run_research_gate.py --level branch-ready`
   - `uv run python scripts/ci/run_research_gate.py --level integration-ready`

## 7) Hand-off Summary
The codebase is migrated to the new rubric framework and test-green. Next work should focus on regenerating artifacts under the new schema and validating end-to-end report/gate readiness with fresh runs.
