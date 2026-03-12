# Agent-5 Handoff: Program 9 Robustness Suite

## Branch
`codex/p9-robustness-suite`

## Mission
Create perturbation-based robustness evaluation and release gates.

## Required Outputs
- Robustness case definitions using `RobustnessCase`.
- Perturbation generators (negation, noise, context shifts).
- Robustness runner and summarized report.
- Wire `usm eval-robustness` to run suites.

## Exit Criteria
- Suite can run against at least one prompt subset offline.
- Threshold failures are surfaced as explicit non-zero gate outcomes.
- Tests cover perturbation determinism and threshold logic.
