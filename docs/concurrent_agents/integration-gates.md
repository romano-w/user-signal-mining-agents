# Integration Handoff: Event-Driven Merge Branch

## Branch
`codex/integration-research-upgrades`

## Purpose
Aggregate branch-ready program outputs and enforce integration gates.

## Merge Order Policy
1. Merge branches that only add standalone capabilities first.
2. Merge branches that consume shared interfaces next.
3. Resolve contract conflicts by backporting fixes to foundation, then rebase remaining branches.

## Required Checks Before Main
- Full unit and integration suite passes.
- Schema compatibility checks pass.
- Retrieval, robustness, and domain transfer reports generated.
- No blocked high-severity failure tags in release gate summary.
