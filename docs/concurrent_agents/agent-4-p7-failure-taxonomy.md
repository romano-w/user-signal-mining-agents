# Agent-4 Handoff: Program 7 Failure Taxonomy

## Branch
`codex/p7-failure-taxonomy`

## Mission
Automatically classify low-quality outputs into actionable failure tags.

## Required Outputs
- Failure tagging pipeline producing `FailureTag` entries.
- Root-cause report grouped by category/severity.
- Integration with evaluation artifacts.

## Exit Criteria
- Failure tags are generated deterministically from run artifacts.
- Reports summarize top categories with prompt/link references.
- Unit tests cover classification and report generation.
