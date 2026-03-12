# Agent-0 Handoff: Foundation Contracts and Gates

## Branch
`codex/foundation-contracts-gates`

## Mission
Freeze shared interfaces and scaffold command surfaces required by downstream programs.

## Required Outputs
- Add shared schema contracts: `DatasetRecord`, `SnippetProvenance`, `DatasetSnapshotManifest`, `ExperimentManifest`, `JudgePanelResult`, `MetricWithCI`, `SignificanceResult`, `FailureTag`, `RobustnessCase`, `DomainPack`.
- Add CLI scaffold commands: `ingest`, `snapshot-data`, `eval-retrieval`, `eval-robustness`, `compare-runs`.
- Add minimal gate tests for contracts and CLI parsing.

## Constraints
- Keep implementations intentionally placeholder-only for new commands.
- Avoid changing existing behavior for baseline/pipeline workflows.

## Exit Criteria
- New contracts validate through unit tests.
- New commands parse and return deterministic scaffold payloads.
- Foundation commit hash is published for branch fan-out.
