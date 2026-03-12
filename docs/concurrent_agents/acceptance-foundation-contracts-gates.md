# Branch Acceptance Note: Foundation Contracts and Gates

## Branch
`codex/foundation-contracts-gates`

## Status
Branch-ready

## Foundation Commit Hash
`35f00c2cf7058df5a18c8b8174c3a3596207c3e3`

## Scope Verified
- Shared schema contracts are present for:
  - `DatasetRecord`
  - `SnippetProvenance`
  - `DatasetSnapshotManifest`
  - `ExperimentManifest`
  - `JudgePanelResult`
  - `MetricWithCI`
  - `SignificanceResult`
  - `FailureTag`
  - `RobustnessCase`
  - `DomainPack`
- CLI scaffold commands are present and wired:
  - `ingest`
  - `snapshot-data`
  - `eval-retrieval`
  - `eval-robustness`
  - `compare-runs`
- Placeholder handlers return deterministic scaffold payloads (`status=foundation-placeholder`).

## Gate Test Evidence
Command run:
`python -m pytest tests/unit/test_foundation_contracts.py tests/unit/test_cli.py -q`

Result:
- `16 passed, 1 warning`
- Warning: `datetime.utcnow()` deprecation from Pydantic model default factories.

## Notes
- No additional code changes were required during this pass; branch contents already satisfied Agent-0 required outputs.
