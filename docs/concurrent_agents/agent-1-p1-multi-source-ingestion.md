# Agent-1 Handoff: Program 1 Multi-Source Ingestion

## Branch
`codex/p1-multi-source-ingestion`

## Mission
Implement pluggable ingestion adapters and snapshot lineage generation.

## Required Outputs
- Adapter interface and registry.
- Yelp adapter plus `app_reviews` and `support_tickets` adapters.
- Normalized output in `DatasetRecord` schema.
- Snapshot manifest generation via `DatasetSnapshotManifest`.
- Wire `usm ingest` and `usm snapshot-data` to real implementations.

## Exit Criteria
- Adapter selection is CLI-driven.
- Snapshot files include checksum + record count + source manifests.
- Unit and integration tests cover all adapters.
