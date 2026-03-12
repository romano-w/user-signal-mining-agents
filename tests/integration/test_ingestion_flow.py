from __future__ import annotations

import json
from pathlib import Path

import pytest

from user_signal_mining_agents.data.ingestion import build_snapshot, run_ingest


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


@pytest.mark.integration
def test_integration_multi_source_ingestion_and_snapshot(tmp_settings, tmp_path: Path) -> None:
    tmp_settings.yelp_businesses_path = Path("tests/fixtures/businesses.jsonl")
    tmp_settings.yelp_reviews_path = Path("tests/fixtures/reviews.jsonl")

    app_source = _write_jsonl(
        tmp_path / "app_reviews.jsonl",
        [{"review_id": "ar-1", "text": "App crashes when attaching photos."}],
    )
    ticket_source = _write_jsonl(
        tmp_path / "support_tickets.jsonl",
        [{"ticket_id": "tk-1", "description": "Webhook retries keep failing with 502."}],
    )

    yelp_result = run_ingest(settings=tmp_settings, adapter_id="yelp")
    app_result = run_ingest(settings=tmp_settings, adapter_id="app_reviews", input_path=app_source)
    ticket_result = run_ingest(settings=tmp_settings, adapter_id="support_tickets", input_path=ticket_source)

    snapshot = build_snapshot(settings=tmp_settings, dataset_id="default")

    assert yelp_result.record_count == 2
    assert app_result.record_count == 1
    assert ticket_result.record_count == 1

    assert snapshot.manifest.dataset_ids == ["app_reviews", "support_tickets", "yelp_restaurants"]
    assert snapshot.manifest.record_count == 4
    assert len(snapshot.manifest.checksum_sha256) == 64
    assert set(snapshot.manifest.source_manifests) == {
        "app_reviews::app_reviews",
        "support_tickets::support_tickets",
        "yelp_restaurants::yelp_businesses",
        "yelp_restaurants::yelp_reviews",
    }
    assert snapshot.manifest_path.exists()
