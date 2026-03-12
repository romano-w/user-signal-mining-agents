from __future__ import annotations

import json
from pathlib import Path

import pytest

from user_signal_mining_agents.data.ingestion import (
    AppReviewsIngestionAdapter,
    SupportTicketsIngestionAdapter,
    YelpIngestionAdapter,
    build_snapshot,
    get_adapter,
    list_adapter_ids,
    run_ingest,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def test_adapter_registry_exposes_expected_adapters() -> None:
    assert list_adapter_ids() == ["app_reviews", "support_tickets", "yelp"]


@pytest.mark.parametrize("adapter_id", ["app_reviews", "support_tickets", "yelp"])
def test_get_adapter_returns_registered_adapter(adapter_id: str) -> None:
    adapter = get_adapter(adapter_id)
    assert adapter.adapter_id == adapter_id


def test_get_adapter_rejects_unknown_adapter() -> None:
    with pytest.raises(ValueError, match="Unknown adapter"):
        get_adapter("unknown")


def test_app_reviews_adapter_normalizes_jsonl_rows(tmp_settings, tmp_path: Path) -> None:
    source = _write_jsonl(
        tmp_path / "app_reviews.jsonl",
        [
            {
                "review_id": "ar-1",
                "text": " Great release, onboarding is smooth. ",
                "rating": 5,
                "platform": "ios",
            },
            {
                "id": "ar-2",
                "review_text": "Sync can be flaky after updates.",
                "version": "2.1.0",
            },
            {
                "id": "ar-3",
                "rating": 3,
            },
        ],
    )

    records = AppReviewsIngestionAdapter().ingest(tmp_settings, input_path=source)

    assert [record.record_id for record in records] == ["ar-1", "ar-2"]
    assert all(record.dataset_id == "app_reviews" for record in records)
    assert records[0].source_type == "app_review"
    assert records[0].text == "Great release, onboarding is smooth."
    assert records[0].metadata == {"platform": "ios", "rating": 5}
    assert records[1].metadata == {"version": "2.1.0"}


def test_support_tickets_adapter_normalizes_jsonl_rows(tmp_settings, tmp_path: Path) -> None:
    source = _write_jsonl(
        tmp_path / "support_tickets.jsonl",
        [
            {
                "ticket_id": "tk-1",
                "description": "Billing page times out whenever I save card details.",
                "priority": "high",
            },
            {
                "id": "tk-2",
                "body": "Password reset link expired twice.",
                "channel": "email",
            },
        ],
    )

    records = SupportTicketsIngestionAdapter().ingest(tmp_settings, input_path=source)

    assert [record.record_id for record in records] == ["tk-1", "tk-2"]
    assert all(record.dataset_id == "support_tickets" for record in records)
    assert records[0].source_type == "support_ticket"
    assert records[0].metadata == {"priority": "high"}
    assert records[1].metadata == {"channel": "email"}


def test_yelp_adapter_normalizes_fixture_records(tmp_settings) -> None:
    tmp_settings.yelp_businesses_path = Path("tests/fixtures/businesses.jsonl")
    tmp_settings.yelp_reviews_path = Path("tests/fixtures/reviews.jsonl")

    adapter = YelpIngestionAdapter()
    records = adapter.ingest(tmp_settings)
    source_manifests = adapter.source_manifests(tmp_settings)

    assert [record.record_id for record in records] == ["r-1", "r-3"]
    assert all(record.dataset_id == "yelp_restaurants" for record in records)
    assert records[0].metadata["business_id"] == "b-1"
    assert records[0].metadata["categories"] == ["Restaurants", "Italian"]
    assert set(source_manifests) == {"yelp_businesses", "yelp_reviews"}
    assert all(len(checksum) == 64 for checksum in source_manifests.values())


def test_run_ingest_writes_records_and_lineage_manifest(tmp_settings, tmp_path: Path) -> None:
    source = _write_jsonl(
        tmp_path / "app_reviews.jsonl",
        [{"review_id": "ar-1", "text": "Notifications are delayed by hours."}],
    )

    result = run_ingest(
        settings=tmp_settings,
        adapter_id="app_reviews",
        input_path=source,
    )

    assert result.record_count == 1
    assert result.records_path.exists()
    assert result.manifest_path.exists()
    assert result.checksum_sha256
    assert result.source_manifests == {"app_reviews": result.source_manifests["app_reviews"]}

    manifest_payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["dataset_id"] == "app_reviews"
    assert manifest_payload["record_count"] == 1
    assert manifest_payload["checksum_sha256"] == result.checksum_sha256
    assert manifest_payload["source_manifests"] == result.source_manifests


def test_build_snapshot_aggregates_record_counts_and_source_manifests(
    tmp_settings,
    tmp_path: Path,
) -> None:
    app_source = _write_jsonl(
        tmp_path / "app_reviews.jsonl",
        [{"review_id": "ar-1", "text": "Search is very slow on Android."}],
    )
    ticket_source = _write_jsonl(
        tmp_path / "support_tickets.jsonl",
        [{"ticket_id": "tk-1", "description": "Cannot export CSV reports."}],
    )

    run_ingest(settings=tmp_settings, adapter_id="app_reviews", input_path=app_source)
    run_ingest(settings=tmp_settings, adapter_id="support_tickets", input_path=ticket_source)

    snapshot = build_snapshot(settings=tmp_settings, dataset_id="default")

    assert snapshot.manifest.record_count == 2
    assert snapshot.manifest.dataset_ids == ["app_reviews", "support_tickets"]
    assert len(snapshot.manifest.checksum_sha256) == 64
    assert set(snapshot.manifest.source_manifests) == {
        "app_reviews::app_reviews",
        "support_tickets::support_tickets",
    }
    assert snapshot.manifest_path.exists()


def test_build_snapshot_requires_existing_manifests(tmp_settings) -> None:
    with pytest.raises(FileNotFoundError, match="No ingested datasets found"):
        build_snapshot(settings=tmp_settings, dataset_id="default")
