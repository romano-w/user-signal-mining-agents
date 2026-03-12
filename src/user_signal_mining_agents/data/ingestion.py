from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ..config import Settings
from ..schemas import DatasetRecord, DatasetSnapshotManifest, SnippetProvenance
from .yelp_loader import iter_restaurant_reviews, load_restaurant_business_lookup


DATASETS_DIRNAME = "datasets"
SNAPSHOT_DIRNAME = "snapshots"


@dataclass(slots=True, frozen=True)
class IngestionResult:
    adapter_id: str
    dataset_id: str
    record_count: int
    records_path: Path
    checksum_sha256: str
    source_manifests: dict[str, str]
    manifest_path: Path


@dataclass(slots=True, frozen=True)
class SnapshotResult:
    manifest: DatasetSnapshotManifest
    manifest_path: Path


@dataclass(slots=True, frozen=True)
class _IngestionArtifactManifest:
    adapter_id: str
    dataset_id: str
    record_count: int
    checksum_sha256: str
    source_manifests: dict[str, str]


class IngestionAdapter(Protocol):
    adapter_id: str
    dataset_id: str
    source_type: str

    def ingest(self, settings: Settings, *, input_path: Path | None = None) -> list[DatasetRecord]:
        """Produce normalized records for one source."""

    def source_manifests(
        self,
        settings: Settings,
        *,
        input_path: Path | None = None,
    ) -> dict[str, str]:
        """Return source file checksums used for lineage."""


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _normalize_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split())
    return normalized or None


def _iter_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _first_str(payload: dict[str, object], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _metadata_without(payload: dict[str, object], exclude_keys: set[str]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in exclude_keys}


def _require_existing_file(path: Path, *, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _require_input_path(input_path: Path | None, *, adapter_id: str) -> Path:
    if input_path is None:
        raise ValueError(f"Adapter {adapter_id!r} requires --input-path.")
    return _require_existing_file(input_path, label=f"{adapter_id} input path")


def datasets_dir(settings: Settings) -> Path:
    return settings.run_artifacts_dir / DATASETS_DIRNAME


def dataset_records_path(settings: Settings, dataset_id: str) -> Path:
    return datasets_dir(settings) / f"{dataset_id}.jsonl"


def dataset_manifest_path(settings: Settings, dataset_id: str) -> Path:
    return datasets_dir(settings) / f"{dataset_id}.manifest.json"


def snapshot_manifest_path(settings: Settings, snapshot_id: str) -> Path:
    return datasets_dir(settings) / SNAPSHOT_DIRNAME / f"{snapshot_id}.json"


def _write_records(records: list[DatasetRecord], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = record.model_dump(mode="json", exclude_none=True)
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            count += 1
    return count


def _write_ingestion_manifest(
    *,
    adapter_id: str,
    dataset_id: str,
    record_count: int,
    checksum_sha256: str,
    source_manifests: dict[str, str],
    records_path: Path,
    manifest_path: Path,
) -> None:
    payload = {
        "adapter_id": adapter_id,
        "dataset_id": dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "record_count": record_count,
        "checksum_sha256": checksum_sha256,
        "source_manifests": source_manifests,
        "records_path": str(records_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_ingestion_manifest(path: Path) -> _IngestionArtifactManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Ingestion manifest at {path} is not an object.")

    adapter_id = payload.get("adapter_id")
    dataset_id = payload.get("dataset_id")
    record_count = payload.get("record_count")
    checksum_sha256 = payload.get("checksum_sha256")
    source_manifests = payload.get("source_manifests", {})

    if not isinstance(adapter_id, str) or not adapter_id:
        raise ValueError(f"Ingestion manifest at {path} has invalid adapter_id.")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError(f"Ingestion manifest at {path} has invalid dataset_id.")
    if not isinstance(record_count, int) or record_count < 0:
        raise ValueError(f"Ingestion manifest at {path} has invalid record_count.")
    if not isinstance(checksum_sha256, str) or not checksum_sha256:
        raise ValueError(f"Ingestion manifest at {path} has invalid checksum_sha256.")
    if not isinstance(source_manifests, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in source_manifests.items()
    ):
        raise ValueError(f"Ingestion manifest at {path} has invalid source_manifests.")

    return _IngestionArtifactManifest(
        adapter_id=adapter_id,
        dataset_id=dataset_id,
        record_count=record_count,
        checksum_sha256=checksum_sha256,
        source_manifests=source_manifests,
    )


class _JsonlIngestionAdapter:
    adapter_id = ""
    dataset_id = ""
    source_type = ""
    _id_keys: tuple[str, ...] = ()
    _text_keys: tuple[str, ...] = ()

    def ingest(self, settings: Settings, *, input_path: Path | None = None) -> list[DatasetRecord]:
        del settings
        source_path = _require_input_path(input_path, adapter_id=self.adapter_id)
        rows = _iter_jsonl(source_path)
        records: list[DatasetRecord] = []
        excluded_keys = set(self._id_keys) | set(self._text_keys)

        for line_number, payload in enumerate(rows, start=1):
            record_id = _first_str(payload, self._id_keys) or f"{self.dataset_id}_{line_number}"

            text = _normalize_text(_first_str(payload, self._text_keys))
            if text is None:
                continue

            records.append(
                DatasetRecord(
                    record_id=record_id,
                    dataset_id=self.dataset_id,
                    source_type=self.source_type,
                    text=text,
                    metadata=_metadata_without(payload, excluded_keys),
                    provenance=SnippetProvenance(
                        source_dataset_id=self.dataset_id,
                        source_record_id=record_id,
                        source_type=self.source_type,
                    ),
                )
            )
        return records

    def source_manifests(
        self,
        settings: Settings,
        *,
        input_path: Path | None = None,
    ) -> dict[str, str]:
        del settings
        source_path = _require_input_path(input_path, adapter_id=self.adapter_id)
        return {self.adapter_id: _sha256_file(source_path)}


class YelpIngestionAdapter:
    adapter_id = "yelp"
    dataset_id = "yelp_restaurants"
    source_type = "yelp_review"

    def ingest(self, settings: Settings, *, input_path: Path | None = None) -> list[DatasetRecord]:
        if input_path is not None:
            raise ValueError("Adapter 'yelp' does not accept --input-path.")

        businesses_path = _require_existing_file(settings.yelp_businesses_path, label="Yelp businesses file")
        reviews_path = _require_existing_file(settings.yelp_reviews_path, label="Yelp reviews file")
        businesses = load_restaurant_business_lookup(businesses_path)

        records: list[DatasetRecord] = []
        reviews = iter_restaurant_reviews(
            reviews_path,
            businesses,
            review_limit=settings.restaurant_review_limit,
            min_review_characters=settings.min_review_characters,
            max_reviews_per_business=settings.max_reviews_per_business,
        )
        for review in reviews:
            business = businesses.get(review.business_id)
            if business is None:
                continue

            records.append(
                DatasetRecord(
                    record_id=review.review_id,
                    dataset_id=self.dataset_id,
                    source_type=self.source_type,
                    text=review.text,
                    metadata={
                        "business_id": review.business_id,
                        "business_name": business.name,
                        "city": business.city,
                        "state": business.state,
                        "categories": list(business.categories),
                        "review_stars": review.stars,
                        "business_stars": business.stars,
                        "business_review_count": business.review_count,
                        "review_date": review.date,
                    },
                    provenance=SnippetProvenance(
                        source_dataset_id=self.dataset_id,
                        source_record_id=review.review_id,
                        source_type=self.source_type,
                    ),
                )
            )
        return records

    def source_manifests(
        self,
        settings: Settings,
        *,
        input_path: Path | None = None,
    ) -> dict[str, str]:
        if input_path is not None:
            raise ValueError("Adapter 'yelp' does not accept --input-path.")
        businesses_path = _require_existing_file(settings.yelp_businesses_path, label="Yelp businesses file")
        reviews_path = _require_existing_file(settings.yelp_reviews_path, label="Yelp reviews file")
        return {
            "yelp_businesses": _sha256_file(businesses_path),
            "yelp_reviews": _sha256_file(reviews_path),
        }


class AppReviewsIngestionAdapter(_JsonlIngestionAdapter):
    adapter_id = "app_reviews"
    dataset_id = "app_reviews"
    source_type = "app_review"
    _id_keys = ("review_id", "id")
    _text_keys = ("text", "review_text", "content", "body")


class SupportTicketsIngestionAdapter(_JsonlIngestionAdapter):
    adapter_id = "support_tickets"
    dataset_id = "support_tickets"
    source_type = "support_ticket"
    _id_keys = ("ticket_id", "id")
    _text_keys = ("text", "description", "body")


_ADAPTER_REGISTRY: dict[str, IngestionAdapter] = {
    adapter.adapter_id: adapter
    for adapter in (
        YelpIngestionAdapter(),
        AppReviewsIngestionAdapter(),
        SupportTicketsIngestionAdapter(),
    )
}


def list_adapter_ids() -> list[str]:
    return sorted(_ADAPTER_REGISTRY)


def get_adapter(adapter_id: str) -> IngestionAdapter:
    adapter = _ADAPTER_REGISTRY.get(adapter_id)
    if adapter is None:
        valid = ", ".join(list_adapter_ids())
        raise ValueError(f"Unknown adapter {adapter_id!r}. Expected one of: {valid}.")
    return adapter


def run_ingest(
    *,
    settings: Settings,
    adapter_id: str,
    input_path: Path | None = None,
) -> IngestionResult:
    adapter = get_adapter(adapter_id)
    records = adapter.ingest(settings, input_path=input_path)
    records_path = dataset_records_path(settings, adapter.dataset_id)
    record_count = _write_records(records, records_path)
    checksum_sha256 = _sha256_file(records_path)
    source_manifests = adapter.source_manifests(settings, input_path=input_path)
    manifest_path = dataset_manifest_path(settings, adapter.dataset_id)

    _write_ingestion_manifest(
        adapter_id=adapter.adapter_id,
        dataset_id=adapter.dataset_id,
        record_count=record_count,
        checksum_sha256=checksum_sha256,
        source_manifests=source_manifests,
        records_path=records_path,
        manifest_path=manifest_path,
    )

    return IngestionResult(
        adapter_id=adapter.adapter_id,
        dataset_id=adapter.dataset_id,
        record_count=record_count,
        records_path=records_path,
        checksum_sha256=checksum_sha256,
        source_manifests=source_manifests,
        manifest_path=manifest_path,
    )


def _load_ingestion_manifests(
    settings: Settings,
    *,
    dataset_id: str,
) -> list[_IngestionArtifactManifest]:
    root = datasets_dir(settings)
    if not root.exists():
        raise FileNotFoundError(f"No ingested datasets found under {root}. Run `usm ingest` first.")

    manifest_paths = sorted(root.glob("*.manifest.json"))
    if not manifest_paths:
        raise FileNotFoundError(f"No ingestion manifests found under {root}. Run `usm ingest` first.")

    manifests = [_read_ingestion_manifest(path) for path in manifest_paths]
    if dataset_id == "default":
        return manifests

    selected = [manifest for manifest in manifests if manifest.dataset_id == dataset_id]
    if not selected:
        raise FileNotFoundError(
            f"No ingestion manifest found for dataset_id={dataset_id!r} under {root}. "
            "Run `usm ingest --adapter <id>` first."
        )
    return selected


def build_snapshot(
    *,
    settings: Settings,
    dataset_id: str = "default",
) -> SnapshotResult:
    manifests = sorted(
        _load_ingestion_manifests(settings, dataset_id=dataset_id),
        key=lambda item: item.dataset_id,
    )

    digest = hashlib.sha256()
    total_records = 0
    dataset_ids: list[str] = []
    source_manifests: dict[str, str] = {}

    for manifest in manifests:
        dataset_ids.append(manifest.dataset_id)
        total_records += manifest.record_count

        digest.update(manifest.dataset_id.encode("utf-8"))
        digest.update(b":")
        digest.update(manifest.checksum_sha256.encode("utf-8"))
        digest.update(b"\n")

        for source_name, source_checksum in sorted(manifest.source_manifests.items()):
            lineage_key = f"{manifest.dataset_id}::{source_name}"
            source_manifests[lineage_key] = source_checksum
            digest.update(lineage_key.encode("utf-8"))
            digest.update(b":")
            digest.update(source_checksum.encode("utf-8"))
            digest.update(b"\n")

    snapshot_id = f"{dataset_id}_{_utc_stamp()}"
    snapshot_manifest = DatasetSnapshotManifest(
        snapshot_id=snapshot_id,
        dataset_ids=dataset_ids,
        record_count=total_records,
        checksum_sha256=digest.hexdigest(),
        source_manifests=source_manifests,
    )
    output_path = snapshot_manifest_path(settings, snapshot_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(snapshot_manifest.model_dump_json(indent=2), encoding="utf-8")

    return SnapshotResult(
        manifest=snapshot_manifest,
        manifest_path=output_path,
    )
