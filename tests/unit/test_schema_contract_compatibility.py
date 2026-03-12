from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from user_signal_mining_agents.schemas import (
    DatasetRecord,
    DatasetSnapshotManifest,
    DomainPack,
    ExperimentManifest,
    FailureTag,
    JudgePanelResult,
    MetricWithCI,
    RobustnessCase,
    SignificanceResult,
    SnippetProvenance,
)


SCHEMA_FIELD_EXPECTATIONS: list[tuple[type[BaseModel], set[str], set[str]]] = [
    (
        SnippetProvenance,
        {"source_dataset_id", "source_record_id", "source_type"},
        {"start_char", "end_char", "extracted_at"},
    ),
    (
        DatasetRecord,
        {"record_id", "dataset_id", "source_type", "text"},
        {"metadata", "provenance"},
    ),
    (
        DatasetSnapshotManifest,
        {"snapshot_id", "record_count", "checksum_sha256"},
        {"created_at", "dataset_ids", "source_manifests"},
    ),
    (
        ExperimentManifest,
        {
            "run_id",
            "dataset_snapshot_id",
            "prompt_bundle_id",
            "embedding_index_id",
            "llm_provider",
            "llm_model",
            "git_commit",
        },
        {"created_at", "system_variants", "parameters"},
    ),
    (
        MetricWithCI,
        {"metric", "mean", "ci95_lower", "ci95_upper", "sample_size"},
        set(),
    ),
    (
        SignificanceResult,
        {"metric", "p_value", "is_significant"},
        {"effect_size", "notes"},
    ),
    (
        JudgePanelResult,
        {"prompt_id", "system_variant", "panel_size", "aggregate_scores"},
        {"per_judge_scores", "metrics_with_ci", "significance"},
    ),
    (
        FailureTag,
        {"tag_id", "category", "severity", "description"},
        {"prompt_id", "evidence_refs"},
    ),
    (
        RobustnessCase,
        {"case_id", "family", "description", "expected_behavior"},
        {"transform_spec"},
    ),
    (
        DomainPack,
        {"domain_id", "title", "founder_prompts_path"},
        {"evaluation_notes", "enabled"},
    ),
]


def _field_sets(model: type[BaseModel]) -> tuple[set[str], set[str]]:
    schema: dict[str, Any] = model.model_json_schema()
    properties = set(schema.get("properties", {}).keys())
    required = set(schema.get("required", []))
    optional = properties - required
    return required, optional


@pytest.mark.parametrize(
    ("model", "expected_required", "expected_optional"),
    SCHEMA_FIELD_EXPECTATIONS,
    ids=[model.__name__ for model, _, _ in SCHEMA_FIELD_EXPECTATIONS],
)
def test_schema_field_contracts_are_stable(
    model: type[BaseModel],
    expected_required: set[str],
    expected_optional: set[str],
) -> None:
    required, optional = _field_sets(model)
    assert required == expected_required
    assert optional == expected_optional


def test_dataset_record_accepts_legacy_minimal_payload() -> None:
    record = DatasetRecord.model_validate(
        {
            "record_id": "r1",
            "dataset_id": "restaurants_2026",
            "source_type": "yelp_review",
            "text": "Service was slow, but food quality was high.",
        }
    )

    assert record.metadata == {}
    assert record.provenance is None


def test_snapshot_manifest_accepts_legacy_minimal_payload() -> None:
    manifest = DatasetSnapshotManifest.model_validate(
        {
            "snapshot_id": "snap_001",
            "record_count": 5,
            "checksum_sha256": "checksum",
        }
    )

    assert manifest.dataset_ids == []
    assert manifest.source_manifests == {}


def test_experiment_manifest_accepts_legacy_minimal_payload() -> None:
    manifest = ExperimentManifest.model_validate(
        {
            "run_id": "run_001",
            "dataset_snapshot_id": "snap_001",
            "prompt_bundle_id": "founder_prompts_v1",
            "embedding_index_id": "idx_v1",
            "llm_provider": "openai",
            "llm_model": "gpt-test",
            "git_commit": "abc123",
        }
    )

    assert manifest.parameters == {}
    assert manifest.system_variants == []


def test_judge_panel_result_accepts_minimal_aggregate_payload() -> None:
    result = JudgePanelResult.model_validate(
        {
            "prompt_id": "p1",
            "system_variant": "pipeline",
            "panel_size": 3,
            "aggregate_scores": {
                "relevance": 4.2,
                "overall_preference": 4.1,
                "coverage": 4.3,
                "contradiction": 4.0,
                "distinctiveness": 4.0,
                "rationale": "aggregate",
            },
        }
    )

    assert result.per_judge_scores == []
    assert result.metrics_with_ci == []
    assert result.significance == []


def test_failure_tag_robustness_case_and_domain_pack_accept_minimal_payloads() -> None:
    tag = FailureTag.model_validate(
        {
            "tag_id": "ft_1",
            "category": "retrieval_miss",
            "severity": 3,
            "description": "Missing primary evidence cluster.",
        }
    )
    case = RobustnessCase.model_validate(
        {
            "case_id": "rb_negation",
            "family": "negation",
            "description": "Negate primary sentiment phrase.",
            "expected_behavior": "System should preserve grounding with adjusted polarity.",
        }
    )
    pack = DomainPack.model_validate(
        {
            "domain_id": "saas",
            "title": "SaaS retention",
            "founder_prompts_path": "founder_prompts/saas.json",
        }
    )

    assert tag.evidence_refs == []
    assert case.transform_spec == {}
    assert pack.enabled is True

