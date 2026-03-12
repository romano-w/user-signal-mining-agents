from __future__ import annotations

import pytest
from pydantic import ValidationError

from user_signal_mining_agents.schemas import (
    DatasetRecord,
    DatasetSnapshotManifest,
    DomainPack,
    FailureTag,
    JudgePanelResult,
    JudgeScores,
    MetricWithCI,
    RobustnessCase,
    SignificanceResult,
    SnippetProvenance,
)


def _judge_scores(value: float) -> JudgeScores:
    return JudgeScores(
        relevance=value,
        actionability=value,
        evidence_grounding=value,
        contradiction_handling=value,
        non_redundancy=value,
        rationale="consistent",
    )


def test_dataset_record_supports_provenance_contract() -> None:
    provenance = SnippetProvenance(
        source_dataset_id="yelp_2026_03",
        source_record_id="review_123",
        source_type="yelp_review",
        start_char=10,
        end_char=42,
    )
    record = DatasetRecord(
        record_id="record_1",
        dataset_id="restaurants",
        source_type="yelp_review",
        text="Service was slow but food was good.",
        provenance=provenance,
    )

    assert record.provenance is not None
    assert record.provenance.source_record_id == "review_123"


def test_dataset_snapshot_manifest_rejects_negative_record_count() -> None:
    with pytest.raises(ValidationError, match="record_count"):
        DatasetSnapshotManifest(
            snapshot_id="snapshot_1",
            dataset_ids=["restaurants"],
            record_count=-1,
            checksum_sha256="abc123",
        )


def test_judge_panel_result_contract() -> None:
    result = JudgePanelResult(
        prompt_id="p1",
        system_variant="pipeline",
        panel_size=3,
        per_judge_scores=[_judge_scores(4.0), _judge_scores(4.5), _judge_scores(3.8)],
        aggregate_scores=_judge_scores(4.1),
        metrics_with_ci=[
            MetricWithCI(
                metric="overall",
                mean=4.1,
                ci95_lower=3.9,
                ci95_upper=4.3,
                sample_size=3,
            )
        ],
        significance=[
            SignificanceResult(
                metric="overall",
                p_value=0.03,
                is_significant=True,
                effect_size=0.35,
            )
        ],
    )

    assert result.panel_size == 3
    assert result.metrics_with_ci[0].metric == "overall"
    assert result.significance[0].is_significant is True


def test_failure_tag_robustness_case_and_domain_pack_contracts() -> None:
    tag = FailureTag(
        tag_id="ft_1",
        category="retrieval_miss",
        severity=4,
        description="Top evidence misses the core complaint cluster.",
        evidence_refs=["snippet_12", "snippet_31"],
    )
    case = RobustnessCase(
        case_id="rb_negation",
        family="negation",
        description="Flip polarity markers in founder statement.",
        transform_spec={"strategy": "negation_flip"},
        expected_behavior="Should preserve grounding and avoid contradiction errors.",
    )
    domain = DomainPack(
        domain_id="saas",
        title="SaaS retention",
        founder_prompts_path="founder_prompts/saas.json",
    )

    assert tag.severity == 4
    assert case.family == "negation"
    assert domain.enabled is True

