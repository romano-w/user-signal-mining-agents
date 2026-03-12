from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..schemas import FailureTag


class GateInputs(BaseModel):
    """File locations and thresholds used by integration readiness checks."""

    model_config = ConfigDict(extra="forbid")

    schema_compatibility_path: Path
    retrieval_report_path: Path
    robustness_report_path: Path
    domain_transfer_report_path: Path
    failure_tags_report_path: Path
    high_severity_threshold: int = 4

    def model_post_init(self, __context: object) -> None:
        if not 1 <= self.high_severity_threshold <= 5:
            raise ValueError("high_severity_threshold must be between 1 and 5")


class GateCheck(BaseModel):
    """One gate check result in the integration summary."""

    model_config = ConfigDict(extra="forbid")

    check: str
    status: Literal["pass", "fail"]
    blocking: bool = True
    path: str
    details: str


class IntegrationGateSummary(BaseModel):
    """Machine-readable result for all integration readiness checks."""

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    overall_status: Literal["pass", "fail"]
    blocking_failures: int = Field(ge=0)
    checks: list[GateCheck] = Field(default_factory=list)


class _StatusReport(BaseModel):
    model_config = ConfigDict(extra="allow")

    report_type: str
    status: Literal["pass", "fail"]


class _FailureTagReport(BaseModel):
    model_config = ConfigDict(extra="allow")

    report_type: str
    tags: list[FailureTag] = Field(default_factory=list)


RETRIEVAL_BENCHMARK_METRICS = ("recall_at_k", "mrr_at_k", "ndcg_at_k")


def default_gate_inputs(
    reports_dir: Path,
    *,
    high_severity_threshold: int = 4,
) -> GateInputs:
    """Build default gate input paths under a shared reports directory."""

    return GateInputs(
        schema_compatibility_path=reports_dir / "schema_compatibility.json",
        retrieval_report_path=reports_dir / "retrieval_eval_summary.json",
        robustness_report_path=reports_dir / "robustness_report.json",
        domain_transfer_report_path=reports_dir / "domain_transfer_report.json",
        failure_tags_report_path=reports_dir / "failure_tags_report.json",
        high_severity_threshold=high_severity_threshold,
    )


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("report payload must be a JSON object")
    return payload


def _validation_error_message(error: ValidationError) -> str:
    first_error = error.errors()[0] if error.errors() else {"msg": str(error)}
    location = ".".join(str(part) for part in first_error.get("loc", ()))
    message = str(first_error.get("msg", "validation failed"))
    return f"{location}: {message}" if location else message


def _report_missing(check_name: str, path: Path) -> GateCheck:
    return GateCheck(
        check=check_name,
        status="fail",
        path=str(path),
        details="required report file is missing",
    )


def _status_report_check(
    *,
    check_name: str,
    expected_type: str,
    path: Path,
) -> GateCheck:
    if not path.exists():
        return _report_missing(check_name, path)

    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return GateCheck(
            check=check_name,
            status="fail",
            path=str(path),
            details=f"unable to read report JSON: {exc}",
        )

    try:
        report = _StatusReport.model_validate(payload)
    except ValidationError as exc:
        return GateCheck(
            check=check_name,
            status="fail",
            path=str(path),
            details=f"invalid report schema: {_validation_error_message(exc)}",
        )

    if report.report_type != expected_type:
        return GateCheck(
            check=check_name,
            status="fail",
            path=str(path),
            details=f"report_type was {report.report_type!r}, expected {expected_type!r}",
        )

    if report.status != "pass":
        return GateCheck(
            check=check_name,
            status="fail",
            path=str(path),
            details="report status is fail",
        )

    return GateCheck(
        check=check_name,
        status="pass",
        path=str(path),
        details="report present with pass status",
    )


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _retrieval_report_check(path: Path) -> GateCheck:
    check_name = "retrieval"
    if not path.exists():
        return _report_missing(check_name, path)

    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return GateCheck(
            check=check_name,
            status="fail",
            path=str(path),
            details=f"unable to read report JSON: {exc}",
        )

    if "report_type" in payload or "status" in payload:
        return _status_report_check(
            check_name=check_name,
            expected_type="retrieval",
            path=path,
        )

    query_count = payload.get("query_count")
    if not isinstance(query_count, int) or query_count < 1:
        return GateCheck(
            check=check_name,
            status="fail",
            path=str(path),
            details="retrieval benchmark must contain query_count >= 1",
        )

    k_values_raw = payload.get("k_values")
    if not isinstance(k_values_raw, list) or not k_values_raw:
        return GateCheck(
            check=check_name,
            status="fail",
            path=str(path),
            details="retrieval benchmark must include non-empty k_values list",
        )
    if any(not isinstance(k, int) or k <= 0 for k in k_values_raw):
        return GateCheck(
            check=check_name,
            status="fail",
            path=str(path),
            details="retrieval benchmark k_values entries must be positive integers",
        )
    expected_k_keys = {str(k) for k in k_values_raw}

    aggregates = payload.get("aggregates")
    if not isinstance(aggregates, dict):
        return GateCheck(
            check=check_name,
            status="fail",
            path=str(path),
            details="retrieval benchmark must include an aggregates object",
        )

    missing_metrics = [metric for metric in RETRIEVAL_BENCHMARK_METRICS if metric not in aggregates]
    if missing_metrics:
        return GateCheck(
            check=check_name,
            status="fail",
            path=str(path),
            details=f"retrieval benchmark missing aggregate metrics: {', '.join(missing_metrics)}",
        )

    for metric in RETRIEVAL_BENCHMARK_METRICS:
        metric_values = aggregates.get(metric)
        if not isinstance(metric_values, dict) or not metric_values:
            return GateCheck(
                check=check_name,
                status="fail",
                path=str(path),
                details=f"aggregate metric {metric!r} must be a non-empty object",
            )

        missing_k = sorted(expected_k_keys - set(metric_values.keys()), key=int)
        if missing_k:
            return GateCheck(
                check=check_name,
                status="fail",
                path=str(path),
                details=f"aggregate metric {metric!r} missing K entries: {', '.join(missing_k)}",
            )

        for k in expected_k_keys:
            score = metric_values.get(k)
            if not _is_number(score):
                return GateCheck(
                    check=check_name,
                    status="fail",
                    path=str(path),
                    details=f"aggregate metric {metric!r} at K={k} must be numeric",
                )

    return GateCheck(
        check=check_name,
        status="pass",
        path=str(path),
        details=(
            "retrieval benchmark present with aggregate metrics "
            f"({', '.join(RETRIEVAL_BENCHMARK_METRICS)}) across K={k_values_raw}"
        ),
    )


def _failure_tag_check(path: Path, severity_threshold: int) -> GateCheck:
    check_name = "failure_tags"
    if not path.exists():
        return _report_missing(check_name, path)

    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return GateCheck(
            check=check_name,
            status="fail",
            path=str(path),
            details=f"unable to read report JSON: {exc}",
        )

    try:
        report = _FailureTagReport.model_validate(payload)
    except ValidationError as exc:
        return GateCheck(
            check=check_name,
            status="fail",
            path=str(path),
            details=f"invalid report schema: {_validation_error_message(exc)}",
        )

    if report.report_type != "failure_taxonomy":
        return GateCheck(
            check=check_name,
            status="fail",
            path=str(path),
            details=f"report_type was {report.report_type!r}, expected 'failure_taxonomy'",
        )

    blocked = [tag for tag in report.tags if tag.severity >= severity_threshold]
    if blocked:
        blocked_labels = ", ".join(f"{tag.tag_id}(severity={tag.severity})" for tag in blocked)
        return GateCheck(
            check=check_name,
            status="fail",
            path=str(path),
            details=(
                f"{len(blocked)} high-severity failure tag(s) at threshold {severity_threshold}: "
                f"{blocked_labels}"
            ),
        )

    return GateCheck(
        check=check_name,
        status="pass",
        path=str(path),
        details=(
            f"no failure tags at or above severity threshold {severity_threshold} "
            f"(total tags: {len(report.tags)})"
        ),
    )


def run_integration_gates(inputs: GateInputs) -> IntegrationGateSummary:
    """Run all integration-readiness checks and return a summary payload."""

    checks = [
        _status_report_check(
            check_name="schema_compatibility",
            expected_type="schema_compatibility",
            path=inputs.schema_compatibility_path,
        ),
        _retrieval_report_check(inputs.retrieval_report_path),
        _status_report_check(
            check_name="robustness",
            expected_type="robustness",
            path=inputs.robustness_report_path,
        ),
        _status_report_check(
            check_name="domain_transfer",
            expected_type="domain_transfer",
            path=inputs.domain_transfer_report_path,
        ),
        _failure_tag_check(
            path=inputs.failure_tags_report_path,
            severity_threshold=inputs.high_severity_threshold,
        ),
    ]

    blocking_failures = sum(1 for check in checks if check.blocking and check.status == "fail")

    return IntegrationGateSummary(
        generated_at=datetime.now(timezone.utc),
        overall_status="pass" if blocking_failures == 0 else "fail",
        blocking_failures=blocking_failures,
        checks=checks,
    )
