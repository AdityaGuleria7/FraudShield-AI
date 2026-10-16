"""Normalize existing orchestrator output into the versioned result contract."""

from typing import Any

from app.models.investigation_result import (
    InvestigationResult,
    ModelSignal,
    ReviewStatus,
    RiskAssessment,
)


def normalize_investigation_result(
    task_id: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Return a JSON-compatible normalized result without mutating input."""

    data_quality = result.get("data_quality")
    pattern_review = result.get("pattern_review")
    evidence = result.get("evidence_assessment")
    case = result.get("investigation_case")
    findings = result.get("pattern_findings") or []

    if data_quality and data_quality.get("status") == "REJECT":
        status = "REJECTED"
        review = ReviewStatus(
            status="REJECT",
            approved_for_next_stage=False,
            issues=["Data quality validation rejected the transaction data."],
        )
        risk = None
    elif (
        result.get("review") is not None
        and (
            result["review"].status != "PASS"
            or not result["review"].approved_for_next_stage
        )
    ):
        status = "REJECTED"
        source_review = result["review"]
        review = ReviewStatus(
            status="REJECT",
            approved_for_next_stage=False,
            issues=list(source_review.issues),
            required_actions=list(source_review.required_actions),
        )
        risk = None
    elif (
        pattern_review is not None
        and (
            pattern_review.status != "PASS"
            or not pattern_review.approved_for_next_stage
        )
    ):
        status = "BLOCKED"
        review = ReviewStatus(
            status="REJECT",
            approved_for_next_stage=False,
            issues=list(pattern_review.issues),
            required_actions=list(pattern_review.required_actions),
        )
        risk = None
    elif evidence is not None:
        score = int(evidence["risk_score"])
        risk = RiskAssessment(
            score=score,
            level=evidence["risk_level"],
            decision=evidence["decision"],
            evidence_quality=evidence["evidence_quality"],
            evidence_count=int(evidence["evidence_count"]),
            reasons=[
                str(item.get("explanation", {}).get("plain_summary"))
                for item in findings
                if item.get("explanation", {}).get("plain_summary")
            ],
        )
        status = (
            "REQUIRES_REVIEW"
            if score > 0
            else "COMPLETED"
        )
        review = ReviewStatus(
            status="PASS",
            approved_for_next_stage=True,
            required_actions=list(case.get("required_actions", []))
            if case else [],
        )
    else:
        status = "BLOCKED"
        risk = None
        review = ReviewStatus(
            status="PENDING",
            approved_for_next_stage=False,
            issues=["The investigation did not produce a final evidence assessment."],
        )

    limitations = [
        "Deterministic risk score is a rule-based score, not a fraud probability.",
        "No independent claim of real-world fraud detection performance is made.",
    ]

    normalized = InvestigationResult(
        task_id=task_id,
        case_id=case.get("case_id") if case else None,
        status=status,
        risk=risk,
        model_signal=ModelSignal(),
        review=review,
        findings=list(findings),
        explanation=result.get("executive_summary"),
        limitations=limitations,
    )
    return normalized.model_dump(mode="json")
