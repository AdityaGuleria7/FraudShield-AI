from typing import Any
from uuid import uuid4


class InvestigationCaseBuilder:

    def build(
        self,
        task_id: str,
        evidence_assessment: dict[str, Any],
        findings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Build a structured investigation case from correlated evidence.

        This builder does not make a final fraud decision.
        """

        risk_score = evidence_assessment["risk_score"]
        risk_level = evidence_assessment["risk_level"]

        if risk_score == 0:
            expected_risk_level = "LOW"
        elif risk_score < 50:
            expected_risk_level = "MEDIUM"
        elif risk_score < 75:
            expected_risk_level = "HIGH"
        else:
            expected_risk_level = "CRITICAL"

        if risk_level != expected_risk_level:
            raise ValueError(
                "Investigation case risk level does not match risk score"
            )

        expected_decision = (
            "NO_SUSPICIOUS_ACTIVITY"
            if risk_score == 0
            else "REQUIRES_FURTHER_INVESTIGATION"
        )

        if evidence_assessment["decision"] != expected_decision:
            raise ValueError(
                "Investigation case decision does not match risk score"
            )

        if evidence_assessment["evidence_count"] != len(findings):
            raise ValueError(
                "Investigation case evidence count does not match findings"
            )

        finding_patterns = {
            finding.get("pattern")
            for finding in findings
            if finding.get("pattern")
        }

        declared_patterns = set(evidence_assessment["patterns"])

        if finding_patterns != declared_patterns:
            raise ValueError(
                "Investigation case patterns do not match findings"
            )

        if risk_score > 0 and not evidence_assessment.get("provenance"):
            raise ValueError(
                "Investigation case requires provenance for risky evidence"
            )

        if risk_score == 0:
            case_status = "NO_ACTION_REQUIRED"
        else:
            case_status = "OPEN_FOR_INVESTIGATION"

        independent_patterns = [
            pattern
            for pattern in evidence_assessment["patterns"]
            if not any(
                pattern in patterns
                for patterns in evidence_assessment[
                    "overlapping_transactions"
                ].values()
            )
        ]

        overlapping_patterns = sorted(
            {
                pattern
                for patterns in evidence_assessment[
                    "overlapping_transactions"
                ].values()
                for pattern in patterns
            }
        )

        return {
            "case_id": f"CASE-{uuid4().hex[:12].upper()}",
            "task_id": task_id,
            "status": case_status,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "decision": evidence_assessment["decision"],
            "patterns": evidence_assessment["patterns"],
            "provenance": evidence_assessment.get("provenance", []),
            "independent_patterns": independent_patterns,
            "overlapping_patterns": overlapping_patterns,
            "overlapping_transactions": evidence_assessment[
                "overlapping_transactions"
            ],
            "evidence_count": evidence_assessment["evidence_count"],
            "evidence_quality": evidence_assessment["evidence_quality"],
            "findings": findings,
            "required_actions": (
                ["Investigate supporting evidence."]
                if risk_score > 0
                else []
            ),
        }
