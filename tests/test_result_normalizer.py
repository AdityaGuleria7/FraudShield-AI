from types import SimpleNamespace

import pytest

from app.models.result_normalizer import normalize_investigation_result


@pytest.mark.parametrize("invalid_result", [None, [], 123, "invalid"])
def test_normalizer_rejects_non_dictionary_result(invalid_result):
    with pytest.raises(
        ValueError,
        match="Investigation result must be a dictionary",
    ):
        normalize_investigation_result("task-invalid-result", invalid_result)


def test_normalizes_completed_low_risk_result():
    result = normalize_investigation_result(
        "task-1",
        {
            "data_quality": {"status": "PASS"},
            "pattern_review": SimpleNamespace(
                status="PASS",
                approved_for_next_stage=True,
                issues=[],
                required_actions=[],
            ),
            "evidence_assessment": {
                "risk_score": 0,
                "risk_level": "LOW",
                "decision": "NO_SUSPICIOUS_ACTIVITY",
                "evidence_quality": "NONE",
                "evidence_count": 0,
            },
            "investigation_case": {
                "case_id": "CASE-ABC123",
                "required_actions": [],
            },
            "pattern_findings": [],
            "executive_summary": "No recognized suspicious patterns were detected.",
        },
    )
    assert result["schema_version"] == "1.0"
    assert result["status"] == "COMPLETED"
    assert result["risk"]["score"] == 0
    assert result["model_signal"]["score_semantics"] == "not_available"


def test_normalizes_risky_result_and_preserves_reason():
    result = normalize_investigation_result(
        "task-2",
        {
            "data_quality": {"status": "PASS"},
            "pattern_review": SimpleNamespace(
                status="PASS",
                approved_for_next_stage=True,
                issues=[],
                required_actions=[],
            ),
            "evidence_assessment": {
                "risk_score": 55,
                "risk_level": "HIGH",
                "decision": "REQUIRES_FURTHER_INVESTIGATION",
                "evidence_quality": "STRONG",
                "evidence_count": 1,
            },
            "investigation_case": {
                "case_id": "CASE-ABC456",
                "required_actions": ["Investigate supporting evidence."],
            },
            "pattern_findings": [
                {
                    "pattern": "VELOCITY_ATTACK",
                    "explanation": {
                        "plain_summary": "Several transactions occurred within a short period."
                    },
                }
            ],
            "executive_summary": "Review the transaction burst.",
        },
    )
    assert result["status"] == "REQUIRES_REVIEW"
    assert result["risk"]["reasons"] == [
        "Several transactions occurred within a short period."
    ]
    assert result["review"]["required_actions"] == [
        "Investigate supporting evidence."
    ]


@pytest.mark.parametrize("invalid_score", [47.9, "47", True, False, None])
def test_normalizer_rejects_non_integer_risk_scores(invalid_score):
    result = {
        "data_quality": {"status": "PASS"},
        "pattern_review": SimpleNamespace(
            status="PASS",
            approved_for_next_stage=True,
            issues=[],
            required_actions=[],
        ),
        "evidence_assessment": {
            "risk_score": invalid_score,
            "risk_level": "MEDIUM",
            "decision": "REQUIRES_FURTHER_INVESTIGATION",
            "evidence_quality": "STRONG",
            "evidence_count": 1,
        },
        "pattern_findings": [],
    }

    with pytest.raises(ValueError, match="risk_score must be an integer"):
        normalize_investigation_result("task-invalid-score", result)


@pytest.mark.parametrize("invalid_count", [2.7, "2", True, False, None])
def test_normalizer_rejects_non_integer_evidence_counts(invalid_count):
    result = {
        "data_quality": {"status": "PASS"},
        "pattern_review": SimpleNamespace(
            status="PASS",
            approved_for_next_stage=True,
            issues=[],
            required_actions=[],
        ),
        "evidence_assessment": {
            "risk_score": 55,
            "risk_level": "HIGH",
            "decision": "REQUIRES_FURTHER_INVESTIGATION",
            "evidence_quality": "STRONG",
            "evidence_count": invalid_count,
        },
        "pattern_findings": [],
    }

    with pytest.raises(ValueError, match="Evidence count must be a non-negative integer"):
        normalize_investigation_result("task-invalid-evidence-count", result)


def test_rejected_data_is_not_given_a_risk_score():
    result = normalize_investigation_result(
        "task-3",
        {"data_quality": {"status": "REJECT"}},
    )
    assert result["status"] == "REJECTED"
    assert result["risk"] is None
    assert result["review"]["approved_for_next_stage"] is False


def test_unapproved_pattern_review_blocks_result():
    result = normalize_investigation_result(
        "task-4",
        {
            "data_quality": {"status": "PASS"},
            "pattern_review": SimpleNamespace(
                status="REJECT",
                approved_for_next_stage=False,
                issues=["Evidence mismatch."],
                required_actions=["Review the findings."],
            ),
            "evidence_assessment": {
                "risk_score": 55,
                "risk_level": "HIGH",
                "decision": "REQUIRES_FURTHER_INVESTIGATION",
                "evidence_quality": "STRONG",
                "evidence_count": 1,
            },
            "pattern_findings": [],
        },
    )
    assert result["status"] == "BLOCKED"
    assert result["risk"] is None
    assert result["review"]["approved_for_next_stage"] is False


def test_guardian_rejection_is_rejected_without_risk_score():
    result = normalize_investigation_result(
        "task-5",
        {
            "review": SimpleNamespace(
                status="REJECT",
                approved_for_next_stage=False,
                issues=["Critical objection: source data is inconsistent."],
                required_actions=["Correct the source data."],
            ),
            "data_quality": None,
            "evidence_assessment": None,
            "pattern_findings": [],
        },
    )

    assert result["status"] == "REJECTED"
    assert result["risk"] is None
    assert result["review"]["status"] == "REJECT"
    assert result["review"]["approved_for_next_stage"] is False
    assert "Critical objection: source data is inconsistent." in result["review"]["issues"]


def test_evidence_count_does_not_depend_on_reason_summary_count():
    result = normalize_investigation_result(
        "task-6",
        {
            "data_quality": {"status": "PASS"},
            "pattern_review": SimpleNamespace(
                status="PASS",
                approved_for_next_stage=True,
                issues=[],
                required_actions=[],
            ),
            "evidence_assessment": {
                "risk_score": 55,
                "risk_level": "HIGH",
                "decision": "REQUIRES_FURTHER_INVESTIGATION",
                "evidence_quality": "STRONG",
                "evidence_count": 2,
            },
            "investigation_case": {
                "case_id": "CASE-ABC789",
                "required_actions": ["Review both evidence items."],
            },
            "pattern_findings": [
                {
                    "pattern": "VELOCITY_ATTACK",
                    "explanation": {
                        "plain_summary": "Several transactions occurred within a short period."
                    },
                },
                {
                    "pattern": "NEW_DEVICE",
                    "explanation": {},
                },
            ],
        },
    )

    assert result["risk"]["evidence_count"] == 2
    assert len(result["risk"]["reasons"]) == 1
