import pytest
from pydantic import ValidationError

from app.models.investigation_result import (
    InvestigationResult,
    ModelSignal,
    RiskAssessment,
    ReviewStatus,
)


def make_risk(**overrides):
    data = {
        "score": 47,
        "level": "MEDIUM",
        "decision": "REQUIRES_FURTHER_INVESTIGATION",
        "evidence_quality": "STRONG",
        "evidence_count": 2,
        "reasons": ["Rapid transaction burst detected."],
    }
    data.update(overrides)
    return RiskAssessment(**data)


def test_risk_assessment_is_not_a_fraud_probability():
    risk = make_risk()
    assert risk.score == 47
    assert not hasattr(risk, "fraud_probability")


def test_risk_level_must_match_score():
    with pytest.raises(ValidationError, match="Risk level"):
        make_risk(score=80)


def test_zero_score_requires_no_evidence():
    with pytest.raises(ValidationError, match="zero risk score"):
        RiskAssessment(
            score=0,
            level="LOW",
            decision="NO_SUSPICIOUS_ACTIVITY",
            evidence_quality="STRONG",
            evidence_count=1,
        )


def test_unavailable_model_score_is_explicit():
    signal = ModelSignal()
    assert signal.score is None
    assert signal.score_semantics == "not_available"


def test_uncalibrated_model_score_is_not_called_probability():
    signal = ModelSignal(
        score=0.83,
        score_semantics="uncalibrated_model_score",
        model_approved=False,
    )
    assert signal.score == 0.83
    assert signal.score_semantics == "uncalibrated_model_score"


def test_unapproved_score_cannot_be_labeled_calibrated_probability():
    with pytest.raises(ValidationError, match="explicitly approved"):
        ModelSignal(
            score=0.83,
            score_semantics="calibrated_probability",
            model_approved=False,
        )


def test_completed_result_requires_risk_assessment():
    with pytest.raises(ValidationError, match="requires a risk assessment"):
        InvestigationResult(
            task_id="task-1",
            status="COMPLETED",
            review=ReviewStatus(status="PASS", approved_for_next_stage=True),
        )


def test_result_separates_risk_model_and_review():
    result = InvestigationResult(
        task_id="task-2",
        status="REQUIRES_REVIEW",
        risk=make_risk(),
        model_signal=ModelSignal(
            score=0.71,
            score_semantics="uncalibrated_model_score",
        ),
        review=ReviewStatus(status="PENDING"),
        limitations=["Experimental ML score is not a calibrated probability."],
    )
    payload = result.model_dump()
    assert payload["schema_version"] == "1.0"
    assert payload["risk"]["score"] == 47
    assert payload["model_signal"]["score"] == 0.71
    assert payload["review"]["status"] == "PENDING"
    assert payload["limitations"]


def test_missing_identity_is_not_a_fraud_reason():
    risk = make_risk(reasons=["Identity information is unavailable."])
    assert risk.reasons == ["Identity information is unavailable."]
    assert "fraud" not in risk.reasons[0].lower()


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        make_risk(unreviewed_override=True)
