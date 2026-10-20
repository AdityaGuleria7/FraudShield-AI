"""Versioned, normalized result contract for FraudShield investigations."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RiskAssessment(BaseModel):
    """Deterministic evidence score; not a probability of fraud."""

    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=0, le=100)
    level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    decision: Literal[
        "NO_SUSPICIOUS_ACTIVITY",
        "REQUIRES_FURTHER_INVESTIGATION",
    ]
    evidence_quality: Literal["NONE", "PARTIAL", "STRONG"]
    evidence_count: int = Field(ge=0)
    reasons: list[str] = Field(
        default_factory=list,
        description=(
            "Human-readable summaries of findings; these may not map "
            "one-to-one to evidence_count and are not proof of fraud."
        ),
    )

    @model_validator(mode="after")
    def validate_score_consistency(self):
        expected_level = (
            "LOW"
            if self.score == 0
            else "MEDIUM"
            if self.score < 50
            else "HIGH"
            if self.score < 75
            else "CRITICAL"
        )
        if self.level != expected_level:
            raise ValueError("Risk level does not match risk score.")

        expected_decision = (
            "NO_SUSPICIOUS_ACTIVITY"
            if self.score == 0
            else "REQUIRES_FURTHER_INVESTIGATION"
        )
        if self.decision != expected_decision:
            raise ValueError("Decision does not match risk score.")

        if self.score == 0 and self.evidence_count != 0:
            raise ValueError("A zero risk score cannot have recognized evidence.")

        if self.evidence_count == 0 and self.evidence_quality != "NONE":
            raise ValueError("Evidence quality must be NONE when evidence count is zero.")

        return self


class ModelSignal(BaseModel):
    """Optional ML output with explicit semantics and provenance."""

    model_config = ConfigDict(extra="forbid")

    score: float | None = Field(default=None, ge=0.0, le=1.0)
    score_semantics: Literal[
        "uncalibrated_model_score",
        "calibrated_probability",
        "not_available",
    ] = "not_available"
    prediction_provenance: str | None = None
    model_approved: bool = False
    decision_threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_score_semantics(self):
        if self.score is None and self.score_semantics != "not_available":
            raise ValueError("A missing model score must be marked not_available.")
        if self.score is not None and self.score_semantics == "not_available":
            raise ValueError("A present model score must declare its semantics.")
        if (
            self.score_semantics == "calibrated_probability"
            and self.model_approved is not True
        ):
            raise ValueError("A calibrated probability must be explicitly approved.")
        return self


class ReviewStatus(BaseModel):
    """Pipeline gate review, not a final fraud determination.

    PASS means this review gate approved progression to the next stage.
    The overall investigation status indicates whether analyst follow-up
    is still required.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["PENDING", "PASS", "REJECT", "ERROR"]
    approved_for_next_stage: bool = False
    issues: list[str] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_approval_consistency(self):
        if self.status == "PASS" and not self.approved_for_next_stage:
            raise ValueError("PASS review status requires downstream approval.")
        if self.status in {"REJECT", "ERROR", "PENDING"} and self.approved_for_next_stage:
            raise ValueError(
                "REJECT, ERROR, and PENDING review statuses cannot approve downstream processing."
            )
        return self


class InvestigationResult(BaseModel):
    """Stable API-friendly view over an investigation result."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    task_id: str = Field(min_length=1)
    case_id: str | None = None
    status: Literal[
        "COMPLETED",
        "REQUIRES_REVIEW",
        "BLOCKED",
        "REJECTED",
        "ERROR",
    ]
    risk: RiskAssessment | None = None
    model_signal: ModelSignal = Field(default_factory=ModelSignal)
    review: ReviewStatus
    findings: list[dict[str, Any]] = Field(default_factory=list)
    explanation: str | None = None
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_identifiers(self):
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty or whitespace.")
        if self.case_id is not None and not self.case_id.strip():
            raise ValueError("case_id must not be empty or whitespace.")
        return self

    @model_validator(mode="after")
    def validate_result_consistency(self):
        if self.status in {"COMPLETED", "REQUIRES_REVIEW"} and self.risk is None:
            raise ValueError(
                f"A {self.status.lower()} result requires a risk assessment."
            )
        if self.status in {"BLOCKED", "REJECTED", "ERROR"}:
            if self.review.approved_for_next_stage:
                raise ValueError("Blocked, rejected, or errored results cannot be approved.")
        return self
