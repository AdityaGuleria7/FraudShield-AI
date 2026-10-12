import pytest
from pydantic import ValidationError

from app.models.agent_contract import AgentMessage
from app.agents.guardians.data_reviewer import DataReviewerAgent
from app.agents.guardians.fraud_pattern_reviewer import (
    FraudPatternReviewerAgent,
)


def test_data_guardian_rejects_wrong_agent_identity():
    guardian = DataReviewerAgent()

    message = AgentMessage(
        agent_id="FAKE_AGENT_001",
        agent_role="Fake Agent",
        task_id="IDENTITY-001",
        status="PASS",
        confidence=0.95,
        evidence=[
            {
                "type": "input_validation",
                "source": "FAKE_AGENT_001",
            }
        ],
        approved_for_next_stage=True,
    )

    result = guardian.review(message)

    assert result.status == "REJECT"
    assert result.approved_for_next_stage is False


def test_data_guardian_rejects_wrong_agent_role():
    guardian = DataReviewerAgent()

    message = AgentMessage(
        agent_id="DATA_ANALYST_001",
        agent_role="Fake Role",
        task_id="IDENTITY-002",
        status="PASS",
        confidence=0.95,
        evidence=[
            {
                "type": "input_validation",
                "source": "DATA_ANALYST_001",
            }
        ],
        approved_for_next_stage=True,
    )

    result = guardian.review(message)

    assert result.status == "REJECT"
    assert result.approved_for_next_stage is False


def test_fraud_guardian_rejects_wrong_agent_identity():
    guardian = FraudPatternReviewerAgent()

    message = AgentMessage(
        agent_id="FAKE_AGENT_002",
        agent_role="Fake Agent",
        task_id="IDENTITY-003",
        status="CONDITIONAL_PASS",
        confidence=0.95,
        findings=["Detected suspicious activity."],
        evidence=[
            {
                "type": "deterministic_pattern_detection",
                "pattern": "VELOCITY_ATTACK",
                "details": {
                    "pattern": "VELOCITY_ATTACK",
                },
            }
        ],
        approved_for_next_stage=True,
    )

    result = guardian.review(message)

    assert result.status == "REJECT"
    assert result.approved_for_next_stage is False


def test_fraud_guardian_rejects_wrong_agent_role():
    guardian = FraudPatternReviewerAgent()

    message = AgentMessage(
        agent_id="FRAUD_PATTERN_ANALYST_001",
        agent_role="Fake Role",
        task_id="IDENTITY-004",
        status="CONDITIONAL_PASS",
        confidence=0.95,
        findings=["Detected suspicious activity."],
        evidence=[
            {
                "type": "deterministic_pattern_detection",
                "pattern": "VELOCITY_ATTACK",
                "details": {
                    "pattern": "VELOCITY_ATTACK",
                },
            }
        ],
        approved_for_next_stage=True,
    )

    result = guardian.review(message)

    assert result.status == "REJECT"
    assert result.approved_for_next_stage is False
