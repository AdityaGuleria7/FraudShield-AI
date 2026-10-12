import pytest
from pydantic import ValidationError

from app.models.agent_contract import AgentMessage


def make_message(**overrides):
    payload = {
        "agent_id": "TEST_AGENT_001",
        "agent_role": "Test Agent",
        "task_id": "TASK-001",
        "status": "PASS",
        "confidence": 0.95,
        "approved_for_next_stage": True,
    }
    payload.update(overrides)
    return payload


def test_pass_requires_downstream_approval():
    with pytest.raises(ValidationError):
        AgentMessage(
            **make_message(
                status="PASS",
                approved_for_next_stage=False,
            )
        )


def test_conditional_pass_requires_downstream_approval():
    with pytest.raises(ValidationError):
        AgentMessage(
            **make_message(
                status="CONDITIONAL_PASS",
                approved_for_next_stage=False,
            )
        )


def test_reject_cannot_approve_downstream_processing():
    with pytest.raises(ValidationError):
        AgentMessage(
            **make_message(
                status="REJECT",
                approved_for_next_stage=True,
            )
        )


def test_error_cannot_approve_downstream_processing():
    with pytest.raises(ValidationError):
        AgentMessage(
            **make_message(
                status="ERROR",
                approved_for_next_stage=True,
            )
        )


def test_agent_id_cannot_be_blank():
    with pytest.raises(ValidationError):
        AgentMessage(
            **make_message(agent_id="")
        )


def test_agent_role_cannot_be_blank():
    with pytest.raises(ValidationError):
        AgentMessage(
            **make_message(agent_role="")
        )


def test_task_id_cannot_be_blank():
    with pytest.raises(ValidationError):
        AgentMessage(
            **make_message(task_id="")
        )
