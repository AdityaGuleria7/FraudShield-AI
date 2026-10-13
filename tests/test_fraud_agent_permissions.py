import pytest

from app.security.policy_engine import PolicyEngine


@pytest.mark.parametrize(
    ("agent_id", "resource"),
    [
        ("FRAUD_PATTERN_ANALYST_001", "PATTERN_FINDINGS"),
        ("FRAUD_PATTERN_REVIEWER_001", "PATTERN_ANALYSIS_RESULTS"),
    ],
)
def test_fraud_agents_can_read_assigned_resource(agent_id, resource):
    decision = PolicyEngine().check(agent_id, "READ", resource)

    assert decision.allowed is True


@pytest.mark.parametrize(
    ("agent_id", "resource"),
    [
        ("FRAUD_PATTERN_ANALYST_001", "PATTERN_ANALYSIS_RESULTS"),
        ("FRAUD_PATTERN_REVIEWER_001", "PATTERN_FINDINGS"),
    ],
)
def test_fraud_agents_cannot_read_other_agents_resource(
    agent_id, resource
):
    decision = PolicyEngine().check(agent_id, "READ", resource)

    assert decision.allowed is False


@pytest.mark.parametrize(
    ("agent_id", "action", "resource"),
    [
        (
            "FRAUD_PATTERN_ANALYST_001",
            "WRITE",
            "PATTERN_FINDINGS",
        ),
        (
            "FRAUD_PATTERN_REVIEWER_001",
            "BLOCK",
            "CUSTOMER_ACCOUNT",
        ),
        (
            "FRAUD_PATTERN_ANALYST_001",
            "DELETE",
            "PATTERN_FINDINGS",
        ),
    ],
)
def test_fraud_agents_cannot_perform_unauthorized_actions(
    agent_id, action, resource
):
    decision = PolicyEngine().check(agent_id, action, resource)

    assert decision.allowed is False
