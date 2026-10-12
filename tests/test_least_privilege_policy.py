import pytest

from app.security.policy_engine import PolicyEngine


@pytest.mark.parametrize(
    "agent_id, action, resource",
    [
        (
            "DATA_ANALYST_001",
            "WRITE",
            "TRANSACTION_DATA",
        ),
        (
            "DATA_ANALYST_001",
            "DELETE",
            "TRANSACTION_DATA",
        ),
        (
            "DATA_ANALYST_001",
            "BLOCK",
            "CUSTOMER_ACCOUNT",
        ),
        (
            "DATA_ANALYST_001",
            "READ",
            "CUSTOMER_PII",
        ),
        (
            "DATA_REVIEWER_001",
            "WRITE",
            "TRANSACTION_DATA",
        ),
        (
            "DATA_REVIEWER_001",
            "DELETE",
            "ANALYSIS_RESULTS",
        ),
        (
            "DATA_REVIEWER_001",
            "BLOCK",
            "CUSTOMER_ACCOUNT",
        ),
        (
            "DATA_REVIEWER_001",
            "READ",
            "CUSTOMER_PII",
        ),
    ],
)
def test_sensitive_capabilities_remain_denied(
    agent_id,
    action,
    resource,
):
    policy = PolicyEngine()

    decision = policy.check(
        agent_id=agent_id,
        action=action,
        resource=resource,
    )

    assert decision.allowed is False


@pytest.mark.parametrize(
    "agent_id, action, resource",
    [
        (
            "UNKNOWN_AGENT",
            "READ",
            "TRANSACTION_DATA",
        ),
        (
            "UNKNOWN_AGENT",
            "WRITE",
            "TRANSACTION_DATA",
        ),
        (
            "UNKNOWN_AGENT",
            "BLOCK",
            "CUSTOMER_ACCOUNT",
        ),
        (
            "UNKNOWN_AGENT",
            "READ",
            "CUSTOMER_PII",
        ),
    ],
)
def test_unknown_agents_are_always_denied(
    agent_id,
    action,
    resource,
):
    policy = PolicyEngine()

    decision = policy.check(
        agent_id=agent_id,
        action=action,
        resource=resource,
    )

    assert decision.allowed is False


def test_data_analyst_has_only_intended_read_resources():
    policy = PolicyEngine()

    permissions = policy.PERMISSIONS["DATA_ANALYST_001"]

    assert set(permissions.keys()) == {"READ"}
    assert permissions["READ"] == {
        "TRANSACTION_DATA",
        "AGGREGATED_DATA",
    }


def test_data_reviewer_has_only_intended_read_resources():
    policy = PolicyEngine()

    permissions = policy.PERMISSIONS["DATA_REVIEWER_001"]

    assert set(permissions.keys()) == {"READ"}
    assert permissions["READ"] == {
        "TRANSACTION_DATA",
        "ANALYSIS_RESULTS",
    }
