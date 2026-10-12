import pytest

from app.security.policy_engine import PolicyEngine


@pytest.mark.parametrize(
    "agent_id",
    ["", " ", None],
)
def test_invalid_agent_identity_is_denied(agent_id):
    policy = PolicyEngine()

    decision = policy.check(
        agent_id=agent_id,
        action="READ",
        resource="TRANSACTION_DATA",
    )

    assert decision.allowed is False


@pytest.mark.parametrize(
    "action",
    ["", " ", None],
)
def test_invalid_action_is_denied(action):
    policy = PolicyEngine()

    decision = policy.check(
        agent_id="DATA_ANALYST_001",
        action=action,
        resource="TRANSACTION_DATA",
    )

    assert decision.allowed is False


@pytest.mark.parametrize(
    "resource",
    ["", " ", None],
)
def test_invalid_resource_is_denied(resource):
    policy = PolicyEngine()

    decision = policy.check(
        agent_id="DATA_ANALYST_001",
        action="READ",
        resource=resource,
    )

    assert decision.allowed is False


def test_policy_never_allows_unknown_permission_combination():
    policy = PolicyEngine()

    decision = policy.check(
        agent_id="DATA_ANALYST_001",
        action="READ",
        resource="CUSTOMER_ACCOUNT",
    )

    assert decision.allowed is False
