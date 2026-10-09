from app.security.policy_engine import PolicyEngine


def test_allowed_action():

    policy = PolicyEngine()

    decision = policy.check(
        agent_id="DATA_ANALYST_001",
        action="READ_TRANSACTION_DATA",
    )

    assert decision.allowed is True


def test_denied_action():

    policy = PolicyEngine()

    decision = policy.check(
        agent_id="DATA_ANALYST_001",
        action="BLOCK_CUSTOMER_ACCOUNT",
    )

    assert decision.allowed is False


def test_unknown_agent_is_denied():

    policy = PolicyEngine()

    decision = policy.check(
        agent_id="UNKNOWN_AGENT",
        action="READ_TRANSACTION_DATA",
    )

    assert decision.allowed is False
