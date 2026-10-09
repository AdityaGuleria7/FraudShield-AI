from app.security.policy_engine import PolicyEngine


def test_allowed_action():

    policy = PolicyEngine()

    decision = policy.check(
        agent_id="DATA_ANALYST_001",
        action="READ",
        resource="TRANSACTION_DATA",
    )

    assert decision.allowed is True


def test_denied_action():

    policy = PolicyEngine()

    decision = policy.check(
        agent_id="DATA_ANALYST_001",
        action="BLOCK_CUSTOMER_ACCOUNT",
        resource="TRANSACTION_DATA",
    )

    assert decision.allowed is False


def test_unknown_agent_is_denied():

    policy = PolicyEngine()

    decision = policy.check(
        agent_id="UNKNOWN_AGENT",
        action="READ",
        resource="TRANSACTION_DATA",
    )

    assert decision.allowed is False


def test_agent_identity():

    policy = PolicyEngine()

    agent = policy.get_agent("DATA_ANALYST_001")

    assert agent is not None
    assert agent.role == "Data Analyst"
    assert agent.risk_level == "LOW"


def test_guardian_permission():

    policy = PolicyEngine()

    decision = policy.check(
        agent_id="DATA_REVIEWER_001",
        action="READ",
        resource="ANALYSIS_RESULTS",
    )

    assert decision.allowed is True


def test_sensitive_action_denied():

    policy = PolicyEngine()

    decision = policy.check(
        agent_id="DATA_REVIEWER_001",
        action="BLOCK",
        resource="CUSTOMER_ACCOUNT",
    )

    assert decision.allowed is False


def test_allowed_resource():

    policy = PolicyEngine()

    decision = policy.check(
        agent_id="DATA_ANALYST_001",
        action="READ",
        resource="AGGREGATED_DATA",
    )

    assert decision.allowed is True


def test_forbidden_resource():

    policy = PolicyEngine()

    decision = policy.check(
        agent_id="DATA_ANALYST_001",
        action="READ",
        resource="CUSTOMER_PII",
    )

    assert decision.allowed is False


def test_write_is_denied_by_default():

    policy = PolicyEngine()

    decision = policy.check(
        agent_id="DATA_ANALYST_001",
        action="WRITE",
        resource="TRANSACTION_DATA",
    )

    assert decision.allowed is False
