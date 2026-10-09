import json

from app.security.secure_executor import SecureExecutor


def test_denied_action_is_not_executed(tmp_path):

    audit_file = tmp_path / "audit_ledger.jsonl"

    executor = SecureExecutor(
        audit_path=str(audit_file)
    )

    executed = False

    def dangerous_operation():
        nonlocal executed
        executed = True
        return "DANGEROUS ACTION EXECUTED"

    decision, result = executor.execute(
        agent_id="DATA_ANALYST_001",
        action="BLOCK",
        resource="CUSTOMER_ACCOUNT",
        operation=dangerous_operation,
        task_id="SECURITY-001",
    )

    assert decision.allowed is False
    assert result is None
    assert executed is False

    events = [
        json.loads(line)
        for line in audit_file.read_text().splitlines()
    ]

    assert len(events) == 1
    assert events[0]["status"] == "DENY"
    assert events[0]["data"]["resource"] == "CUSTOMER_ACCOUNT"


def test_allowed_action_is_executed_and_audited(tmp_path):

    audit_file = tmp_path / "audit_ledger.jsonl"

    executor = SecureExecutor(
        audit_path=str(audit_file)
    )

    executed = False

    def safe_operation():
        nonlocal executed
        executed = True
        return "SAFE ACTION EXECUTED"

    decision, result = executor.execute(
        agent_id="DATA_ANALYST_001",
        action="READ",
        resource="TRANSACTION_DATA",
        operation=safe_operation,
        task_id="SECURITY-002",
    )

    assert decision.allowed is True
    assert result == "SAFE ACTION EXECUTED"
    assert executed is True

    events = [
        json.loads(line)
        for line in audit_file.read_text().splitlines()
    ]

    assert len(events) == 2

    assert events[0]["status"] == "ALLOW"
    assert events[0]["data"]["resource"] == "TRANSACTION_DATA"

    assert events[1]["status"] == "PASS"
    assert events[1]["action"] == "ACTION_EXECUTED:READ"


def test_allowed_action_on_forbidden_resource_is_blocked(tmp_path):

    audit_file = tmp_path / "audit_ledger.jsonl"

    executor = SecureExecutor(
        audit_path=str(audit_file)
    )

    executed = False

    def operation():
        nonlocal executed
        executed = True

    decision, result = executor.execute(
        agent_id="DATA_ANALYST_001",
        action="READ",
        resource="CUSTOMER_PII",
        operation=operation,
        task_id="SECURITY-003",
    )

    assert decision.allowed is False
    assert result is None
    assert executed is False
