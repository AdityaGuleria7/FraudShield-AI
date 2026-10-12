import json

import pytest

from app.security.secure_executor import SecureExecutor


def test_authorized_operation_failure_is_audited(tmp_path):
    audit_file = tmp_path / "audit_ledger.jsonl"

    from app.security.operation_registry import OperationRegistry

    registry = OperationRegistry()

    def failing_operation():
        raise RuntimeError("simulated execution failure")

    registry.register(
        action="READ",
        resource="TRANSACTION_DATA",
        operation=failing_operation,
    )

    executor = SecureExecutor(
        operation_registry=registry,
        audit_path=str(audit_file),
    )

    with pytest.raises(RuntimeError, match="simulated execution failure"):
        executor.execute(
            agent_id="DATA_ANALYST_001",
            action="READ",
            resource="TRANSACTION_DATA",
            operation=failing_operation,
            task_id="EXECUTION-FAILURE-001",
        )

    events = [
        json.loads(line)
        for line in audit_file.read_text().splitlines()
    ]

    assert len(events) == 2

    assert events[0]["status"] == "ALLOW"

    assert events[1]["action"] == "ACTION_EXECUTED:READ"
    assert events[1]["status"] == "ERROR"
    assert events[1]["data"]["resource"] == "TRANSACTION_DATA"


def test_allowed_but_unregistered_operation_is_not_executed(tmp_path):
    audit_file = tmp_path / "audit_ledger.jsonl"

    executor = SecureExecutor(
        audit_path=str(audit_file)
    )

    executed = False

    def unregistered_operation():
        nonlocal executed
        executed = True
        return "SHOULD NOT EXECUTE"

    with pytest.raises(KeyError):
        executor.execute(
            agent_id="DATA_ANALYST_001",
            action="READ",
            resource="TRANSACTION_DATA",
            operation=unregistered_operation,
            task_id="EXECUTION-REGISTRY-001",
        )

    assert executed is False
