import pytest

from app.security.operation_registry import OperationRegistry
from app.security.secure_executor import SecureExecutor


def test_fraud_analyst_can_execute_registered_findings_read(tmp_path):
    registry = OperationRegistry()
    findings = [{"pattern": "VELOCITY"}]

    def read_findings():
        return findings

    registry.register(
        action="READ",
        resource="PATTERN_FINDINGS",
        operation=read_findings,
    )

    executor = SecureExecutor(
        operation_registry=registry,
        audit_path=str(tmp_path / "audit.jsonl"),
    )

    decision, result = executor.execute(
        agent_id="FRAUD_PATTERN_ANALYST_001",
        action="READ",
        resource="PATTERN_FINDINGS",
        operation=read_findings,
        task_id="SECURE-FRAUD-001",
    )

    assert decision.allowed is True
    assert result == findings


def test_fraud_analyst_cannot_execute_unregistered_findings_operation(
    tmp_path,
):
    registry = OperationRegistry()
    executed = False

    def read_findings():
        nonlocal executed
        executed = True
        return []

    executor = SecureExecutor(
        operation_registry=registry,
        audit_path=str(tmp_path / "audit.jsonl"),
    )

    with pytest.raises(KeyError):
        executor.execute(
            agent_id="FRAUD_PATTERN_ANALYST_001",
            action="READ",
            resource="PATTERN_FINDINGS",
            operation=read_findings,
            task_id="SECURE-FRAUD-002",
        )

    assert executed is False
