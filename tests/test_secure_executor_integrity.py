import json

import pytest

from app.security.secure_executor import SecureExecutor


def test_authorized_operation_failure_is_audited(tmp_path):
    audit_file = tmp_path / "audit_ledger.jsonl"

    executor = SecureExecutor(
        audit_path=str(audit_file)
    )

    def failing_operation():
        raise RuntimeError("simulated execution failure")

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
