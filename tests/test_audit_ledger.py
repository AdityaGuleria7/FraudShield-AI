import json

from app.core.orchestrator import FraudShieldOrchestrator


def test_audit_ledger_records_pipeline(tmp_path):

    audit_file = tmp_path / "audit_ledger.jsonl"

    orchestrator = FraudShieldOrchestrator(
        audit_path=str(audit_file)
    )

    orchestrator.run(
        task_id="AUDIT-001",
        data={
            "transaction_id": "TX-AUDIT-001",
            "amount": 1500,
            "country": "IN",
        },
    )

    lines = audit_file.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2

    events = [json.loads(line) for line in lines]

    assert events[0]["action"] == "DATA_ANALYSIS"
    assert events[1]["action"] == "GUARDIAN_REVIEW"

    assert events[0]["task_id"] == "AUDIT-001"
    assert events[1]["task_id"] == "AUDIT-001"

    assert events[0]["status"] == "PASS"
    assert events[1]["status"] == "PASS"
