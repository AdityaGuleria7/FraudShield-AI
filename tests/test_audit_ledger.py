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

    lines = audit_file.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 7

    events = [
        json.loads(line)
        for line in lines
    ]

    assert events[0]["action"] == "DATA_ANALYSIS"
    assert events[1]["action"] == "GUARDIAN_REVIEW"
    assert events[2]["action"] == "DATA_QUALITY_VALIDATION"
    assert events[3]["action"] == "EVIDENCE_CORRELATION"
    assert events[4]["action"] == "INVESTIGATION_CASE_BUILT"
    assert events[5]["action"] == "FRAUD_PATTERN_ANALYSIS"
    assert events[6]["action"] == "FRAUD_PATTERN_GUARDIAN_REVIEW"

    assert all(
        event["task_id"] == "AUDIT-001"
        for event in events
    )

    assert events[0]["status"] == "PASS"
    assert events[1]["status"] == "PASS"
    assert events[2]["status"] == "PASS"
    assert events[3]["status"] == "PASS"
    assert events[4]["status"] == "PASS"
