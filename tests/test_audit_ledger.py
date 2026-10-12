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


def test_audit_ledger_rejects_blank_agent_id(tmp_path):
    from app.audit.ledger import AuditLedger

    ledger = AuditLedger(str(tmp_path / "audit.jsonl"))

    try:
        ledger.record(
            agent_id="   ",
            agent_role="SECURITY_EXECUTOR",
            task_id="TASK-001",
            action="TEST",
            status="PASS",
            confidence=1.0,
            data={},
        )
    except ValueError as exc:
        assert "agent_id" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected blank agent_id to be rejected"
        )


def test_audit_ledger_rejects_blank_agent_role(tmp_path):
    from app.audit.ledger import AuditLedger

    ledger = AuditLedger(str(tmp_path / "audit.jsonl"))

    try:
        ledger.record(
            agent_id="AGENT-001",
            agent_role="   ",
            task_id="TASK-001",
            action="TEST",
            status="PASS",
            confidence=1.0,
            data={},
        )
    except ValueError as exc:
        assert "agent_role" in str(exc).lower()
    else:
        raise AssertionError("Expected blank agent_role to be rejected")


def test_audit_ledger_rejects_blank_task_id(tmp_path):
    from app.audit.ledger import AuditLedger

    ledger = AuditLedger(str(tmp_path / "audit.jsonl"))

    try:
        ledger.record(
            agent_id="AGENT-001",
            agent_role="SECURITY_EXECUTOR",
            task_id="   ",
            action="TEST",
            status="PASS",
            confidence=1.0,
            data={},
        )
    except ValueError as exc:
        assert "task_id" in str(exc).lower()
    else:
        raise AssertionError("Expected blank task_id to be rejected")


def test_audit_ledger_rejects_blank_action(tmp_path):
    from app.audit.ledger import AuditLedger

    ledger = AuditLedger(str(tmp_path / "audit.jsonl"))

    try:
        ledger.record(
            agent_id="AGENT-001",
            agent_role="SECURITY_EXECUTOR",
            task_id="TASK-001",
            action="   ",
            status="PASS",
            confidence=1.0,
            data={},
        )
    except ValueError as exc:
        assert "action" in str(exc).lower()
    else:
        raise AssertionError("Expected blank action to be rejected")


def test_audit_ledger_rejects_invalid_confidence(tmp_path):
    from app.audit.ledger import AuditLedger

    ledger = AuditLedger(str(tmp_path / "audit.jsonl"))

    for invalid_confidence in (-0.1, 1.1):
        try:
            ledger.record(
                agent_id="AGENT-001",
                agent_role="SECURITY_EXECUTOR",
                task_id="TASK-001",
                action="TEST",
                status="PASS",
                confidence=invalid_confidence,
                data={},
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Expected confidence {invalid_confidence} to be rejected"
            )


def test_audit_ledger_rejects_non_finite_confidence(tmp_path):
    import math
    from app.audit.ledger import AuditLedger

    ledger = AuditLedger(str(tmp_path / "audit.jsonl"))

    for invalid_confidence in (math.nan, math.inf, -math.inf):
        try:
            ledger.record(
                agent_id="AGENT-001",
                agent_role="SECURITY_EXECUTOR",
                task_id="TASK-001",
                action="TEST",
                status="PASS",
                confidence=invalid_confidence,
                data={},
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Expected non-finite confidence {invalid_confidence} to be rejected"
            )
