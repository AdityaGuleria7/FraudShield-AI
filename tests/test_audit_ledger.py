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

    assert len(lines) == 8

    events = [
        json.loads(line)
        for line in lines
    ]

    assert events[0]["action"] == "DATA_ANALYSIS"
    assert events[1]["action"] == "GUARDIAN_REVIEW"
    assert events[2]["action"] == "DATA_QUALITY_VALIDATION"
    assert events[3]["action"] == "GRAPH_ANALYSIS"
    assert events[3]["agent_id"] == "FRAUD_GRAPH_DETECTOR_001"
    assert events[3]["status"] == "PASS"
    assert events[4]["action"] == "EVIDENCE_CORRELATION"
    assert events[5]["action"] == "INVESTIGATION_CASE_BUILT"
    assert events[6]["action"] == "FRAUD_PATTERN_ANALYSIS"
    assert events[7]["action"] == "FRAUD_PATTERN_GUARDIAN_REVIEW"

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


def test_audit_ledger_rejects_unknown_status(tmp_path):
    from app.audit.ledger import AuditLedger

    ledger = AuditLedger(str(tmp_path / "audit.jsonl"))

    try:
        ledger.record(
            agent_id="AGENT-001",
            agent_role="SECURITY_EXECUTOR",
            task_id="TASK-001",
            action="TEST",
            status="SUCCESS_WHATEVER",
            confidence=1.0,
            data={},
        )
    except ValueError as exc:
        assert "status" in str(exc).lower()
    else:
        raise AssertionError("Expected unsupported status to be rejected")


def test_audit_ledger_accepts_all_allowed_statuses(tmp_path):
    from app.audit.ledger import AuditLedger

    ledger = AuditLedger(str(tmp_path / "audit.jsonl"))

    for status in sorted(ledger.ALLOWED_STATUSES):
        entry = ledger.record(
            agent_id="AGENT-001",
            agent_role="SECURITY_EXECUTOR",
            task_id=f"TASK-{status}",
            action="TEST",
            status=status,
            confidence=1.0,
            data={},
        )
        assert entry["status"] == status


def test_invalid_status_does_not_write_audit_event(tmp_path):
    from app.audit.ledger import AuditLedger

    audit_file = tmp_path / "audit.jsonl"
    ledger = AuditLedger(str(audit_file))

    try:
        ledger.record(
            agent_id="AGENT-001",
            agent_role="SECURITY_EXECUTOR",
            task_id="TASK-INVALID",
            action="TEST",
            status="INVALID",
            confidence=1.0,
            data={},
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid status to be rejected")

    assert not audit_file.exists() or audit_file.read_text() == ""


def test_audit_ledger_rejects_non_dict_data(tmp_path):
    from app.audit.ledger import AuditLedger

    ledger = AuditLedger(str(tmp_path / "audit.jsonl"))

    for invalid_data in ("not-a-dict", ["item"], 123):
        try:
            ledger.record(
                agent_id="AGENT-001",
                agent_role="SECURITY_EXECUTOR",
                task_id="TASK-001",
                action="TEST",
                status="PASS",
                confidence=1.0,
                data=invalid_data,
            )
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(
                f"Expected invalid data type to be rejected: {type(invalid_data).__name__}"
            )


def test_audit_ledger_rejects_invalid_confidence_types(tmp_path):
    from app.audit.ledger import AuditLedger

    ledger = AuditLedger(str(tmp_path / "audit.jsonl"))

    for invalid_confidence in (True, False, "0.9", None):
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
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(
                f"Expected invalid confidence to be rejected: {invalid_confidence!r}"
            )


def test_audit_ledger_rejects_non_json_data_without_writing(tmp_path):
    from app.audit.ledger import AuditLedger

    audit_file = tmp_path / "audit.jsonl"
    ledger = AuditLedger(str(audit_file))

    try:
        ledger.record(
            agent_id="AGENT-001",
            agent_role="SECURITY_EXECUTOR",
            task_id="TASK-JSON",
            action="TEST",
            status="PASS",
            confidence=1.0,
            data={"unsupported": object()},
        )
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("Expected non-JSON data to be rejected")

    assert not audit_file.exists() or audit_file.read_text() == ""


def test_audit_ledger_rejects_non_standard_json_numbers(tmp_path):
    from app.audit.ledger import AuditLedger

    ledger = AuditLedger(str(tmp_path / "audit.jsonl"))

    for invalid_value in (float("nan"), float("inf"), float("-inf")):
        try:
            ledger.record(
                agent_id="AGENT-001",
                agent_role="SECURITY_EXECUTOR",
                task_id="TASK-JSON",
                action="TEST",
                status="PASS",
                confidence=1.0,
                data={"value": invalid_value},
            )
        except ValueError as exc:
            assert "JSON" in str(exc)
        else:
            raise AssertionError(
                f"Expected non-standard JSON number to be rejected: {invalid_value}"
            )


def test_audit_ledger_links_events_with_hashes(tmp_path):
    from app.audit.ledger import AuditLedger

    ledger = AuditLedger(str(tmp_path / "audit.jsonl"))

    first = ledger.record(
        agent_id="AGENT-001",
        agent_role="SECURITY_EXECUTOR",
        task_id="TASK-001",
        action="FIRST_ACTION",
        status="PASS",
        confidence=1.0,
        data={"step": 1},
    )

    second = ledger.record(
        agent_id="AGENT-001",
        agent_role="SECURITY_EXECUTOR",
        task_id="TASK-001",
        action="SECOND_ACTION",
        status="PASS",
        confidence=1.0,
        data={"step": 2},
    )

    assert len(first["event_hash"]) == 64
    assert first["previous_hash"] == "0" * 64
    assert len(second["event_hash"]) == 64
    assert second["previous_hash"] == first["event_hash"]


def test_audit_ledger_verifies_valid_chain(tmp_path):
    from app.audit.ledger import AuditLedger

    ledger = AuditLedger(str(tmp_path / "audit.jsonl"))

    for index in range(3):
        ledger.record(
            agent_id="AGENT-001",
            agent_role="SECURITY_EXECUTOR",
            task_id="TASK-CHAIN",
            action=f"ACTION_{index}",
            status="PASS",
            confidence=1.0,
            data={"step": index},
        )

    assert ledger.verify_chain() is True


def test_audit_ledger_detects_modified_event(tmp_path):
    import json
    from app.audit.ledger import AuditLedger

    audit_file = tmp_path / "audit.jsonl"
    ledger = AuditLedger(str(audit_file))

    ledger.record(
        agent_id="AGENT-001",
        agent_role="SECURITY_EXECUTOR",
        task_id="TASK-TAMPER",
        action="ORIGINAL_ACTION",
        status="PASS",
        confidence=1.0,
        data={"amount": 100},
    )

    event = json.loads(audit_file.read_text().splitlines()[0])
    event["data"]["amount"] = 999

    audit_file.write_text(json.dumps(event) + "\\n", encoding="utf-8")

    assert ledger.verify_chain() is False


def test_audit_ledger_detects_deleted_event(tmp_path):
    from app.audit.ledger import AuditLedger

    audit_file = tmp_path / "audit.jsonl"
    ledger = AuditLedger(str(audit_file))

    for index in range(3):
        ledger.record(
            agent_id="AGENT-001",
            agent_role="SECURITY_EXECUTOR",
            task_id="TASK-DELETE",
            action=f"ACTION_{index}",
            status="PASS",
            confidence=1.0,
            data={"step": index},
        )

    lines = audit_file.read_text(encoding="utf-8").splitlines()
    audit_file.write_text(
        lines[0] + "\n" + lines[2] + "\n",
        encoding="utf-8",
    )

    assert ledger.verify_chain() is False


def test_audit_ledger_rejects_append_to_legacy_chain(tmp_path):
    from app.audit.ledger import AuditLedger

    audit_file = tmp_path / "audit.jsonl"
    audit_file.write_text(
        '{"action":"LEGACY_EVENT","status":"PASS"}\n',
        encoding="utf-8",
    )
    ledger = AuditLedger(str(audit_file))

    try:
        ledger.record(
            agent_id="AGENT-001",
            agent_role="SECURITY_EXECUTOR",
            task_id="TASK-LEGACY",
            action="NEW_EVENT",
            status="PASS",
            confidence=1.0,
            data={},
        )
    except ValueError as exc:
        assert "chain" in str(exc).lower()
    else:
        raise AssertionError("Expected legacy chain append to be rejected")

    assert len(audit_file.read_text(encoding="utf-8").splitlines()) == 1
