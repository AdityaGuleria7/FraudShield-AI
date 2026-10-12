from app.core.orchestrator import FraudShieldOrchestrator


def test_first_pipeline(tmp_path):
    audit_path = tmp_path / "audit_ledger.jsonl"
    orchestrator = FraudShieldOrchestrator(
        audit_path=str(audit_path)
    )

    result = orchestrator.run(
        task_id="TEST-001",
        data={
            "transaction_id": "TX001",
            "amount": 1500,
            "country": "IN",
        },
    )

    assert result["analysis"].status == "PASS"
    assert result["review"].status == "PASS"
    assert result["review"].approved_for_next_stage is True

    # Pipeline audit records must form a valid hash chain.
    assert orchestrator.audit_ledger.verify_chain() is True
    assert audit_path.exists()
    assert len(audit_path.read_text(encoding="utf-8").splitlines()) >= 2
