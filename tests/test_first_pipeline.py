from app.core.orchestrator import FraudShieldOrchestrator


def test_first_pipeline():

    orchestrator = FraudShieldOrchestrator()

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
