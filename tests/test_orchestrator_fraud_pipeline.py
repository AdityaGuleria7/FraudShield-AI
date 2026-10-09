from app.core.orchestrator import FraudShieldOrchestrator
from app.data.transaction_generator import TransactionGenerator
from app.fraud.scenario_generator import FraudScenarioGenerator


def test_orchestrator_runs_fraud_pattern_pipeline(tmp_path):

    transactions = TransactionGenerator(seed=42).generate(10)

    transactions = FraudScenarioGenerator().inject_velocity_attack(
        transactions
    )

    audit_file = tmp_path / "audit_ledger.jsonl"

    orchestrator = FraudShieldOrchestrator(
        audit_path=str(audit_file)
    )

    result = orchestrator.run(
        task_id="ORCH-FRAUD-001",
        data={
            "transactions": transactions,
        },
    )

    assert len(result["pattern_findings"]) >= 1

    assert result["pattern_analysis"].status == "CONDITIONAL_PASS"

    assert result["pattern_analysis"].approved_for_next_stage is True

    assert result["pattern_review"].status == "PASS"

    assert result["pattern_review"].approved_for_next_stage is True
