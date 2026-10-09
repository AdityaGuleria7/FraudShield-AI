from app.data.transaction_generator import TransactionGenerator
from app.fraud.pattern_engine import FraudPatternEngine
from app.fraud.scenario_generator import FraudScenarioGenerator
from app.agents.fraud_pattern_analyst import (
    FraudPatternAnalystAgent,
)
from app.agents.guardians.fraud_pattern_reviewer import (
    FraudPatternReviewerAgent,
)


def test_fraud_pattern_analyst_produces_evidence():

    transactions = TransactionGenerator(seed=42).generate(10)

    transactions = FraudScenarioGenerator().inject_velocity_attack(
        transactions
    )

    findings = FraudPatternEngine().detect_velocity(
        transactions,
        window_minutes=10,
        minimum_transactions=5,
    )

    agent = FraudPatternAnalystAgent()

    result = agent.analyze(
        task_id="FRAUD-001",
        findings=findings,
    )

    assert result.status == "CONDITIONAL_PASS"
    assert result.approved_for_next_stage is True
    assert len(result.evidence) == 1
    assert (
        result.evidence[0]["pattern"]
        == "VELOCITY_ATTACK"
    )


def test_guardian_accepts_valid_pattern_evidence():

    transactions = TransactionGenerator(seed=42).generate(10)

    transactions = FraudScenarioGenerator().inject_velocity_attack(
        transactions
    )

    findings = FraudPatternEngine().detect_velocity(
        transactions,
        window_minutes=10,
        minimum_transactions=5,
    )

    analyst = FraudPatternAnalystAgent()

    result = analyst.analyze(
        task_id="FRAUD-002",
        findings=findings,
    )

    guardian = FraudPatternReviewerAgent()

    review = guardian.review(result)

    assert review.status == "PASS"
    assert review.approved_for_next_stage is True


def test_guardian_rejects_missing_evidence():

    analyst_result = FraudPatternAnalystAgent().analyze(
        task_id="FRAUD-003",
        findings=[],
    )

    # Simulate a suspicious claim with no supporting evidence.
    analyst_result.evidence = [
        {
            "type": "unsupported_claim",
            "claim": "Fraud detected",
        }
    ]

    review = FraudPatternReviewerAgent().review(
        analyst_result
    )

    assert review.status == "REJECT"
    assert review.approved_for_next_stage is False
