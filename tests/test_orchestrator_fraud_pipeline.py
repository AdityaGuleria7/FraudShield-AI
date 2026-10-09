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


def test_orchestrator_detects_impossible_travel(tmp_path):

    transactions = [
        {
            "transaction_id": "TX-TRAVEL-001",
            "account_id": "ACC9000",
            "device_id": "DEV-9000",
            "ip_address": "192.168.1.10",
            "country": "IN",
            "merchant_category": "ONLINE_SERVICES",
            "amount": 1500.0,
            "timestamp": "2026-01-01T10:00:00+00:00",
            "is_fraud": False,
        },
        {
            "transaction_id": "TX-TRAVEL-002",
            "account_id": "ACC9000",
            "device_id": "DEV-9000",
            "ip_address": "192.168.1.11",
            "country": "US",
            "merchant_category": "ONLINE_SERVICES",
            "amount": 1800.0,
            "timestamp": "2026-01-01T11:00:00+00:00",
            "is_fraud": False,
        },
    ]

    audit_file = tmp_path / "audit_ledger.jsonl"

    orchestrator = FraudShieldOrchestrator(
        audit_path=str(audit_file)
    )

    result = orchestrator.run(
        task_id="ORCH-TRAVEL-001",
        data={
            "transactions": transactions,
        },
    )

    travel_findings = [
        finding
        for finding in result["pattern_findings"]
        if finding["pattern"] == "IMPOSSIBLE_TRAVEL"
    ]

    assert len(travel_findings) == 1
    assert travel_findings[0]["account_id"] == "ACC9000"

    assert result["pattern_analysis"].status == "CONDITIONAL_PASS"
    assert result["pattern_review"].status == "PASS"


def test_orchestrator_detects_account_takeover(tmp_path):
    from app.core.orchestrator import FraudShieldOrchestrator

    transactions = [
        {
            "transaction_id": "TX-ATO-001",
            "account_id": "ACC-ATO-001",
            "device_id": "DEV-NEW-ATO",
            "ip_address": "10.10.10.99",
            "country": "IN",
            "merchant_category": "ELECTRONICS",
            "amount": 85000.00,
            "timestamp": "2026-01-05T10:00:00+00:00",
            "is_fraud": True,
        }
    ]

    orchestrator = FraudShieldOrchestrator(
        audit_path=str(tmp_path / "audit_ledger.jsonl")
    )

    result = orchestrator.run(
        task_id="ATO-ORCH-001",
        data={
            "transactions": transactions,
        },
    )

    account_takeover_findings = [
        finding
        for finding in result["pattern_findings"]
        if finding["pattern"] == "ACCOUNT_TAKEOVER"
    ]

    assert len(account_takeover_findings) == 1
    assert (
        account_takeover_findings[0]["account_id"]
        == "ACC-ATO-001"
    )

    assert result["pattern_analysis"].status == "CONDITIONAL_PASS"
    assert result["pattern_review"].status == "PASS"
    assert result["pattern_review"].approved_for_next_stage is True


def test_orchestrator_returns_evidence_assessment(tmp_path):
    from app.core.orchestrator import FraudShieldOrchestrator

    transactions = [
        {
            "transaction_id": "TX-EVIDENCE-001",
            "account_id": "ACC-EVIDENCE-001",
            "device_id": "DEV-NEW-EVIDENCE",
            "ip_address": "10.20.30.40",
            "country": "IN",
            "merchant_category": "ELECTRONICS",
            "amount": 85000.00,
            "timestamp": "2026-01-06T10:00:00+00:00",
            "is_fraud": True,
        }
    ]

    orchestrator = FraudShieldOrchestrator(
        audit_path=str(tmp_path / "audit_ledger.jsonl")
    )

    result = orchestrator.run(
        task_id="EVIDENCE-ORCH-001",
        data={
            "transactions": transactions,
        },
    )

    assessment = result["evidence_assessment"]

    assert assessment["risk_score"] == 60
    assert assessment["risk_level"] == "HIGH"
    assert assessment["patterns"] == [
        "ACCOUNT_TAKEOVER",
        "NEW_DEVICE_HIGH_VALUE",
    ]
    assert assessment["evidence_count"] == 2
    assert assessment["evidence_quality"] == "STRONG"
    assert (
        assessment["decision"]
        == "REQUIRES_FURTHER_INVESTIGATION"
    )
