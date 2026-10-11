import json
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

    assert assessment["risk_score"] == 47
    assert assessment["risk_level"] == "MEDIUM"
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


def test_orchestrator_builds_investigation_case(tmp_path):

    transactions = [
        {
            "transaction_id": "TX-CASE-001",
            "account_id": "ACC-CASE-001",
            "device_id": "DEV-CASE-NEW",
            "ip_address": "10.50.50.50",
            "country": "IN",
            "merchant_category": "ELECTRONICS",
            "amount": 85000.00,
            "timestamp": "2026-01-07T10:00:00+00:00",
            "is_fraud": True,
        }
    ]

    orchestrator = FraudShieldOrchestrator(
        audit_path=str(tmp_path / "audit_ledger.jsonl")
    )

    result = orchestrator.run(
        task_id="CASE-ORCH-001",
        data={
            "transactions": transactions,
        },
    )

    case = result["investigation_case"]

    assert case["case_id"].startswith("CASE-")
    assert case["task_id"] == "CASE-ORCH-001"
    assert case["status"] == "OPEN_FOR_INVESTIGATION"
    assert case["risk_score"] == 47
    assert case["risk_level"] == "MEDIUM"
    assert case["decision"] == "REQUIRES_FURTHER_INVESTIGATION"
    assert case["patterns"] == [
        "ACCOUNT_TAKEOVER",
        "NEW_DEVICE_HIGH_VALUE",
    ]
    assert case["evidence_quality"] == "STRONG"
    provenance_by_detector = {
        item["detector"]: item
        for item in case["provenance"]
    }

    assert provenance_by_detector["account_takeover"]["method"] == "deterministic_rule"
    assert provenance_by_detector["account_takeover"]["version"] == "1"
    assert provenance_by_detector["new_device_high_value"]["method"] == "deterministic_rule"
    assert provenance_by_detector["new_device_high_value"]["version"] == "1"
    assert len(case["provenance"]) == 2
    assert case["required_actions"] == [
        "Investigate supporting evidence."
    ]


def test_data_quality_rejection_stops_fraud_pipeline(tmp_path):
    audit_file = tmp_path / "audit_ledger.jsonl"
    orchestrator = FraudShieldOrchestrator(
        audit_path=str(audit_file)
    )

    result = orchestrator.run(
        task_id="DQ-REJECT-001",
        data={
            "transactions": [
                {
                    "transaction_id": "TX-BAD-001",
                    "account_id": "ACC1000",
                    "device_id": "DEV1000",
                    "ip_address": "192.168.1.10",
                    "country": "XX",
                    "merchant_category": "ELECTRONICS",
                    "amount": 1500.0,
                    "timestamp": "2026-01-01T10:00:00+00:00",
                    "is_fraud": False,
                }
            ]
        },
    )

    assert result["data_quality"]["status"] == "REJECT"
    assert result["data_quality"]["valid_transactions"] == 0
    assert result["pattern_findings"] == []
    assert result["evidence_assessment"] is None
    assert result["investigation_case"] is None

    lines = audit_file.read_text(
        encoding="utf-8"
    ).splitlines()

    actions = [
        json.loads(line)["action"]
        for line in lines
    ]

    assert "DATA_QUALITY_VALIDATION" in actions
    assert "EVIDENCE_CORRELATION" not in actions
    assert "INVESTIGATION_CASE_BUILT" not in actions
