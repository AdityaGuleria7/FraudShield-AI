import pytest
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


def test_orchestrator_rejects_tampered_evidence_count(tmp_path, monkeypatch):
    orchestrator = FraudShieldOrchestrator(
        audit_path=str(tmp_path / "audit.jsonl")
    )

    original_assess = orchestrator.evidence_engine.assess

    def tampered_assess(findings):
        assessment = original_assess(findings)
        assessment["evidence_count"] += 1
        return assessment

    monkeypatch.setattr(
        orchestrator.evidence_engine,
        "assess",
        tampered_assess,
    )

    with pytest.raises(ValueError, match="evidence count"):
        orchestrator.run(
            task_id="ORCH-INTEGRITY-001",
            data={"transactions": []},
        )


def test_orchestrator_rejects_tampered_patterns(tmp_path, monkeypatch):
    orchestrator = FraudShieldOrchestrator(
        audit_path=str(tmp_path / "audit.jsonl")
    )

    original_assess = orchestrator.evidence_engine.assess

    def tampered_assess(findings):
        assessment = original_assess(findings)
        assessment["patterns"] = ["ACCOUNT_TAKEOVER"]
        return assessment

    monkeypatch.setattr(
        orchestrator.evidence_engine,
        "assess",
        tampered_assess,
    )

    with pytest.raises(ValueError, match="patterns"):
        orchestrator.run(
            task_id="ORCH-INTEGRITY-002",
            data={"transactions": []},
        )


def test_orchestrator_rejects_tampered_risk_score(tmp_path, monkeypatch):
    orchestrator = FraudShieldOrchestrator(
        audit_path=str(tmp_path / "audit.jsonl")
    )

    original_assess = orchestrator.evidence_engine.assess

    def tampered_assess(findings):
        assessment = original_assess(findings)
        assessment["risk_score"] = 99
        return assessment

    monkeypatch.setattr(
        orchestrator.evidence_engine,
        "assess",
        tampered_assess,
    )

    with pytest.raises(ValueError, match="risk"):
        orchestrator.run(
            task_id="ORCH-INTEGRITY-003",
            data={"transactions": []},
        )


def test_orchestrator_rejects_tampered_decision(tmp_path, monkeypatch):
    orchestrator = FraudShieldOrchestrator(
        audit_path=str(tmp_path / "audit.jsonl")
    )

    original_assess = orchestrator.evidence_engine.assess

    def tampered_assess(findings):
        assessment = original_assess(findings)
        assessment["decision"] = "REQUIRES_FURTHER_INVESTIGATION"
        return assessment

    monkeypatch.setattr(
        orchestrator.evidence_engine,
        "assess",
        tampered_assess,
    )

    with pytest.raises(ValueError, match="decision"):
        orchestrator.run(
            task_id="ORCH-INTEGRITY-004",
            data={"transactions": []},
        )


def test_orchestrator_stops_when_data_guardian_rejects(
    tmp_path,
    monkeypatch,
):
    orchestrator = FraudShieldOrchestrator(
        audit_path=str(tmp_path / "audit.jsonl")
    )

    original_review = orchestrator.data_reviewer.review

    def rejected_review(analysis):
        review = original_review(analysis)
        review.status = "REJECT"
        review.approved_for_next_stage = False
        review.critical_objections = ["Guardian rejected the analysis."]
        return review

    monkeypatch.setattr(
        orchestrator.data_reviewer,
        "review",
        rejected_review,
    )

    result = orchestrator.run(
        task_id="ORCH-GUARDIAN-001",
        data={"transactions": []},
    )

    assert result["review"].status == "REJECT"
    assert result["review"].approved_for_next_stage is False

    assert result["data_quality"] is None
    assert result["pattern_analysis"] is None
    assert result["pattern_review"] is None
    assert result["pattern_findings"] == []
    assert result["evidence_assessment"] is None
    assert result["investigation_case"] is None


def test_orchestrator_stops_when_fraud_pattern_guardian_rejects(
    tmp_path,
    monkeypatch,
):
    orchestrator = FraudShieldOrchestrator(
        audit_path=str(tmp_path / "audit.jsonl")
    )

    original_review = orchestrator.fraud_pattern_reviewer.review

    def rejected_review(analysis):
        review = original_review(analysis)
        review.status = "REJECT"
        review.approved_for_next_stage = False
        review.critical_objections = [
            "Fraud pattern guardian rejected the analysis."
        ]
        return review

    monkeypatch.setattr(
        orchestrator.fraud_pattern_reviewer,
        "review",
        rejected_review,
    )

    result = orchestrator.run(
        task_id="ORCH-PATTERN-GUARDIAN-001",
        data={"transactions": []},
    )

    assert result["pattern_review"].status == "REJECT"
    assert result["pattern_review"].approved_for_next_stage is False
    assert result["investigation_case"] is None


def test_orchestrator_returns_graph_analysis_separately(tmp_path):
    transactions = [
        {
            "transaction_id": "TX-GRAPH-001",
            "account_id": "ACC-GRAPH-001",
            "device_id": "DEV-SHARED",
            "ip_address": "10.0.0.1",
            "country": "IN",
            "merchant_category": "ONLINE_SERVICES",
            "amount": 1200.0,
            "timestamp": "2026-01-01T10:00:00+00:00",
            "is_fraud": False,
        },
        {
            "transaction_id": "TX-GRAPH-002",
            "account_id": "ACC-GRAPH-002",
            "device_id": "DEV-SHARED",
            "ip_address": "10.0.0.2",
            "country": "IN",
            "merchant_category": "ONLINE_SERVICES",
            "amount": 1300.0,
            "timestamp": "2026-01-01T10:01:00+00:00",
            "is_fraud": False,
        },
        {
            "transaction_id": "TX-GRAPH-003",
            "account_id": "ACC-GRAPH-003",
            "device_id": "DEV-OTHER",
            "ip_address": "10.0.0.1",
            "country": "IN",
            "merchant_category": "ONLINE_SERVICES",
            "amount": 1400.0,
            "timestamp": "2026-01-01T10:02:00+00:00",
            "is_fraud": False,
        },
    ]

    orchestrator = FraudShieldOrchestrator(
        audit_path=str(tmp_path / "audit_graph.jsonl")
    )
    result = orchestrator.run(
        task_id="GRAPH-ORCH-001",
        data={"transactions": transactions},
    )

    graph = result["graph_analysis"]
    assert graph["graph_type"] == "account_shared_identifier_graph"
    assert graph["account_count"] == 3
    assert graph["cluster_count"] == 1
    assert graph["review_candidate_count"] == 1

    # Graph findings remain supplementary and do not alter risk assessment.
    assert result["evidence_assessment"]["risk_score"] == 0
    assert result["evidence_assessment"]["decision"] == "NO_SUSPICIOUS_ACTIVITY"
    assert result["investigation_case"]["risk_score"] == 0
    assert result["investigation_case"]["decision"] == "NO_SUSPICIOUS_ACTIVITY"


def test_orchestrator_does_not_run_graph_after_data_quality_rejection(
    tmp_path, monkeypatch
):
    orchestrator = FraudShieldOrchestrator(
        audit_path=str(tmp_path / "audit_graph_reject.jsonl")
    )

    def should_not_run(_transactions):
        raise AssertionError("Graph analysis must not run after rejection.")

    monkeypatch.setattr(orchestrator.graph_detector, "analyze", should_not_run)

    result = orchestrator.run(
        task_id="GRAPH-ORCH-REJECT-001",
        data={
            "transactions": [
                {
                    "transaction_id": "TX-GRAPH-BAD",
                    "account_id": "ACC-GRAPH-BAD",
                    "device_id": "DEV-BAD",
                    "ip_address": "10.10.10.10",
                    "country": "XX",
                    "merchant_category": "ELECTRONICS",
                    "amount": 1000.0,
                    "timestamp": "2026-01-01T10:00:00+00:00",
                    "is_fraud": False,
                }
            ]
        },
    )

    assert result["data_quality"]["status"] == "REJECT"
    assert result["graph_analysis"] is None
