from app.fraud.evidence_engine import EvidenceEngine


def approved_ml_evidence(transaction_id="TX002"):
    return {
        "evidence_type": "ml_model_prediction",
        "transaction_id": transaction_id,
        "model_type": "logistic_regression",
        "predicted_fraud": True,
        "fraud_probability": 0.87,
        "validation_status": "PASS",
        "model_approved": True,
        "metrics": {
            "precision": 0.80,
            "recall": 0.75,
            "f1": 0.77,
            "roc_auc": 0.90,
            "pr_auc": 0.70,
        },
    }


def account_takeover_finding(transaction_id="TX002"):
    return {
        "pattern": "ACCOUNT_TAKEOVER",
        "account_id": "ACC001",
        "transaction_id": transaction_id,
        "device_id": "DEV999",
        "ip_address": "10.0.0.88",
        "amount": 95000.0,
        "signals": [
            "new_device",
            "new_ip",
            "high_value",
        ],
    }


def test_ml_evidence_is_tracked_alongside_deterministic_evidence():
    evidence = [
        account_takeover_finding(),
        approved_ml_evidence(),
    ]

    assessment = EvidenceEngine().assess(evidence)

    assert assessment["patterns"] == ["ACCOUNT_TAKEOVER"]
    assert assessment["ml_evidence_count"] == 1
    assert assessment["approved_ml_evidence_count"] == 1
    assert assessment["risk_score"] == 35


def test_ml_evidence_does_not_double_count_same_transaction():
    evidence = [
        account_takeover_finding(),
        approved_ml_evidence(),
    ]

    assessment = EvidenceEngine().assess(evidence)

    assert assessment["risk_score"] == 35
    assert assessment["risk_score"] != 35 + 35


def test_rejected_ml_evidence_cannot_corrobate_deterministic_evidence():
    evidence = [
        account_takeover_finding(),
        {
            **approved_ml_evidence(),
            "validation_status": "REJECT",
            "model_approved": False,
        },
    ]

    assessment = EvidenceEngine().assess(evidence)

    assert assessment["approved_ml_evidence_count"] == 0
    assert assessment["risk_score"] == 35
