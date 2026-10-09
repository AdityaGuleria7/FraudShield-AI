from app.fraud.evidence_engine import EvidenceEngine


def deterministic_finding(transaction_id="TX002"):
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


def test_approved_ml_evidence_is_marked_as_correlated():
    assessment = EvidenceEngine().assess(
        [
            deterministic_finding("TX002"),
            approved_ml_evidence("TX002"),
        ]
    )

    assert assessment["ml_corroboration"] == ["TX002"]


def test_ml_evidence_on_different_transaction_is_not_corroboration():
    assessment = EvidenceEngine().assess(
        [
            deterministic_finding("TX002"),
            approved_ml_evidence("TX003"),
        ]
    )

    assert assessment["ml_corroboration"] == []
