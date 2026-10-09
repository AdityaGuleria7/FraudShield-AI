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


def ml_evidence(
    transaction_id="TX002",
    probability=0.87,
    predicted_fraud=True,
):
    return {
        "evidence_type": "ml_model_prediction",
        "transaction_id": transaction_id,
        "model_type": "logistic_regression",
        "predicted_fraud": predicted_fraud,
        "fraud_probability": probability,
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


def test_high_probability_ml_corroboration_is_strong():
    assessment = EvidenceEngine().assess(
        [
            deterministic_finding(),
            ml_evidence(probability=0.87),
        ]
    )

    assert assessment["ml_corroboration"] == ["TX002"]
    assert assessment["ml_corroboration_quality"] == "STRONG"


def test_lower_probability_ml_corroboration_is_weak():
    assessment = EvidenceEngine().assess(
        [
            deterministic_finding(),
            ml_evidence(probability=0.62),
        ]
    )

    assert assessment["ml_corroboration"] == ["TX002"]
    assert assessment["ml_corroboration_quality"] == "WEAK"


def test_non_fraud_ml_prediction_is_not_corroboration():
    assessment = EvidenceEngine().assess(
        [
            deterministic_finding(),
            ml_evidence(
                probability=0.20,
                predicted_fraud=False,
            ),
        ]
    )

    assert assessment["ml_corroboration"] == []
    assert assessment["ml_corroboration_quality"] == "NONE"
