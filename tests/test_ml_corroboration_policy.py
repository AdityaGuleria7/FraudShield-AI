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


def ml_evidence(probability, transaction_id="TX002"):
    return {
        "evidence_type": "ml_model_prediction",
        "transaction_id": transaction_id,
        "model_type": "logistic_regression",
        "predicted_fraud": True,
        "fraud_probability": probability,
        "validation_status": "PASS",
        "model_approved": True,
    }


def test_corroboration_thresholds_are_explicitly_configurable():
    engine = EvidenceEngine(
        ml_strong_threshold=0.90,
        ml_weak_threshold=0.60,
    )

    strong = engine.assess(
        [
            deterministic_finding(),
            ml_evidence(0.92),
        ]
    )

    weak = engine.assess(
        [
            deterministic_finding(),
            ml_evidence(0.70),
        ]
    )

    assert strong["ml_corroboration_quality"] == "STRONG"
    assert weak["ml_corroboration_quality"] == "WEAK"


def test_probability_below_configured_weak_threshold_is_not_corroboration():
    engine = EvidenceEngine(
        ml_strong_threshold=0.90,
        ml_weak_threshold=0.60,
    )

    assessment = engine.assess(
        [
            deterministic_finding(),
            ml_evidence(0.55),
        ]
    )

    assert assessment["ml_corroboration"] == []
    assert assessment["ml_corroboration_quality"] == "NONE"


def test_invalid_threshold_order_is_rejected():
    try:
        EvidenceEngine(
            ml_strong_threshold=0.60,
            ml_weak_threshold=0.80,
        )
    except ValueError:
        return

    raise AssertionError("Invalid ML threshold ordering was accepted")
