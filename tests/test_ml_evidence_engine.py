from app.fraud.evidence_engine import EvidenceEngine


def test_approved_ml_evidence_is_recognized():
    evidence = [
        {
            "evidence_type": "ml_model_prediction",
            "transaction_id": "TX002",
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
    ]

    assessment = EvidenceEngine().assess(evidence)

    assert assessment["ml_evidence_count"] == 1
    assert assessment["approved_ml_evidence_count"] == 1


def test_rejected_ml_evidence_is_not_trusted():
    evidence = [
        {
            "evidence_type": "ml_model_prediction",
            "transaction_id": "TX002",
            "model_type": "logistic_regression",
            "predicted_fraud": True,
            "fraud_probability": 0.87,
            "validation_status": "REJECT",
            "model_approved": False,
            "metrics": {
                "precision": 0.60,
                "recall": 0.75,
                "f1": 0.67,
                "roc_auc": 0.90,
                "pr_auc": 0.70,
            },
        }
    ]

    assessment = EvidenceEngine().assess(evidence)

    assert assessment["ml_evidence_count"] == 1
    assert assessment["approved_ml_evidence_count"] == 0


def test_ml_evidence_does_not_change_deterministic_score_by_itself():
    evidence = [
        {
            "evidence_type": "ml_model_prediction",
            "transaction_id": "TX002",
            "model_type": "logistic_regression",
            "predicted_fraud": True,
            "fraud_probability": 0.99,
            "validation_status": "PASS",
            "model_approved": True,
            "metrics": {
                "precision": 0.99,
                "recall": 0.99,
                "f1": 0.99,
                "roc_auc": 0.99,
                "pr_auc": 0.99,
            },
        }
    ]

    assessment = EvidenceEngine().assess(evidence)

    assert assessment["risk_score"] == 0
    assert assessment["decision"] == "NO_SUSPICIOUS_ACTIVITY"


def test_ml_corroboration_is_not_counted_as_independent_pattern():
    evidence = [
        {
            "pattern": "NEW_DEVICE_HIGH_VALUE",
            "transaction_id": "TX002",
            "account_id": "ACC0001",
            "device_id": "DEV-NEW-999",
            "amount": 95000.0,
            "threshold": 50000.0,
        },
        {
            "evidence_type": "ml_model_prediction",
            "transaction_id": "TX002",
            "model_type": "logistic_regression",
            "predicted_fraud": True,
            "fraud_probability": 0.95,
            "prediction_provenance": "out_of_sample",
            "validation_status": "PASS",
            "model_approved": True,
            "metrics": {
                "precision": 0.80,
                "recall": 0.75,
                "f1": 0.77,
                "roc_auc": 0.90,
                "pr_auc": 0.70,
            },
        },
    ]

    assessment = EvidenceEngine().assess(evidence)

    assert assessment["patterns"] == ["NEW_DEVICE_HIGH_VALUE"]
    assert assessment["risk_score"] == 25
    assert assessment["ml_corroboration"] == ["TX002"]
    assert assessment["ml_corroboration_quality"] == "STRONG"
    assert assessment["independent_evidence_streams"] == 1
