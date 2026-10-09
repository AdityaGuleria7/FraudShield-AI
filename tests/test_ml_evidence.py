from app.ml.evidence import MLEvidenceBuilder


def sample_ml_result():
    return {
        "model_type": "logistic_regression",
        "feature_names": [
            "account_unique_device_count",
            "transaction_amount",
            "transactions_last_5m",
        ],
        "trained": True,
        "transaction_ids": ["TX001", "TX002"],
        "predictions": [0, 1],
        "probabilities": [0.12, 0.87],
        "validation_transaction_ids": ["TX001", "TX002"],
        "validation_predictions": [0, 1],
        "validation_probabilities": [0.12, 0.87],
        "prediction_provenance": "in_sample",
        "validation_prediction_provenance": "out_of_sample",
        "metrics": {
            "precision": 0.80,
            "recall": 0.75,
            "f1": 0.77,
            "roc_auc": 0.90,
            "pr_auc": 0.70,
        },
        "validation": {
            "status": "PASS",
            "approved": True,
            "issues": [],
        },
        "dataset_size": 2,
        "fraud_count": 1,
        "non_fraud_count": 1,
    }


def test_ml_evidence_builder_creates_structured_evidence():
    result = sample_ml_result()

    evidence = MLEvidenceBuilder().build(
        ml_result=result,
        transaction_id="TX002",
    )

    assert evidence["evidence_type"] == "ml_model_prediction"
    assert evidence["transaction_id"] == "TX002"
    assert evidence["model_type"] == "logistic_regression"
    assert evidence["predicted_fraud"] is True
    assert evidence["fraud_probability"] == 0.87
    assert evidence["prediction_provenance"] == "out_of_sample"
    assert evidence["validation_status"] == "PASS"
    assert evidence["model_approved"] is True


def test_unvalidated_model_cannot_produce_approved_evidence():
    result = sample_ml_result()
    result["validation"] = {
        "status": "REJECT",
        "approved": False,
        "issues": ["precision"],
    }

    evidence = MLEvidenceBuilder().build(
        ml_result=result,
        transaction_id="TX002",
    )

    assert evidence["validation_status"] == "REJECT"
    assert evidence["model_approved"] is False


def test_missing_transaction_prediction_is_rejected():
    result = sample_ml_result()

    try:
        MLEvidenceBuilder().build(
            ml_result=result,
            transaction_id="TX999",
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "No out_of_sample ML prediction" in str(exc)
