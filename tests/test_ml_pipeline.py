from app.ml.pipeline import MLPipeline


def sample_data():
    features = {
        "TX001": {
            "transaction_amount": 1000.0,
            "transactions_last_5m": 0,
            "account_unique_device_count": 1,
        },
        "TX002": {
            "transaction_amount": 5000.0,
            "transactions_last_5m": 3,
            "account_unique_device_count": 2,
        },
        "TX003": {
            "transaction_amount": 1200.0,
            "transactions_last_5m": 0,
            "account_unique_device_count": 1,
        },
        "TX004": {
            "transaction_amount": 9000.0,
            "transactions_last_5m": 5,
            "account_unique_device_count": 3,
        },
    }

    transactions = [
        {"transaction_id": "TX001", "is_fraud": False},
        {"transaction_id": "TX002", "is_fraud": True},
        {"transaction_id": "TX003", "is_fraud": False},
        {"transaction_id": "TX004", "is_fraud": True},
    ]

    return features, transactions


def test_pipeline_trains_evaluates_and_validates_model():
    features, transactions = sample_data()

    pipeline = MLPipeline()

    result = pipeline.run(features, transactions)

    assert result["model_type"] == "logistic_regression"
    assert result["trained"] is True
    assert result["feature_names"] == [
        "account_unique_device_count",
        "transaction_amount",
        "transactions_last_5m",
    ]

    assert len(result["predictions"]) == 4
    assert len(result["probabilities"]) == 4

    assert "precision" in result["metrics"]
    assert "recall" in result["metrics"]
    assert "f1" in result["metrics"]
    assert "roc_auc" in result["metrics"]
    assert "pr_auc" in result["metrics"]

    assert result["validation"]["status"] in ["PASS", "REJECT"]
    assert isinstance(result["validation"]["approved"], bool)


def test_pipeline_returns_dataset_information():
    features, transactions = sample_data()

    pipeline = MLPipeline()

    result = pipeline.run(features, transactions)

    assert result["dataset_size"] == 4
    assert result["fraud_count"] == 2
    assert result["non_fraud_count"] == 2
