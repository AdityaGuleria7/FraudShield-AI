from app.ml.pipeline import MLPipeline


def sample_data():
    features = {}

    for index in range(1, 11):
        features[f"TX{index:03d}"] = {
            "transaction_amount": float(1000 + index * 500),
            "transactions_last_5m": index % 4,
            "account_unique_device_count": 1 + (index % 3),
        }

    transactions = []

    for index in range(1, 11):
        transactions.append(
            {
                "transaction_id": f"TX{index:03d}",
                "timestamp": (
                    f"2026-01-{index:02d}T10:00:00+00:00"
                ),
                "is_fraud": index % 2 == 0,
            }
        )

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

    assert len(result["predictions"]) == 10
    assert len(result["probabilities"]) == 10

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

    assert result["dataset_size"] == 10
    assert result["fraud_count"] == 5
    assert result["non_fraud_count"] == 5


def test_pipeline_returns_transaction_ids():
    features, transactions = sample_data()

    pipeline = MLPipeline()

    result = pipeline.run(features, transactions)

    assert result["transaction_ids"] == [
        "TX001",
        "TX002",
        "TX003",
        "TX004",
        "TX005",
        "TX006",
        "TX007",
        "TX008",
        "TX009",
        "TX010",
    ]
