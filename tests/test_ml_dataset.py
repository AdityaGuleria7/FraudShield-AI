import pytest

from app.ml.dataset import FraudDatasetBuilder


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
    }

    transactions = [
        {
            "transaction_id": "TX001",
            "is_fraud": False,
        },
        {
            "transaction_id": "TX002",
            "is_fraud": True,
        },
    ]

    return features, transactions


def test_builds_feature_matrix_and_labels():
    features, transactions = sample_data()

    builder = FraudDatasetBuilder()

    dataset = builder.build(features, transactions)

    assert dataset["feature_names"] == [
        "account_unique_device_count",
        "transaction_amount",
        "transactions_last_5m",
    ]

    assert dataset["X"] == [
        [1, 1000.0, 0],
        [2, 5000.0, 3],
    ]

    assert dataset["y"] == [0, 1]


def test_transaction_id_and_label_are_not_features():
    features, transactions = sample_data()

    builder = FraudDatasetBuilder()

    dataset = builder.build(features, transactions)

    assert "transaction_id" not in dataset["feature_names"]
    assert "is_fraud" not in dataset["feature_names"]


def test_missing_feature_vector_is_rejected():
    features, transactions = sample_data()
    del features["TX002"]

    builder = FraudDatasetBuilder()

    with pytest.raises(ValueError, match="Missing feature vector"):
        builder.build(features, transactions)


def test_invalid_label_is_rejected():
    features, transactions = sample_data()
    transactions[1]["is_fraud"] = "fraud"

    builder = FraudDatasetBuilder()

    with pytest.raises(ValueError, match="Invalid fraud label"):
        builder.build(features, transactions)
