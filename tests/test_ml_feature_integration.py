from app.features.builder import UnifiedFeatureBuilder
from app.ml.pipeline import MLPipeline


def sample_transactions():
    return [
        {
            "transaction_id": "TX001",
            "account_id": "ACC001",
            "device_id": "DEV001",
            "ip_address": "192.168.1.10",
            "country": "IN",
            "merchant_category": "GROCERY",
            "amount": 1000.0,
            "timestamp": "2026-01-01T10:00:00+00:00",
            "is_fraud": False,
        },
        {
            "transaction_id": "TX002",
            "account_id": "ACC001",
            "device_id": "DEV002",
            "ip_address": "192.168.1.11",
            "country": "IN",
            "merchant_category": "ELECTRONICS",
            "amount": 5000.0,
            "timestamp": "2026-01-01T10:05:00+00:00",
            "is_fraud": True,
        },
        {
            "transaction_id": "TX003",
            "account_id": "ACC002",
            "device_id": "DEV003",
            "ip_address": "192.168.2.10",
            "country": "IN",
            "merchant_category": "GROCERY",
            "amount": 1200.0,
            "timestamp": "2026-01-01T11:00:00+00:00",
            "is_fraud": False,
        },
        {
            "transaction_id": "TX004",
            "account_id": "ACC002",
            "device_id": "DEV004",
            "ip_address": "192.168.2.11",
            "country": "US",
            "merchant_category": "TRAVEL",
            "amount": 9000.0,
            "timestamp": "2026-01-01T12:00:00+00:00",
            "is_fraud": True,
        },
    ]


def test_feature_builder_output_can_feed_ml_pipeline():
    transactions = sample_transactions()

    feature_vectors = UnifiedFeatureBuilder().build(transactions)

    result = MLPipeline().run(
        feature_vectors,
        transactions,
    )

    assert result["dataset_size"] == 4
    assert result["fraud_count"] == 2
    assert result["non_fraud_count"] == 2
    assert result["trained"] is True
    assert len(result["predictions"]) == 4
    assert len(result["probabilities"]) == 4
    assert "roc_auc" in result["metrics"]
    assert "pr_auc" in result["metrics"]
