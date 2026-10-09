from app.features.builder import UnifiedFeatureBuilder
from app.ml.pipeline import MLPipeline


def sample_transactions():
    return [
        {
            "transaction_id": f"TX{index:03d}",
            "account_id": f"ACC{((index - 1) % 2) + 1:03d}",
            "device_id": f"DEV{index:03d}",
            "ip_address": f"192.168.{((index - 1) % 2) + 1}.{10 + index}",
            "country": "IN" if index < 9 else "US",
            "merchant_category": (
                "GROCERY"
                if index % 2 == 1
                else "ELECTRONICS"
            ),
            "amount": float(1000 + index * 1000),
            "timestamp": (
                f"2026-01-{index:02d}T10:00:00+00:00"
            ),
            "is_fraud": index % 2 == 0,
        }
        for index in range(1, 11)
    ]


def test_feature_builder_output_can_feed_ml_pipeline():
    transactions = sample_transactions()

    feature_vectors = UnifiedFeatureBuilder().build(transactions)

    result = MLPipeline().run(
        feature_vectors,
        transactions,
    )

    assert result["dataset_size"] == 10
    assert result["fraud_count"] == 5
    assert result["non_fraud_count"] == 5
    assert result["trained"] is True
    assert len(result["predictions"]) == 10
    assert len(result["probabilities"]) == 10
    assert "roc_auc" in result["metrics"]
    assert "pr_auc" in result["metrics"]
