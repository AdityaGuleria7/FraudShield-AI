from app.features.builder import UnifiedFeatureBuilder


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
            "amount": 3000.0,
            "timestamp": "2026-01-01T11:00:00+00:00",
            "is_fraud": False,
        },
    ]


def test_builds_unified_feature_vector():
    builder = UnifiedFeatureBuilder()

    features = builder.build(sample_transactions())

    assert "TX001" in features
    assert "TX002" in features

    assert features["TX001"]["transaction_amount"] == 1000.0
    assert features["TX001"]["transactions_last_1h"] == 0
    assert features["TX001"]["account_unique_device_count"] == 1
    assert features["TX001"]["amount_vs_account_average"] == 0.0
    assert features["TX001"]["hour_of_day"] == 10
    assert features["TX001"]["category_transaction_count"] == 0


def test_unified_features_include_all_feature_families():
    builder = UnifiedFeatureBuilder()

    features = builder.build(sample_transactions())

    transaction = features["TX002"]

    assert "transaction_amount" in transaction
    assert "transactions_last_5m" in transaction
    assert "account_unique_device_count" in transaction
    assert "amount_vs_account_average" in transaction
    assert "hour_of_day" in transaction
    assert "category_transaction_count" in transaction
