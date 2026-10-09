from app.features.behavioral import BehavioralFeatureBuilder


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
        {
            "transaction_id": "TX003",
            "account_id": "ACC002",
            "device_id": "DEV003",
            "ip_address": "192.168.1.12",
            "country": "US",
            "merchant_category": "TRAVEL",
            "amount": 5000.0,
            "timestamp": "2026-01-01T12:00:00+00:00",
            "is_fraud": False,
        },
    ]


def test_builds_account_behavioral_features():
    builder = BehavioralFeatureBuilder()

    features = builder.build(sample_transactions())

    assert features["TX001"]["transaction_amount"] == 1000.0
    assert features["TX001"]["account_transaction_count"] == 2
    assert features["TX001"]["account_total_amount"] == 4000.0
    assert features["TX001"]["account_average_amount"] == 2000.0
    assert features["TX001"]["account_max_amount"] == 3000.0


def test_features_are_computed_per_account():
    builder = BehavioralFeatureBuilder()

    features = builder.build(sample_transactions())

    assert features["TX003"]["account_transaction_count"] == 1
    assert features["TX003"]["account_total_amount"] == 5000.0
    assert features["TX003"]["account_average_amount"] == 5000.0
    assert features["TX003"]["account_max_amount"] == 5000.0
