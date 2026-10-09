from app.features.velocity import VelocityFeatureBuilder


def sample_transactions():
    return [
        {
            "transaction_id": "TX001",
            "account_id": "ACC001",
            "amount": 1000.0,
            "timestamp": "2026-01-01T10:00:00+00:00",
        },
        {
            "transaction_id": "TX002",
            "account_id": "ACC001",
            "amount": 2000.0,
            "timestamp": "2026-01-01T10:04:00+00:00",
        },
        {
            "transaction_id": "TX003",
            "account_id": "ACC001",
            "amount": 3000.0,
            "timestamp": "2026-01-01T10:09:00+00:00",
        },
        {
            "transaction_id": "TX004",
            "account_id": "ACC001",
            "amount": 4000.0,
            "timestamp": "2026-01-01T11:00:00+00:00",
        },
        {
            "transaction_id": "TX005",
            "account_id": "ACC002",
            "amount": 5000.0,
            "timestamp": "2026-01-01T10:05:00+00:00",
        },
    ]


def test_velocity_features_use_only_prior_transactions():
    builder = VelocityFeatureBuilder()

    features = builder.build(sample_transactions())

    assert features["TX001"]["transactions_last_5m"] == 0
    assert features["TX001"]["amount_last_5m"] == 0.0

    assert features["TX002"]["transactions_last_5m"] == 1
    assert features["TX002"]["amount_last_5m"] == 1000.0

    assert features["TX003"]["transactions_last_5m"] == 1
    assert features["TX003"]["amount_last_5m"] == 2000.0


def test_velocity_features_respect_time_windows():
    builder = VelocityFeatureBuilder()

    features = builder.build(sample_transactions())

    assert features["TX003"]["transactions_last_10m"] == 2
    assert features["TX003"]["amount_last_10m"] == 3000.0

    assert features["TX004"]["transactions_last_1h"] == 3
    assert features["TX004"]["amount_last_1h"] == 6000.0


def test_velocity_features_are_calculated_per_account():
    builder = VelocityFeatureBuilder()

    features = builder.build(sample_transactions())

    assert features["TX005"]["transactions_last_5m"] == 0
    assert features["TX005"]["transactions_last_10m"] == 0
    assert features["TX005"]["transactions_last_1h"] == 0
