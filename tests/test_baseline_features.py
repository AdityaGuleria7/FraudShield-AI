from app.features.baseline import BaselineFeatureBuilder


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
            "timestamp": "2026-01-01T11:00:00+00:00",
        },
        {
            "transaction_id": "TX003",
            "account_id": "ACC001",
            "amount": 6000.0,
            "timestamp": "2026-01-01T12:00:00+00:00",
        },
        {
            "transaction_id": "TX004",
            "account_id": "ACC002",
            "amount": 5000.0,
            "timestamp": "2026-01-01T12:00:00+00:00",
        },
    ]


def test_baseline_uses_only_previous_transactions():
    builder = BaselineFeatureBuilder()

    features = builder.build(sample_transactions())

    assert features["TX001"]["amount_vs_account_average"] == 0.0
    assert features["TX001"]["amount_vs_account_max"] == 0.0
    assert features["TX001"]["is_above_account_average"] == 0
    assert features["TX001"]["is_above_account_max"] == 0

    assert features["TX002"]["amount_vs_account_average"] == 2.0
    assert features["TX002"]["amount_vs_account_max"] == 2.0

    assert features["TX003"]["amount_vs_account_average"] == 4.0
    assert features["TX003"]["amount_vs_account_max"] == 3.0


def test_baseline_flags_unusual_transactions():
    builder = BaselineFeatureBuilder()

    features = builder.build(sample_transactions())

    assert features["TX002"]["is_above_account_average"] == 1
    assert features["TX002"]["is_above_account_max"] == 1

    assert features["TX003"]["is_above_account_average"] == 1
    assert features["TX003"]["is_above_account_max"] == 1


def test_first_transaction_for_account_has_no_baseline():
    builder = BaselineFeatureBuilder()

    features = builder.build(sample_transactions())

    assert features["TX004"]["amount_vs_account_average"] == 0.0
    assert features["TX004"]["amount_vs_account_max"] == 0.0
    assert features["TX004"]["is_above_account_average"] == 0
    assert features["TX004"]["is_above_account_max"] == 0
