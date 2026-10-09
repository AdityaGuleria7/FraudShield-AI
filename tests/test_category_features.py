from app.features.category import CategoryFeatureBuilder


def sample_transactions():
    return [
        {
            "transaction_id": "TX001",
            "account_id": "ACC001",
            "merchant_category": "GROCERY",
            "amount": 1000.0,
            "timestamp": "2026-01-01T10:00:00+00:00",
        },
        {
            "transaction_id": "TX002",
            "account_id": "ACC001",
            "merchant_category": "GROCERY",
            "amount": 2000.0,
            "timestamp": "2026-01-01T11:00:00+00:00",
        },
        {
            "transaction_id": "TX003",
            "account_id": "ACC001",
            "merchant_category": "ELECTRONICS",
            "amount": 3000.0,
            "timestamp": "2026-01-01T12:00:00+00:00",
        },
        {
            "transaction_id": "TX004",
            "account_id": "ACC002",
            "merchant_category": "ELECTRONICS",
            "amount": 5000.0,
            "timestamp": "2026-01-01T12:00:00+00:00",
        },
    ]


def test_category_history_uses_only_previous_transactions():
    builder = CategoryFeatureBuilder()

    features = builder.build(sample_transactions())

    assert features["TX001"]["category_transaction_count"] == 0
    assert features["TX001"]["category_total_amount"] == 0.0
    assert features["TX001"]["is_new_category_for_account"] == 1
    assert features["TX001"]["category_share_of_account_spend"] == 0.0

    assert features["TX002"]["category_transaction_count"] == 1
    assert features["TX002"]["category_total_amount"] == 1000.0
    assert features["TX002"]["is_new_category_for_account"] == 0
    assert features["TX002"]["category_share_of_account_spend"] == 1.0


def test_new_category_is_detected():
    builder = CategoryFeatureBuilder()

    features = builder.build(sample_transactions())

    assert features["TX003"]["category_transaction_count"] == 0
    assert features["TX003"]["category_total_amount"] == 0.0
    assert features["TX003"]["is_new_category_for_account"] == 1


def test_category_spend_share_is_account_specific():
    builder = CategoryFeatureBuilder()

    features = builder.build(sample_transactions())

    assert features["TX003"]["category_share_of_account_spend"] == 0.0

    assert features["TX004"]["category_transaction_count"] == 0
    assert features["TX004"]["category_total_amount"] == 0.0
    assert features["TX004"]["category_share_of_account_spend"] == 0.0
