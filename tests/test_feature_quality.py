import math

from app.features.quality import FeatureQualityValidator


REQUIRED_FEATURES = {
    "transaction_amount",
    "transactions_last_5m",
    "transactions_last_10m",
    "transactions_last_1h",
    "amount_last_5m",
    "amount_last_10m",
    "amount_last_1h",
    "account_unique_device_count",
    "account_unique_ip_count",
    "device_unique_account_count",
    "is_new_device_for_account",
    "is_new_ip_for_account",
    "amount_vs_account_average",
    "amount_vs_account_max",
    "is_above_account_average",
    "is_above_account_max",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_night",
    "category_transaction_count",
    "category_total_amount",
    "is_new_category_for_account",
    "category_share_of_account_spend",
}


def valid_features():
    return {
        feature: 1.0
        for feature in REQUIRED_FEATURES
    }


def test_valid_feature_vector_passes():
    validator = FeatureQualityValidator()

    result = validator.validate({"TX001": valid_features()})

    assert result["status"] == "PASS"
    assert result["valid_transactions"] == 1
    assert result["issues"] == []


def test_missing_feature_is_rejected():
    validator = FeatureQualityValidator()

    features = valid_features()
    del features["transaction_amount"]

    result = validator.validate({"TX001": features})

    assert result["status"] == "REJECT"
    assert any(
        issue["type"] == "MISSING_FEATURE"
        and issue["feature"] == "transaction_amount"
        for issue in result["issues"]
    )


def test_none_feature_value_is_rejected():
    validator = FeatureQualityValidator()

    features = valid_features()
    features["transaction_amount"] = None

    result = validator.validate({"TX001": features})

    assert result["status"] == "REJECT"
    assert any(
        issue["type"] == "NULL_FEATURE"
        and issue["feature"] == "transaction_amount"
        for issue in result["issues"]
    )


def test_non_finite_feature_value_is_rejected():
    validator = FeatureQualityValidator()

    features = valid_features()
    features["amount_vs_account_average"] = math.inf

    result = validator.validate({"TX001": features})

    assert result["status"] == "REJECT"
    assert any(
        issue["type"] == "NON_FINITE_FEATURE"
        and issue["feature"] == "amount_vs_account_average"
        for issue in result["issues"]
    )


def test_unexpected_feature_is_rejected():
    validator = FeatureQualityValidator()

    features = valid_features()
    features["unexpected_feature"] = 123.0

    result = validator.validate({"TX001": features})

    assert result["status"] == "REJECT"
    assert any(
        issue["type"] == "UNEXPECTED_FEATURE"
        and issue["feature"] == "unexpected_feature"
        for issue in result["issues"]
    )
