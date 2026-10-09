from app.data_quality.validator import DataQualityValidator


def valid_transaction(transaction_id="TX1000"):
    return {
        "transaction_id": transaction_id,
        "account_id": "ACC1000",
        "device_id": "DEV1000",
        "ip_address": "192.168.1.10",
        "country": "IN",
        "merchant_category": "ELECTRONICS",
        "amount": 1500.0,
        "timestamp": "2026-01-01T10:00:00+00:00",
        "is_fraud": False,
    }


def test_valid_transactions_pass():

    validator = DataQualityValidator()

    result = validator.validate(
        [valid_transaction()]
    )

    assert result["status"] == "PASS"
    assert result["quality_score"] == 100
    assert result["total_transactions"] == 1
    assert result["valid_transactions"] == 1
    assert result["issues"] == []


def test_missing_required_field_is_rejected():

    validator = DataQualityValidator()

    transaction = valid_transaction()
    del transaction["account_id"]

    result = validator.validate([transaction])

    assert result["status"] == "REJECT"
    assert result["valid_transactions"] == 0
    assert any(
        issue["type"] == "MISSING_FIELD"
        for issue in result["issues"]
    )


def test_duplicate_transaction_ids_are_detected():

    validator = DataQualityValidator()

    transactions = [
        valid_transaction("TX-DUP-001"),
        valid_transaction("TX-DUP-001"),
    ]

    result = validator.validate(transactions)

    assert result["status"] == "REJECT"
    assert any(
        issue["type"] == "DUPLICATE_TRANSACTION_ID"
        for issue in result["issues"]
    )


def test_negative_amount_is_rejected():

    validator = DataQualityValidator()

    transaction = valid_transaction()
    transaction["amount"] = -100.0

    result = validator.validate([transaction])

    assert result["status"] == "REJECT"
    assert any(
        issue["type"] == "INVALID_AMOUNT"
        for issue in result["issues"]
    )


def test_invalid_country_is_rejected():

    validator = DataQualityValidator()

    transaction = valid_transaction()
    transaction["country"] = "XX"

    result = validator.validate([transaction])

    assert result["status"] == "REJECT"
    assert any(
        issue["type"] == "INVALID_COUNTRY"
        for issue in result["issues"]
    )


def test_invalid_timestamp_is_rejected():

    validator = DataQualityValidator()

    transaction = valid_transaction()
    transaction["timestamp"] = "not-a-timestamp"

    result = validator.validate([transaction])

    assert result["status"] == "REJECT"
    assert any(
        issue["type"] == "INVALID_TIMESTAMP"
        for issue in result["issues"]
    )


def test_missing_field_metrics_are_reported():
    validator = DataQualityValidator()

    transaction = valid_transaction()
    del transaction["account_id"]

    result = validator.validate([transaction])

    assert result["missing_field_count"] == 1
    assert result["missing_field_rate"] == 1.0


def test_unexpected_fields_are_detected():
    validator = DataQualityValidator()

    transaction = valid_transaction()
    transaction["unexpected_field"] = "schema-drift"

    result = validator.validate([transaction])

    assert result["status"] == "REJECT"
    assert any(
        issue["type"] == "UNEXPECTED_FIELD"
        for issue in result["issues"]
    )


def test_transaction_amount_above_maximum_is_rejected():
    validator = DataQualityValidator()

    transaction = valid_transaction()
    transaction["amount"] = 1_000_001

    result = validator.validate([transaction])

    assert result["status"] == "REJECT"
    assert any(
        issue["type"] == "AMOUNT_ABOVE_MAXIMUM"
        for issue in result["issues"]
    )
