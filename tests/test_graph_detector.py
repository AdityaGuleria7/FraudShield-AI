from app.fraud.graph_detector import FraudGraphDetector


def test_detects_cluster_with_multiple_identifier_types():
    transactions = [
        {
            "transaction_id": "TX1",
            "account_id": "A1",
            "device_id": "D1",
            "ip_address": "IP1",
            "payment_id": "P1",
        },
        {
            "transaction_id": "TX2",
            "account_id": "A2",
            "device_id": "D1",
            "ip_address": "IP2",
            "payment_id": "P1",
        },
        {
            "transaction_id": "TX3",
            "account_id": "A3",
            "device_id": "D2",
            "ip_address": "IP1",
            "payment_id": "P3",
        },
    ]

    result = FraudGraphDetector().analyze(transactions)

    assert result["account_count"] == 3
    assert result["cluster_count"] == 1
    assert result["review_candidate_count"] == 1

    cluster = result["clusters"][0]
    assert cluster["accounts"] == ["A1", "A2", "A3"]
    assert cluster["status"] == "REVIEW_CANDIDATE"
    assert {"device_id", "ip_address", "payment_id"} == set(
        cluster["shared_identifier_types"]
    )
    assert "does not establish fraud" in cluster["explanation"]


def test_shared_ip_alone_is_not_flagged():
    transactions = [
        {"transaction_id": "TX1", "account_id": "A1", "ip_address": "IP1"},
        {"transaction_id": "TX2", "account_id": "A2", "ip_address": "IP1"},
        {"transaction_id": "TX3", "account_id": "A3", "ip_address": "IP1"},
    ]

    result = FraudGraphDetector().analyze(transactions)

    assert result["cluster_count"] == 1
    assert result["review_candidate_count"] == 0
    assert result["clusters"][0]["status"] == "INSUFFICIENT_EVIDENCE"


def test_isolated_account_and_missing_identifiers():
    transactions = [
        {"transaction_id": "TX1", "account_id": "A1"},
        {"transaction_id": "TX2", "account_id": "A2", "device_id": ""},
        {"transaction_id": "TX3", "account_id": "A3", "device_id": None},
    ]

    result = FraudGraphDetector().analyze(transactions)

    assert result["account_count"] == 3
    assert result["connection_count"] == 0
    assert result["cluster_count"] == 0
    assert result["review_candidate_count"] == 0


def test_invalid_configuration_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        FraudGraphDetector(min_accounts=1)

    with pytest.raises(ValueError):
        FraudGraphDetector(min_shared_identifiers=0)


def test_transaction_to_cluster_mapping():
    transactions = [
        {"transaction_id": "TX1", "account_id": "A1", "device_id": "D1"},
        {"transaction_id": "TX2", "account_id": "A2", "device_id": "D1"},
    ]

    result = FraudGraphDetector().analyze(transactions)

    assert result["transaction_cluster"] == {"TX1": 0, "TX2": 0}
