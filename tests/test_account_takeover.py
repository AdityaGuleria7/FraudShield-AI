from app.fraud.account_takeover import (
    AccountTakeoverDetector,
)


def test_detect_account_takeover():

    transactions = [
        {
            "transaction_id": "TX-ATO-001",
            "account_id": "ACC5000",
            "device_id": "DEV-OLD",
            "ip_address": "10.0.0.1",
            "amount": 1000.0,
        },
        {
            "transaction_id": "TX-ATO-002",
            "account_id": "ACC5000",
            "device_id": "DEV-NEW",
            "ip_address": "10.0.0.99",
            "amount": 95000.0,
        },
    ]

    detector = AccountTakeoverDetector()

    findings = detector.detect(transactions)

    assert len(findings) == 1

    finding = findings[0]

    assert finding["pattern"] == "ACCOUNT_TAKEOVER"
    assert finding["account_id"] == "ACC5000"
    assert finding["device_id"] == "DEV-NEW"
    assert finding["ip_address"] == "10.0.0.99"
    assert finding["amount"] == 95000.0


def test_normal_transaction_is_not_account_takeover():

    transactions = [
        {
            "transaction_id": "TX-NORMAL-001",
            "account_id": "ACC6000",
            "device_id": "DEV-6000",
            "ip_address": "10.0.0.10",
            "amount": 5000.0,
        },
        {
            "transaction_id": "TX-NORMAL-002",
            "account_id": "ACC6000",
            "device_id": "DEV-6000",
            "ip_address": "10.0.0.10",
            "amount": 6000.0,
        },
    ]

    detector = AccountTakeoverDetector()

    findings = detector.detect(transactions)

    assert findings == []
