from app.data.transaction_generator import TransactionGenerator
from app.fraud.pattern_engine import FraudPatternEngine
from app.fraud.scenario_generator import FraudScenarioGenerator


def test_detect_velocity_attack():

    transactions = TransactionGenerator(seed=42).generate(10)

    transactions = FraudScenarioGenerator().inject_velocity_attack(
        transactions
    )

    engine = FraudPatternEngine()

    findings = engine.detect_velocity(
        transactions,
        window_minutes=10,
        minimum_transactions=5,
    )

    velocity_findings = [
        finding
        for finding in findings
        if finding["pattern"] == "VELOCITY_ATTACK"
    ]

    assert len(velocity_findings) == 1
    assert velocity_findings[0]["account_id"] == "ACC9999"
    assert velocity_findings[0]["transaction_count"] == 5


def test_detect_new_device_high_value():

    transactions = TransactionGenerator(seed=42).generate(10)

    transactions = FraudScenarioGenerator().inject_new_device_high_value(
        transactions
    )

    engine = FraudPatternEngine()

    findings = engine.detect_new_device_high_value(
        transactions,
        high_value_threshold=50000,
    )

    device_findings = [
        finding
        for finding in findings
        if finding["pattern"] == "NEW_DEVICE_HIGH_VALUE"
    ]

    assert len(device_findings) == 1

    finding = device_findings[0]

    assert finding["account_id"] == "ACC8888"
    assert finding["device_id"] == "DEV-NEW-999"
    assert finding["amount"] == 95000.00


def test_detect_impossible_travel():

    transactions = [
        {
            "transaction_id": "TX-TRAVEL-001",
            "account_id": "ACC7000",
            "country": "IN",
            "device_id": "DEV-7000",
            "timestamp": "2026-01-01T10:00:00+00:00",
            "amount": 1000.0,
        },
        {
            "transaction_id": "TX-TRAVEL-002",
            "account_id": "ACC7000",
            "country": "US",
            "device_id": "DEV-7000",
            "timestamp": "2026-01-01T11:00:00+00:00",
            "amount": 1200.0,
        },
    ]

    engine = FraudPatternEngine()

    findings = engine.detect_impossible_travel(
        transactions
    )

    assert len(findings) == 1
    assert findings[0]["pattern"] == "IMPOSSIBLE_TRAVEL"
    assert findings[0]["account_id"] == "ACC7000"


def test_detect_account_takeover():

    transactions = [
        {
            "transaction_id": "TX-ATO-001",
            "account_id": "ACC8000",
            "device_id": "DEV-OLD",
            "ip_address": "10.0.0.1",
            "amount": 1000.0,
        },
        {
            "transaction_id": "TX-ATO-002",
            "account_id": "ACC8000",
            "device_id": "DEV-NEW",
            "ip_address": "10.0.0.99",
            "amount": 95000.0,
        },
    ]

    engine = FraudPatternEngine()

    findings = engine.detect_account_takeover(
        transactions
    )

    assert len(findings) == 1
    assert findings[0]["pattern"] == "ACCOUNT_TAKEOVER"
    assert findings[0]["account_id"] == "ACC8000"


def test_fraud_pattern_findings_include_detector_provenance():
    from app.fraud.pattern_engine import FraudPatternEngine

    transactions = [
        {
            "transaction_id": "TX001",
            "account_id": "ACC001",
            "device_id": "DEV001",
            "ip_address": "192.168.1.10",
            "country": "IN",
            "merchant_category": "ELECTRONICS",
            "amount": 95000,
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
            "amount": 100,
            "timestamp": "2026-01-01T10:01:00+00:00",
            "is_fraud": False,
        },
    ]

    findings = FraudPatternEngine().detect_new_device_high_value(
        transactions,
        high_value_threshold=50000,
    )

    assert findings
    assert findings[0]["provenance"]["detector"] == "new_device_high_value"
