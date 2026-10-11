from app.fraud.impossible_travel import (
    ImpossibleTravelDetector,
)


def test_detect_impossible_travel():

    transactions = [
        {
            "transaction_id": "TX-001",
            "account_id": "ACC1000",
            "country": "IN",
            "timestamp": "2026-01-01T10:00:00+00:00",
        },
        {
            "transaction_id": "TX-002",
            "account_id": "ACC1000",
            "country": "US",
            "timestamp": "2026-01-01T11:00:00+00:00",
        },
    ]

    detector = ImpossibleTravelDetector(
        max_speed_kmh=900
    )

    findings = detector.detect(transactions)

    assert len(findings) == 1

    finding = findings[0]

    assert finding["pattern"] == "IMPOSSIBLE_TRAVEL"
    assert finding["account_id"] == "ACC1000"
    assert finding["from_country"] == "IN"
    assert finding["to_country"] == "US"
    assert finding["required_speed_kmh"] > 900


def test_normal_travel_is_not_flagged():

    transactions = [
        {
            "transaction_id": "TX-001",
            "account_id": "ACC2000",
            "country": "IN",
            "timestamp": "2026-01-01T10:00:00+00:00",
        },
        {
            "transaction_id": "TX-002",
            "account_id": "ACC2000",
            "country": "US",
            "timestamp": "2026-01-05T10:00:00+00:00",
        },
    ]

    detector = ImpossibleTravelDetector(
        max_speed_kmh=900
    )

    findings = detector.detect(transactions)

    assert findings == []

def test_impossible_travel_findings_include_detector_provenance():
    transactions = [
        {
            "transaction_id": "TX-PROV-001",
            "account_id": "ACC1000",
            "country": "IN",
            "timestamp": "2026-01-01T10:00:00+00:00",
        },
        {
            "transaction_id": "TX-PROV-002",
            "account_id": "ACC1000",
            "country": "US",
            "timestamp": "2026-01-01T10:01:00+00:00",
        },
    ]

    detector = ImpossibleTravelDetector(max_speed_kmh=900)

    findings = detector.detect(transactions)

    assert findings
    assert findings[0]["provenance"]["detector"] == "impossible_travel"
