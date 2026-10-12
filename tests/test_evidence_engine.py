from app.fraud.evidence_engine import EvidenceEngine


def test_overlapping_patterns_are_identified_by_transaction():

    engine = EvidenceEngine()

    findings = [
        {
            "pattern": "ACCOUNT_TAKEOVER",
            "account_id": "ACC5000",
            "transaction_id": "TX5000",
        },
        {
            "pattern": "NEW_DEVICE_HIGH_VALUE",
            "account_id": "ACC5000",
            "transaction_id": "TX5000",
        },
    ]

    assessment = engine.assess(findings)

    assert assessment["overlapping_transactions"] == {
        "TX5000": [
            "ACCOUNT_TAKEOVER",
            "NEW_DEVICE_HIGH_VALUE",
        ]
    }


def test_patterns_on_different_transactions_are_independent():

    engine = EvidenceEngine()

    findings = [
        {
            "pattern": "ACCOUNT_TAKEOVER",
            "account_id": "ACC5000",
            "transaction_id": "TX5001",
        },
        {
            "pattern": "NEW_DEVICE_HIGH_VALUE",
            "account_id": "ACC5000",
            "transaction_id": "TX5002",
        },
    ]

    assessment = engine.assess(findings)

    assert assessment["overlapping_transactions"] == {}


def test_no_findings_produce_low_risk():

    engine = EvidenceEngine()

    assessment = engine.assess([])

    assert assessment["risk_score"] == 0
    assert assessment["risk_level"] == "LOW"
    assert assessment["patterns"] == []
    assert assessment["decision"] == "NO_SUSPICIOUS_ACTIVITY"


def test_single_pattern_produces_explainable_risk():

    engine = EvidenceEngine()

    findings = [
        {
            "pattern": "ACCOUNT_TAKEOVER",
            "account_id": "ACC1000",
        }
    ]

    assessment = engine.assess(findings)

    assert assessment["risk_score"] == 35
    assert assessment["risk_level"] == "MEDIUM"
    assert assessment["patterns"] == ["ACCOUNT_TAKEOVER"]
    assert assessment["evidence_count"] == 1
    assert (
        assessment["decision"]
        == "REQUIRES_FURTHER_INVESTIGATION"
    )


def test_multiple_independent_patterns_increase_risk():

    engine = EvidenceEngine()

    findings = [
        {
            "pattern": "ACCOUNT_TAKEOVER",
            "account_id": "ACC1000",
        },
        {
            "pattern": "IMPOSSIBLE_TRAVEL",
            "account_id": "ACC1000",
        },
    ]

    assessment = engine.assess(findings)

    assert assessment["risk_score"] == 65
    assert assessment["risk_level"] == "HIGH"
    assert assessment["patterns"] == [
        "ACCOUNT_TAKEOVER",
        "IMPOSSIBLE_TRAVEL",
    ]


def test_duplicate_pattern_does_not_inflate_score():

    engine = EvidenceEngine()

    findings = [
        {
            "pattern": "ACCOUNT_TAKEOVER",
            "account_id": "ACC1000",
        },
        {
            "pattern": "ACCOUNT_TAKEOVER",
            "account_id": "ACC1000",
        },
    ]

    assessment = engine.assess(findings)

    assert assessment["risk_score"] == 35
    assert assessment["evidence_count"] == 2
    assert assessment["patterns"] == ["ACCOUNT_TAKEOVER"]


def test_risk_score_is_capped_at_100():

    engine = EvidenceEngine()

    findings = [
        {"pattern": "ACCOUNT_TAKEOVER"},
        {"pattern": "IMPOSSIBLE_TRAVEL"},
        {"pattern": "VELOCITY_ATTACK"},
        {"pattern": "NEW_DEVICE_HIGH_VALUE"},
    ]

    assessment = engine.assess(findings)

    assert assessment["risk_score"] == 100
    assert assessment["risk_level"] == "CRITICAL"


def test_complete_account_takeover_evidence_is_strong():

    engine = EvidenceEngine()

    findings = [
        {
            "pattern": "ACCOUNT_TAKEOVER",
            "account_id": "ACC1000",
            "transaction_id": "TX1000",
            "device_id": "DEV-NEW",
            "ip_address": "10.0.0.99",
            "amount": 85000.0,
            "signals": {
                "new_device": True,
                "new_ip": True,
                "high_value": True,
            },
        }
    ]

    assessment = engine.assess(findings)

    assert assessment["evidence_quality"] == "STRONG"
    assert (
        assessment["evidence_summary"]
        == "All recognized findings contain the required "
        "supporting evidence."
    )


def test_incomplete_account_takeover_evidence_is_partial():

    engine = EvidenceEngine()

    findings = [
        {
            "pattern": "ACCOUNT_TAKEOVER",
            "account_id": "ACC1000",
        }
    ]

    assessment = engine.assess(findings)

    assert assessment["evidence_quality"] == "PARTIAL"
    assert (
        assessment["evidence_summary"]
        == "Some recognized findings are missing "
        "expected supporting evidence."
    )


def test_unknown_pattern_is_not_treated_as_evidence():

    engine = EvidenceEngine()

    findings = [
        {
            "pattern": "UNKNOWN_PATTERN",
            "account_id": "ACC1000",
        }
    ]

    assessment = engine.assess(findings)

    assert assessment["risk_score"] == 0
    assert assessment["evidence_quality"] == "NONE"
    assert assessment["patterns"] == []
    assert assessment["decision"] == "NO_SUSPICIOUS_ACTIVITY"


def test_complete_velocity_evidence_is_strong():

    engine = EvidenceEngine()

    findings = [
        {
            "pattern": "VELOCITY_ATTACK",
            "account_id": "ACC2000",
            "transaction_count": 6,
            "window_minutes": 10,
            "transaction_ids": [
                "TX2001",
                "TX2002",
                "TX2003",
                "TX2004",
                "TX2005",
                "TX2006",
            ],
        }
    ]

    assessment = engine.assess(findings)

    assert assessment["evidence_quality"] == "STRONG"


def test_complete_new_device_high_value_evidence_is_strong():

    engine = EvidenceEngine()

    findings = [
        {
            "pattern": "NEW_DEVICE_HIGH_VALUE",
            "account_id": "ACC3000",
            "transaction_id": "TX3000",
            "device_id": "DEV-NEW-3000",
            "amount": 75000.0,
            "threshold": 50000.0,
        }
    ]

    assessment = engine.assess(findings)

    assert assessment["evidence_quality"] == "STRONG"


def test_complete_impossible_travel_evidence_is_strong():

    engine = EvidenceEngine()

    findings = [
        {
            "pattern": "IMPOSSIBLE_TRAVEL",
            "account_id": "ACC4000",
            "from_country": "IN",
            "to_country": "US",
            "from_transaction_id": "TX4001",
            "to_transaction_id": "TX4002",
            "distance_km": 12000.0,
            "elapsed_hours": 1.0,
            "required_speed_kmh": 12000.0,
        }
    ]

    assessment = engine.assess(findings)

    assert assessment["evidence_quality"] == "STRONG"


def test_complete_velocity_evidence_is_strong():

    engine = EvidenceEngine()

    findings = [
        {
            "pattern": "VELOCITY_ATTACK",
            "account_id": "ACC2000",
            "transaction_count": 6,
            "window_minutes": 10,
            "transaction_ids": [
                "TX2001",
                "TX2002",
                "TX2003",
                "TX2004",
                "TX2005",
                "TX2006",
            ],
        }
    ]

    assessment = engine.assess(findings)

    assert assessment["evidence_quality"] == "STRONG"


def test_complete_new_device_high_value_evidence_is_strong():

    engine = EvidenceEngine()

    findings = [
        {
            "pattern": "NEW_DEVICE_HIGH_VALUE",
            "account_id": "ACC3000",
            "transaction_id": "TX3000",
            "device_id": "DEV-NEW-3000",
            "amount": 75000.0,
            "threshold": 50000.0,
        }
    ]

    assessment = engine.assess(findings)

    assert assessment["evidence_quality"] == "STRONG"


def test_complete_impossible_travel_evidence_is_strong():

    engine = EvidenceEngine()

    findings = [
        {
            "pattern": "IMPOSSIBLE_TRAVEL",
            "account_id": "ACC4000",
            "from_country": "IN",
            "to_country": "US",
            "from_transaction_id": "TX4001",
            "to_transaction_id": "TX4002",
            "distance_km": 12000.0,
            "elapsed_hours": 1.0,
            "required_speed_kmh": 12000.0,
        }
    ]

    assessment = engine.assess(findings)

    assert assessment["evidence_quality"] == "STRONG"


def test_overlapping_patterns_do_not_receive_full_combined_weight():

    engine = EvidenceEngine()

    findings = [
        {
            "pattern": "ACCOUNT_TAKEOVER",
            "account_id": "ACC6000",
            "transaction_id": "TX6000",
        },
        {
            "pattern": "NEW_DEVICE_HIGH_VALUE",
            "account_id": "ACC6000",
            "transaction_id": "TX6000",
        },
    ]

    assessment = engine.assess(findings)

    assert assessment["risk_score"] == 47


def test_same_transaction_patterns_are_not_counted_as_independent_streams():
    from app.fraud.evidence_engine import EvidenceEngine

    findings = [
        {
            "pattern": "VELOCITY_ATTACK",
            "transaction_id": "TX100",
            "account_id": "ACC100",
            "transaction_count": 5,
            "window_minutes": 5,
            "transaction_ids": ["TX096", "TX097", "TX098", "TX099", "TX100"],
        },
        {
            "pattern": "NEW_DEVICE_HIGH_VALUE",
            "transaction_id": "TX100",
            "account_id": "ACC100",
            "device_id": "DEV999",
            "amount": 95000,
            "threshold": 50000,
        },
    ]

    result = EvidenceEngine().assess(findings)

    assert result["independent_evidence_streams"] == 1

def test_evidence_assessment_preserves_detector_provenance():
    engine = EvidenceEngine()

    findings = [
        {
            "pattern": "ACCOUNT_TAKEOVER",
            "account_id": "ACC5000",
            "transaction_id": "TX5000",
            "provenance": {
                "detector": "account_takeover",
                "method": "deterministic_rule",
                "version": "1",
            },
        }
    ]

    assessment = engine.assess(findings)

    assert assessment["provenance"][0]["detector"] == "account_takeover"
    assert assessment["provenance"][0]["method"] == "deterministic_rule"
    assert assessment["provenance"][0]["version"] == "1"

def test_evidence_assessment_rejects_mismatched_detector_provenance():
    engine = EvidenceEngine()

    findings = [
        {
            "pattern": "ACCOUNT_TAKEOVER",
            "account_id": "ACC7001",
            "transaction_id": "TX7001",
            "provenance": {
                "detector": "velocity",
                "method": "deterministic_rule",
                "version": "1",
            },
        }
    ]

    try:
        engine.assess(findings)
    except ValueError as exc:
        assert "provenance" in str(exc).lower()
    else:
        raise AssertionError(
            "Mismatched detector provenance must be rejected"
        )

def test_evidence_assessment_rejects_provenance_without_detector():
    engine = EvidenceEngine()

    findings = [
        {
            "pattern": "ACCOUNT_TAKEOVER",
            "account_id": "ACC7002",
            "transaction_id": "TX7002",
            "provenance": {
                "method": "deterministic_rule",
                "version": "1",
            },
        }
    ]

    try:
        engine.assess(findings)
    except ValueError as exc:
        assert "detector provenance does not match" in str(exc).lower()
    else:
        raise AssertionError(
            "Provenance without a detector must be rejected"
        )

def test_evidence_assessment_rejects_unsupported_provenance_method():
    engine = EvidenceEngine()

    findings = [
        {
            "pattern": "ACCOUNT_TAKEOVER",
            "account_id": "ACC7003",
            "transaction_id": "TX7003",
            "provenance": {
                "detector": "account_takeover",
                "method": "llm_generated",
                "version": "1",
            },
        }
    ]

    try:
        engine.assess(findings)
    except ValueError as exc:
        assert "method" in str(exc).lower()
    else:
        raise AssertionError(
            "Unsupported provenance method must be rejected"
        )

def test_evidence_assessment_rejects_unsupported_provenance_version():
    engine = EvidenceEngine()

    findings = [
        {
            "pattern": "ACCOUNT_TAKEOVER",
            "account_id": "ACC7004",
            "transaction_id": "TX7004",
            "provenance": {
                "detector": "account_takeover",
                "method": "deterministic_rule",
                "version": "999",
            },
        }
    ]

    try:
        engine.assess(findings)
    except ValueError as exc:
        assert "version" in str(exc).lower()
    else:
        raise AssertionError(
            "Unsupported provenance version must be rejected"
        )

def test_evidence_assessment_rejects_provenance_without_method():
    engine = EvidenceEngine()

    findings = [
        {
            "pattern": "ACCOUNT_TAKEOVER",
            "account_id": "ACC7005",
            "transaction_id": "TX7005",
            "provenance": {
                "detector": "account_takeover",
                "version": "1",
            },
        }
    ]

    try:
        engine.assess(findings)
    except ValueError as exc:
        assert "method" in str(exc).lower()
    else:
        raise AssertionError(
            "Provenance without a method must be rejected"
        )
