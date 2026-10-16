"""Unit tests for the plain-language explanation engine."""
import pytest
from app.explainability.plain_language import (
    explain_finding,
    explain_findings,
    build_executive_summary,
)


def test_explain_velocity_attack():
    finding = {
        "pattern": "VELOCITY_ATTACK",
        "account_id": "ACC-101",
        "transaction_count": 6,
        "window_minutes": 8,
        "transaction_ids": ["T1", "T2", "T3", "T4", "T5", "T6"],
    }
    explanation = explain_finding(finding)

    assert "headline" in explanation
    assert "Rapid transaction burst" in explanation["headline"]
    assert "ACC-101" in explanation["plain_summary"]
    assert "6 transactions" in explanation["plain_summary"]
    assert "8 minutes" in explanation["plain_summary"]
    assert "what_this_means" in explanation
    assert "key_facts" in explanation
    assert len(explanation["key_facts"]) == 2
    assert "risk_contribution" in explanation


def test_explain_impossible_travel():
    finding = {
        "pattern": "IMPOSSIBLE_TRAVEL",
        "account_id": "ACC-202",
        "from_country": "US",
        "to_country": "JP",
        "elapsed_hours": 1.5,
        "distance_km": 10500.0,
        "required_speed_kmh": 7000.0,
    }
    explanation = explain_finding(finding)

    assert "Physically impossible" in explanation["headline"]
    assert "US" in explanation["plain_summary"]
    assert "JP" in explanation["plain_summary"]
    assert "1.5 hours" in explanation["plain_summary"]
    assert any("10,500 km" in fact for fact in explanation["key_facts"])
    assert any("7,000 km/h" in fact for fact in explanation["key_facts"])


def test_explain_account_takeover():
    finding = {
        "pattern": "ACCOUNT_TAKEOVER",
        "account_id": "ACC-303",
        "signals": ["new_device", "new_ip", "high_value"],
        "amount": 7500.0,
        "device_id": "DEV-UNKNOWN",
    }
    explanation = explain_finding(finding)

    assert "account takeover" in explanation["headline"].lower()
    assert "ACC-303" in explanation["plain_summary"]
    assert "$7,500.00" in explanation["plain_summary"]
    assert "DEV-UNKNOWN" in explanation["plain_summary"]
    assert "3 takeover signals" in explanation["key_facts"][0]


def test_explain_new_device_high_value():
    finding = {
        "pattern": "NEW_DEVICE_HIGH_VALUE",
        "account_id": "ACC-404",
        "device_id": "DEV-999",
        "amount": 100000.0,
        "threshold": 50000.0,
    }
    explanation = explain_finding(finding)

    assert "Large transaction from a new" in explanation["headline"]
    assert "$100,000.00" in explanation["plain_summary"]
    assert "DEV-999" in explanation["plain_summary"]


def test_explain_unknown_pattern_fallback():
    finding = {
        "pattern": "CUSTOM_EXPERIMENTAL_PATTERN",
        "evidence": "Suspicious behavior observed.",
    }
    explanation = explain_finding(finding)

    assert "Custom Experimental Pattern" in explanation["headline"]
    assert "Suspicious behavior observed." in explanation["plain_summary"]


def test_explain_findings_batch():
    findings = [
        {
            "pattern": "VELOCITY_ATTACK",
            "account_id": "A1",
            "transaction_count": 5,
            "window_minutes": 10,
        },
        {
            "pattern": "NEW_DEVICE_HIGH_VALUE",
            "account_id": "A2",
            "device_id": "D1",
            "amount": 60000.0,
            "threshold": 50000.0,
        },
    ]
    annotated = explain_findings(findings)

    assert len(annotated) == 2
    assert "explanation" in annotated[0]
    assert "explanation" in annotated[1]


def test_build_executive_summary_clean():
    evidence_assessment = {
        "risk_score": 0,
        "risk_level": "LOW",
        "patterns": [],
        "decision": "NO_SUSPICIOUS_ACTIVITY",
    }
    summary = build_executive_summary(evidence_assessment, [])
    assert "No suspicious activity was detected" in summary


def test_build_executive_summary_critical():
    evidence_assessment = {
        "risk_score": 85,
        "risk_level": "CRITICAL",
        "patterns": ["VELOCITY_ATTACK", "ACCOUNT_TAKEOVER"],
        "decision": "REQUIRES_FURTHER_INVESTIGATION",
    }
    findings = [
        {
            "pattern": "VELOCITY_ATTACK",
            "explanation": {"headline": "Rapid transaction burst detected"},
        },
        {
            "pattern": "ACCOUNT_TAKEOVER",
            "explanation": {"headline": "Possible account takeover attempt"},
        },
    ]
    summary = build_executive_summary(evidence_assessment, findings)

    assert "85/100" in summary
    assert "CRITICAL" in summary
    assert "requires immediate escalation" in summary
    assert "Velocity Attack" in summary
    assert "Account Takeover" in summary


def test_velocity_explanation_uses_actual_base_weight_and_caveat():
    explanation = explain_finding({
        "pattern": "VELOCITY_ATTACK",
        "account_id": "ACC-TEST",
        "transaction_count": 3,
        "window_minutes": 5,
        "transaction_ids": ["T1", "T2", "T3"],
    })
    assert "+25 base risk points" in explanation["risk_contribution"]
    assert "before correlation adjustments" in explanation["risk_contribution"]
    assert "confirmed fraud finding" in explanation["risk_contribution"]


def test_new_device_explanation_does_not_claim_validated_fraud():
    explanation = explain_finding({
        "pattern": "NEW_DEVICE_HIGH_VALUE",
        "account_id": "ACC-TEST",
        "device_id": "DEV-TEST",
        "amount": 60000.0,
        "threshold": 50000.0,
    })
    assert "validated fraud signal" not in explanation["risk_contribution"]
    assert "legitimate explanations" in explanation["risk_contribution"]
