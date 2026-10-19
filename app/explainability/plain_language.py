"""
Plain-language explanation engine for FraudShield AI.

Translates deterministic fraud pattern findings into human-readable
summaries that any analyst can understand — no data science background needed.
"""
from __future__ import annotations

from typing import Any


# ── Pattern → plain English templates ────────────────────────────────────────
# Each template receives a `finding` dict and returns a plain-language string.

def _explain_velocity_attack(finding: dict[str, Any]) -> dict[str, Any]:
    count = finding.get("transaction_count", "multiple")
    window = finding.get("window_minutes", 10)
    account = finding.get("account_id", "this account")
    ids = finding.get("transaction_ids", [])
    txn_word = "transaction" if count == 1 else "transactions"
    return {
        "headline": "Rapid transaction burst detected",
        "plain_summary": (
            f"Account {account} made {count} {txn_word} within "
            f"{window} minutes — a pattern consistent with card-testing "
            f"fraud, where stolen cards are tested with many small purchases."
        ),
        "what_this_means": (
            "Fraudsters often rapidly fire many transactions to test whether "
            "a stolen card is still active before making large purchases."
        ),
        "key_facts": [
            f"{count} transactions in {window} minutes",
            f"Affected transactions: {', '.join(str(i) for i in ids[:5])}{'…' if len(ids) > 5 else ''}",
        ],
        "risk_contribution": "+25 base risk points before correlation adjustments; a rule-based indicator, not a confirmed fraud finding",
    }


def _explain_impossible_travel(finding: dict[str, Any]) -> dict[str, Any]:
    from_country = finding.get("from_country", "location A")
    to_country = finding.get("to_country", "location B")
    elapsed_h = finding.get("elapsed_hours", "?")
    distance_km = finding.get("distance_km")
    speed_kmh = finding.get("required_speed_kmh")
    account = finding.get("account_id", "this account")

    distance_str = f"{distance_km:,.0f} km apart" if distance_km else "far apart"
    speed_str = (
        f"reaching {speed_kmh:,.0f} km/h — faster than any commercial aircraft"
        if speed_kmh and speed_kmh > 900
        else f"requiring {speed_kmh:,.0f} km/h travel speed"
        if speed_kmh
        else "physically impossible in that timeframe"
    )

    elapsed_str = (
        f"{elapsed_h:.1f} hours" if isinstance(elapsed_h, (int, float)) else str(elapsed_h)
    )

    return {
        "headline": "Physically impossible location jump",
        "plain_summary": (
            f"Account {account} had a transaction in {from_country} and then "
            f"another in {to_country} just {elapsed_str} later. "
            f"The two locations are {distance_str}, {speed_str}."
        ),
        "what_this_means": (
            "This either means card details were stolen and used in a "
            "different country, or two people are using the same account "
            "simultaneously — both are high-risk signals."
        ),
        "key_facts": [
            f"From: {from_country} → To: {to_country}",
            f"Time gap: {elapsed_str}",
            *(
                [f"Distance: {distance_km:,.0f} km", f"Required speed: {speed_kmh:,.0f} km/h"]
                if distance_km and speed_kmh
                else []
            ),
        ],
        "risk_contribution": "+30 base risk points before correlation adjustments; a rule-based location-anomaly indicator",
    }


def _explain_account_takeover(finding: dict[str, Any]) -> dict[str, Any]:
    account = finding.get("account_id", "this account")
    signals = finding.get("signals", [])
    amount = finding.get("amount")
    device = finding.get("device_id", "unknown device")

    signal_phrases = {
        "new_device": "unrecognized device",
        "new_ip": "new IP address",
        "high_value": "unusually large transaction amount",
        "unusual_time": "transaction at an unusual hour",
        "country_change": "different country than usual",
        "multiple_failures": "recent failed login attempts",
    }
    readable_signals = [
        signal_phrases.get(s, s.replace("_", " ")) for s in signals
    ]

    amount_str = f"${amount:,.2f} " if amount else ""
    signals_str = (
        ", ".join(readable_signals[:-1]) + f" and {readable_signals[-1]}"
        if len(readable_signals) > 1
        else readable_signals[0] if readable_signals
        else "multiple suspicious indicators"
    )

    return {
        "headline": "Possible account takeover attempt",
        "plain_summary": (
            f"Account {account} shows {len(signals)} account takeover signals: "
            f"{signals_str}. "
            f"A {amount_str}transaction was attempted from device {device}."
        ),
        "what_this_means": (
            "Account takeover fraud occurs when an attacker gains access to a "
            "legitimate account — often through phishing or credential stuffing — "
            "and immediately attempts high-value transactions before the owner notices."
        ),
        "key_facts": [
            f"{len(signals)} takeover signals detected",
            *[f"Signal: {s}" for s in readable_signals],
            *(
                [f"Transaction amount: ${amount:,.2f}"]
                if amount else []
            ),
        ],
        "risk_contribution": "+35 base risk points before correlation adjustments; a rule-based account-takeover indicator",
    }


def _explain_new_device_high_value(finding: dict[str, Any]) -> dict[str, Any]:
    account = finding.get("account_id", "this account")
    device = finding.get("device_id", "unknown device")
    amount = finding.get("amount", 0)
    threshold = finding.get("threshold", 50000)

    return {
        "headline": "Large transaction from a new, unrecognized device",
        "plain_summary": (
            f"Account {account} used a brand-new device ({device}) to authorize "
            f"a ${amount:,.2f} transaction — {amount/threshold:.1f}× the high-value "
            f"threshold of ${threshold:,.2f}."
        ),
        "what_this_means": (
            "Legitimate users occasionally make large purchases from new devices, "
            "but the combination of an unrecognized device and a high transaction amount "
            "is a well-known fraud pattern — especially in account takeover scenarios."
        ),
        "key_facts": [
            f"Device: {device} (first time seen on this account)",
            f"Amount: ${amount:,.2f}",
            f"High-value threshold: ${threshold:,.2f}",
        ],
        "risk_contribution": "+25 base risk points before correlation adjustments; a rule-based indicator that may have legitimate explanations",
    }


# ── Pattern dispatcher ────────────────────────────────────────────────────────

_EXPLAINERS = {
    "VELOCITY_ATTACK": _explain_velocity_attack,
    "IMPOSSIBLE_TRAVEL": _explain_impossible_travel,
    "ACCOUNT_TAKEOVER": _explain_account_takeover,
    "NEW_DEVICE_HIGH_VALUE": _explain_new_device_high_value,
}


def explain_finding(finding: dict[str, Any]) -> dict[str, Any]:
    """
    Generate a plain-language explanation for a single fraud finding.

    Returns an explanation dict with:
      - headline: one-line title
      - plain_summary: 2-3 sentence human-readable explanation
      - what_this_means: context about this fraud pattern
      - key_facts: bullet-point list of key numbers
      - risk_contribution: how many risk points this adds and why
    """
    pattern = finding.get("pattern", "UNKNOWN")
    explainer = _EXPLAINERS.get(pattern)

    if explainer is None:
        return {
            "headline": f"Suspicious pattern: {pattern.replace('_', ' ').title()}",
            "plain_summary": finding.get(
                "evidence",
                "A suspicious pattern was detected. Review the full finding for details.",
            ),
            "what_this_means": "This pattern has been flagged for manual review.",
            "key_facts": [],
            "risk_contribution": "Contributes to overall risk score.",
        }

    return explainer(finding)


def explain_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate plain-language explanations for a list of fraud findings."""
    return [
        {**finding, "explanation": explain_finding(finding)}
        for finding in findings
    ]


def build_executive_summary(
    evidence_assessment: dict[str, Any],
    findings_with_explanations: list[dict[str, Any]],
) -> str:
    """
    Build a one-paragraph executive summary of the fraud investigation.

    Written for a non-technical audience: a bank branch manager, a compliance
    officer, or a customer service team lead.
    """
    risk_score = evidence_assessment.get("risk_score", 0)
    risk_level = evidence_assessment.get("risk_level", "LOW")
    patterns = evidence_assessment.get("patterns", [])
    decision = evidence_assessment.get("decision", "NO_SUSPICIOUS_ACTIVITY")

    if risk_score == 0 or not patterns:
        return (
            "No suspicious activity was detected by the configured fraud-pattern "
            "checks in this batch. This does not establish that every transaction "
            "passed all data-quality checks or that fraud is impossible. "
            "Review data-quality results and pipeline coverage before closing "
            "the investigation."
        )

    pattern_count = len(patterns)
    pattern_names = [p.replace("_", " ").title() for p in patterns]

    if len(pattern_names) == 1:
        pattern_str = pattern_names[0]
    elif len(pattern_names) == 2:
        pattern_str = f"{pattern_names[0]} and {pattern_names[1]}"
    else:
        pattern_str = (
            ", ".join(pattern_names[:-1]) + f", and {pattern_names[-1]}"
        )

    severity_phrases = {
        "CRITICAL": "requires immediate escalation",
        "HIGH": "warrants prompt investigation",
        "MEDIUM": "should be reviewed by an analyst",
        "LOW": "may warrant a routine check",
    }
    action = severity_phrases.get(risk_level, "requires review")

    # Collect the most important headline from findings
    headlines = [
        f["explanation"]["headline"]
        for f in findings_with_explanations
        if "explanation" in f
    ]
    top_headline = headlines[0] if headlines else ""

    return (
        f"This investigation identified {pattern_count} fraud indicator(s) "
        f"with a combined risk score of {risk_score}/100 ({risk_level}). "
        f"The primary concern is: {top_headline.lower() if top_headline else pattern_str}. "
        f"{'Multiple corroborating patterns were detected: ' + pattern_str + '. ' if pattern_count > 1 else ''}"
        f"This {action}. "
        f"{'No automated action has been taken — ' if decision == 'REQUIRES_FURTHER_INVESTIGATION' else ''}"
        f"all findings are flagged for human review."
    )
