from typing import Any


class EvidenceEngine:

    PATTERN_WEIGHTS = {
        "VELOCITY_ATTACK": 25,
        "NEW_DEVICE_HIGH_VALUE": 25,
        "IMPOSSIBLE_TRAVEL": 30,
        "ACCOUNT_TAKEOVER": 35,
    }

    def assess(
        self,
        findings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Correlate deterministic fraud findings into
        an explainable risk assessment.

        This engine does not make a final fraud decision.
        """

        unique_patterns = sorted(
            {
                finding["pattern"]
                for finding in findings
                if finding.get("pattern") in self.PATTERN_WEIGHTS
            }
        )

        score = sum(
            self.PATTERN_WEIGHTS[pattern]
            for pattern in unique_patterns
        )

        score = min(score, 100)

        recognized_findings = [
            finding
            for finding in findings
            if finding.get("pattern") in self.PATTERN_WEIGHTS
        ]

        if not recognized_findings:
            evidence_quality = "NONE"
            evidence_summary = "No recognized fraud evidence was supplied."

        else:
            strong_findings = 0

            for finding in recognized_findings:
                pattern = finding["pattern"]

                if pattern == "ACCOUNT_TAKEOVER":
                    required_fields = {
                        "account_id",
                        "transaction_id",
                        "device_id",
                        "ip_address",
                        "amount",
                        "signals",
                    }

                elif pattern == "IMPOSSIBLE_TRAVEL":
                    required_fields = {
                        "account_id",
                        "from_country",
                        "to_country",
                        "from_transaction_id",
                        "to_transaction_id",
                        "distance_km",
                        "elapsed_hours",
                        "required_speed_kmh",
                    }

                elif pattern == "VELOCITY_ATTACK":
                    required_fields = {
                        "account_id",
                        "transaction_count",
                        "window_minutes",
                        "transaction_ids",
                    }

                elif pattern == "NEW_DEVICE_HIGH_VALUE":
                    required_fields = {
                        "account_id",
                        "transaction_id",
                        "device_id",
                        "amount",
                        "threshold",
                    }

                else:
                    required_fields = set()

                if required_fields.issubset(finding):
                    strong_findings += 1

            if strong_findings == len(recognized_findings):
                evidence_quality = "STRONG"
                evidence_summary = (
                    "All recognized findings contain the "
                    "required supporting evidence."
                )
            else:
                evidence_quality = "PARTIAL"
                evidence_summary = (
                    "Some recognized findings are missing "
                    "expected supporting evidence."
                )

        if score == 0:
            risk_level = "LOW"
        elif score < 50:
            risk_level = "MEDIUM"
        elif score < 75:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        return {
            "risk_score": score,
            "risk_level": risk_level,
            "patterns": unique_patterns,
            "evidence_count": len(findings),
            "evidence_quality": evidence_quality,
            "evidence_summary": evidence_summary,
            "decision": (
                "NO_SUSPICIOUS_ACTIVITY"
                if score == 0
                else "REQUIRES_FURTHER_INVESTIGATION"
            ),
        }
