from typing import Any


class EvidenceEngine:

    PATTERN_WEIGHTS = {
        "VELOCITY_ATTACK": 25,
        "NEW_DEVICE_HIGH_VALUE": 25,
        "IMPOSSIBLE_TRAVEL": 30,
        "ACCOUNT_TAKEOVER": 35,
    }

    def __init__(
        self,
        ml_strong_threshold: float = 0.80,
        ml_weak_threshold: float = 0.50,
    ) -> None:
        if not 0.0 <= ml_weak_threshold <= ml_strong_threshold <= 1.0:
            raise ValueError(
                "ML thresholds must satisfy "
                "0.0 <= ml_weak_threshold <= ml_strong_threshold <= 1.0"
            )

        self.ml_strong_threshold = ml_strong_threshold
        self.ml_weak_threshold = ml_weak_threshold

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


        ml_findings = [
            finding
            for finding in findings
            if finding.get("evidence_type") == "ml_model_prediction"
        ]

        approved_ml_findings = [
            finding
            for finding in ml_findings
            if finding.get("model_approved") is True
            and finding.get("validation_status") == "PASS"
        ]

        recognized_findings = [
            finding
            for finding in findings
            if finding.get("pattern") in self.PATTERN_WEIGHTS
        ]

        deterministic_transaction_ids = {
            finding.get("transaction_id")
            for finding in recognized_findings
            if finding.get("transaction_id")
        }

        corroborating_ml_findings = [
            finding
            for finding in approved_ml_findings
            if finding.get("predicted_fraud") is True
            and finding.get("transaction_id") in deterministic_transaction_ids
        ]

        ml_corroboration = sorted(
            {
                finding.get("transaction_id")
                for finding in corroborating_ml_findings
            }
        )

        qualifying_ml_findings = [
            finding
            for finding in corroborating_ml_findings
            if float(finding.get("fraud_probability", 0.0))
            >= self.ml_weak_threshold
        ]

        ml_corroboration = sorted(
            {
                finding.get("transaction_id")
                for finding in qualifying_ml_findings
            }
        )

        if not qualifying_ml_findings:
            ml_corroboration_quality = "NONE"
        elif any(
            float(finding.get("fraud_probability", 0.0))
            >= self.ml_strong_threshold
            for finding in qualifying_ml_findings
        ):
            ml_corroboration_quality = "STRONG"
        else:
            ml_corroboration_quality = "WEAK"

        # Identify findings that refer to the same transaction.
        # Multiple patterns on one transaction are not automatically
        # independent evidence streams.
        transaction_pattern_map: dict[str, set[str]] = {}

        for finding in recognized_findings:
            transaction_id = finding.get("transaction_id")

            if transaction_id:
                transaction_pattern_map.setdefault(
                    transaction_id,
                    set(),
                ).add(finding["pattern"])

        overlapping_transactions = {
            transaction_id: sorted(patterns)
            for transaction_id, patterns in transaction_pattern_map.items()
            if len(patterns) > 1
        }

        # Correlation-aware scoring:
        # independent patterns receive full weight.
        # For patterns tied to the same transaction, the highest-weight
        # pattern receives full weight and additional patterns receive
        # 50% of their normal weight as corroborating evidence.
        score = sum(
            self.PATTERN_WEIGHTS[pattern]
            for pattern in unique_patterns
        )

        for patterns in overlapping_transactions.values():
            overlap_weights = sorted(
                (
                    self.PATTERN_WEIGHTS[pattern]
                    for pattern in patterns
                ),
                reverse=True,
            )

            # The highest-weight signal remains fully counted.
            # Reduce each additional correlated signal by 50%.
            for weight in overlap_weights[1:]:
                score -= weight * 0.5

        score = min(int(score), 100)

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
            "ml_evidence_count": len(ml_findings),
            "approved_ml_evidence_count": len(approved_ml_findings),
            "ml_corroboration": ml_corroboration,
            "ml_corroboration_quality": ml_corroboration_quality,
            "overlapping_transactions": overlapping_transactions,
            "evidence_quality": evidence_quality,
            "evidence_summary": evidence_summary,
            "decision": (
                "NO_SUSPICIOUS_ACTIVITY"
                if score == 0
                else "REQUIRES_FURTHER_INVESTIGATION"
            ),
        }
