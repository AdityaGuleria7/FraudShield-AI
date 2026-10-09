import math
from typing import Any


class FeatureQualityValidator:
    REQUIRED_FEATURES = {
        "transaction_amount",
        "transactions_last_5m",
        "transactions_last_10m",
        "transactions_last_1h",
        "amount_last_5m",
        "amount_last_10m",
        "amount_last_1h",
        "account_unique_device_count",
        "account_unique_ip_count",
        "device_unique_account_count",
        "is_new_device_for_account",
        "is_new_ip_for_account",
        "amount_vs_account_average",
        "amount_vs_account_max",
        "is_above_account_average",
        "is_above_account_max",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "is_night",
        "category_transaction_count",
        "category_total_amount",
        "is_new_category_for_account",
        "category_share_of_account_spend",
    }

    def validate(
        self, feature_vectors: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []
        valid_transactions = 0

        for transaction_id, features in feature_vectors.items():
            transaction_valid = True

            missing_features = self.REQUIRED_FEATURES - features.keys()
            unexpected_features = features.keys() - self.REQUIRED_FEATURES

            for feature in sorted(missing_features):
                issues.append(
                    {
                        "type": "MISSING_FEATURE",
                        "transaction_id": transaction_id,
                        "feature": feature,
                    }
                )
                transaction_valid = False

            for feature in sorted(unexpected_features):
                issues.append(
                    {
                        "type": "UNEXPECTED_FEATURE",
                        "transaction_id": transaction_id,
                        "feature": feature,
                    }
                )
                transaction_valid = False

            for feature, value in features.items():
                if value is None:
                    issues.append(
                        {
                            "type": "NULL_FEATURE",
                            "transaction_id": transaction_id,
                            "feature": feature,
                        }
                    )
                    transaction_valid = False
                    continue

                if isinstance(value, (int, float)) and not math.isfinite(value):
                    issues.append(
                        {
                            "type": "NON_FINITE_FEATURE",
                            "transaction_id": transaction_id,
                            "feature": feature,
                        }
                    )
                    transaction_valid = False

            if transaction_valid:
                valid_transactions += 1

        status = "PASS" if not issues else "REJECT"

        return {
            "status": status,
            "total_transactions": len(feature_vectors),
            "valid_transactions": valid_transactions,
            "issues": issues,
        }
