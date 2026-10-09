from datetime import datetime
from typing import Any


class BaselineFeatureBuilder:
    def build(
        self, transactions: list[dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        account_history: dict[str, list[dict[str, Any]]] = {}
        features: dict[str, dict[str, float]] = {}

        ordered_transactions = sorted(
            transactions,
            key=lambda transaction: datetime.fromisoformat(
                transaction["timestamp"]
            ),
        )

        for transaction in ordered_transactions:
            transaction_id = transaction["transaction_id"]
            account_id = transaction["account_id"]
            amount = float(transaction["amount"])

            history = account_history.setdefault(account_id, [])

            if not history:
                features[transaction_id] = {
                    "amount_vs_account_average": 0.0,
                    "amount_vs_account_max": 0.0,
                    "is_above_account_average": 0,
                    "is_above_account_max": 0,
                }
            else:
                amounts = [item["amount"] for item in history]
                account_average = sum(amounts) / len(amounts)
                account_max = max(amounts)

                features[transaction_id] = {
                    "amount_vs_account_average": round(
                        amount / account_average, 6
                    ),
                    "amount_vs_account_max": round(
                        amount / account_max, 6
                    ),
                    "is_above_account_average": int(
                        amount > account_average
                    ),
                    "is_above_account_max": int(
                        amount > account_max
                    ),
                }

            history.append(
                {
                    "amount": amount,
                    "timestamp": transaction["timestamp"],
                }
            )

        return features
