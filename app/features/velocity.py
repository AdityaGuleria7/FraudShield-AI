from datetime import datetime, timedelta
from typing import Any


class VelocityFeatureBuilder:
    WINDOWS = {
        "5m": timedelta(minutes=5),
        "10m": timedelta(minutes=10),
        "1h": timedelta(hours=1),
    }

    def build(
        self, transactions: list[dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        account_transactions: dict[str, list[dict[str, Any]]] = {}

        for transaction in transactions:
            account_id = transaction["account_id"]
            account_transactions.setdefault(account_id, []).append(transaction)

        for account_id in account_transactions:
            account_transactions[account_id].sort(
                key=lambda transaction: datetime.fromisoformat(
                    transaction["timestamp"]
                )
            )

        features: dict[str, dict[str, float]] = {}

        for transaction in transactions:
            transaction_id = transaction["transaction_id"]
            account_id = transaction["account_id"]
            current_time = datetime.fromisoformat(transaction["timestamp"])

            account_history = account_transactions[account_id]

            transaction_features: dict[str, float] = {}

            for window_name, window in self.WINDOWS.items():
                window_start = current_time - window

                prior_transactions = [
                    previous
                    for previous in account_history
                    if window_start
                    <= datetime.fromisoformat(previous["timestamp"])
                    < current_time
                ]

                transaction_features[f"transactions_last_{window_name}"] = len(
                    prior_transactions
                )
                transaction_features[f"amount_last_{window_name}"] = float(
                    sum(previous["amount"] for previous in prior_transactions)
                )

            features[transaction_id] = transaction_features

        return features
