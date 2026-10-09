from typing import Any


class BehavioralFeatureBuilder:
    def build(
        self, transactions: list[dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        account_transactions: dict[str, list[dict[str, Any]]] = {}

        for transaction in transactions:
            account_id = transaction["account_id"]
            account_transactions.setdefault(account_id, []).append(transaction)

        features: dict[str, dict[str, float]] = {}

        for transaction in transactions:
            transaction_id = transaction["transaction_id"]
            account_id = transaction["account_id"]
            account_data = account_transactions[account_id]

            amounts = [item["amount"] for item in account_data]

            features[transaction_id] = {
                "transaction_amount": float(transaction["amount"]),
                "account_transaction_count": float(len(account_data)),
                "account_total_amount": float(sum(amounts)),
                "account_average_amount": float(sum(amounts) / len(amounts)),
                "account_max_amount": float(max(amounts)),
            }

        return features
