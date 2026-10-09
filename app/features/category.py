from datetime import datetime
from typing import Any


class CategoryFeatureBuilder:
    def build(
        self, transactions: list[dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        ordered_transactions = sorted(
            transactions,
            key=lambda transaction: datetime.fromisoformat(
                transaction["timestamp"]
            ),
        )

        account_category_counts: dict[str, dict[str, int]] = {}
        account_category_amounts: dict[str, dict[str, float]] = {}
        account_total_amounts: dict[str, float] = {}

        features: dict[str, dict[str, float]] = {}

        for transaction in ordered_transactions:
            transaction_id = transaction["transaction_id"]
            account_id = transaction["account_id"]
            category = transaction["merchant_category"]
            amount = float(transaction["amount"])

            category_counts = account_category_counts.setdefault(
                account_id, {}
            )
            category_amounts = account_category_amounts.setdefault(
                account_id, {}
            )
            total_amount = account_total_amounts.get(account_id, 0.0)

            prior_category_count = category_counts.get(category, 0)
            prior_category_amount = category_amounts.get(category, 0.0)

            if total_amount == 0:
                category_share = 0.0
            else:
                category_share = prior_category_amount / total_amount

            features[transaction_id] = {
                "category_transaction_count": prior_category_count,
                "category_total_amount": prior_category_amount,
                "is_new_category_for_account": int(
                    prior_category_count == 0
                ),
                "category_share_of_account_spend": round(
                    category_share, 6
                ),
            }

            category_counts[category] = prior_category_count + 1
            category_amounts[category] = (
                prior_category_amount + amount
            )
            account_total_amounts[account_id] = total_amount + amount

        return features
