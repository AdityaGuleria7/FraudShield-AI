from datetime import datetime
from typing import Any


class DataQualityValidator:
    REQUIRED_FIELDS = {
        "transaction_id",
        "account_id",
        "device_id",
        "ip_address",
        "country",
        "merchant_category",
        "amount",
        "timestamp",
        "is_fraud",
    }

    VALID_COUNTRIES = {"IN", "US", "GB", "SG"}
    OPTIONAL_FIELDS = {"fraud_scenario"}

    def validate(self, transactions: list[dict[str, Any]]) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []
        valid_transactions = 0
        missing_field_count = 0
        seen_transaction_ids: set[str] = set()

        for index, transaction in enumerate(transactions):
            transaction_valid = True

            # Required-field validation
            missing_fields = self.REQUIRED_FIELDS - transaction.keys()
            allowed_fields = self.REQUIRED_FIELDS | self.OPTIONAL_FIELDS
            unexpected_fields = transaction.keys() - allowed_fields

            missing_field_count += len(missing_fields)

            for field in sorted(missing_fields):
                issues.append(
                    {
                        "type": "MISSING_FIELD",
                        "transaction_index": index,
                        "field": field,
                    }
                )
                transaction_valid = False

            # Schema consistency validation
            for field in sorted(unexpected_fields):
                issues.append(
                    {
                        "type": "UNEXPECTED_FIELD",
                        "transaction_index": index,
                        "field": field,
                    }
                )
                transaction_valid = False

            # Stop field-level validation if required fields are missing.
            if missing_fields:
                continue

            transaction_id = transaction["transaction_id"]

            # Duplicate transaction ID validation
            if transaction_id in seen_transaction_ids:
                issues.append(
                    {
                        "type": "DUPLICATE_TRANSACTION_ID",
                        "transaction_index": index,
                        "transaction_id": transaction_id,
                    }
                )
                transaction_valid = False
            else:
                seen_transaction_ids.add(transaction_id)

            # Amount validation
            if not isinstance(transaction["amount"], (int, float)):
                issues.append(
                    {
                        "type": "INVALID_AMOUNT",
                        "transaction_index": index,
                        "value": transaction["amount"],
                    }
                )
                transaction_valid = False
            elif transaction["amount"] < 0:
                issues.append(
                    {
                        "type": "INVALID_AMOUNT",
                        "transaction_index": index,
                        "value": transaction["amount"],
                    }
                )
                transaction_valid = False

            # Country validation
            if transaction["country"] not in self.VALID_COUNTRIES:
                issues.append(
                    {
                        "type": "INVALID_COUNTRY",
                        "transaction_index": index,
                        "value": transaction["country"],
                    }
                )
                transaction_valid = False

            # Timestamp validation
            try:
                datetime.fromisoformat(transaction["timestamp"])
            except (TypeError, ValueError):
                issues.append(
                    {
                        "type": "INVALID_TIMESTAMP",
                        "transaction_index": index,
                        "value": transaction["timestamp"],
                    }
                )
                transaction_valid = False

            if transaction_valid:
                valid_transactions += 1

        total_transactions = len(transactions)

        if total_transactions == 0:
            quality_score = 0
        else:
            quality_score = int(
                (valid_transactions / total_transactions) * 100
            )

        status = "PASS" if not issues else "REJECT"

        return {
            "status": status,
            "quality_score": quality_score,
            "total_transactions": total_transactions,
            "valid_transactions": valid_transactions,
            "missing_field_count": missing_field_count,
            "missing_field_rate": (
                missing_field_count / total_transactions
                if total_transactions
                else 0.0
            ),
            "issues": issues,
        }
