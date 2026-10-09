from datetime import datetime
from typing import Any

from app.fraud.impossible_travel import ImpossibleTravelDetector


class FraudPatternEngine:

    def __init__(self):
        self.impossible_travel_detector = (
            ImpossibleTravelDetector()
        )

    def detect_velocity(
        self,
        transactions: list[dict[str, Any]],
        window_minutes: int = 10,
        minimum_transactions: int = 5,
    ) -> list[dict[str, Any]]:

        grouped: dict[str, list[dict[str, Any]]] = {}

        for transaction in transactions:
            account_id = transaction["account_id"]
            grouped.setdefault(account_id, []).append(transaction)

        findings = []

        for account_id, account_transactions in grouped.items():

            ordered = sorted(
                account_transactions,
                key=lambda transaction: datetime.fromisoformat(
                    transaction["timestamp"]
                ),
            )

            for index, start_transaction in enumerate(ordered):

                start_time = datetime.fromisoformat(
                    start_transaction["timestamp"]
                )

                window = [
                    transaction
                    for transaction in ordered[index:]
                    if (
                        datetime.fromisoformat(
                            transaction["timestamp"]
                        )
                        - start_time
                    ).total_seconds()
                    <= window_minutes * 60
                ]

                if len(window) >= minimum_transactions:

                    findings.append(
                        {
                            "pattern": "VELOCITY_ATTACK",
                            "account_id": account_id,
                            "transaction_count": len(window),
                            "window_minutes": window_minutes,
                            "transaction_ids": [
                                transaction["transaction_id"]
                                for transaction in window
                            ],
                            "evidence": (
                                f"{len(window)} transactions detected "
                                f"within {window_minutes} minutes."
                            ),
                        }
                    )

                    break

        return findings

    def detect_new_device_high_value(
        self,
        transactions: list[dict[str, Any]],
        high_value_threshold: float = 50000.0,
    ) -> list[dict[str, Any]]:

        known_devices: dict[str, set[str]] = {}
        findings = []

        ordered = sorted(
            transactions,
            key=lambda transaction: datetime.fromisoformat(
                transaction["timestamp"]
            ),
        )

        for transaction in ordered:

            account_id = transaction["account_id"]
            device_id = transaction["device_id"]

            devices = known_devices.setdefault(
                account_id,
                set(),
            )

            is_new_device = device_id not in devices

            if (
                is_new_device
                and transaction["amount"] >= high_value_threshold
            ):
                findings.append(
                    {
                        "pattern": "NEW_DEVICE_HIGH_VALUE",
                        "account_id": account_id,
                        "transaction_id": transaction["transaction_id"],
                        "device_id": device_id,
                        "amount": transaction["amount"],
                        "threshold": high_value_threshold,
                        "evidence": (
                            f"New device {device_id} used for "
                            f"high-value transaction of "
                            f"{transaction['amount']:.2f}."
                        ),
                    }
                )

            devices.add(device_id)

        return findings

    def detect_impossible_travel(
        self,
        transactions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        return self.impossible_travel_detector.detect(
            transactions
        )
