from typing import Any


class FraudScenarioGenerator:

    def inject_velocity_attack(
        self,
        transactions: list[dict[str, Any]],
        account_id: str = "ACC9999",
    ) -> list[dict[str, Any]]:

        scenario = []

        for index in range(5):
            scenario.append(
                {
                    "transaction_id": f"FRAUD-VEL-{index + 1:04d}",
                    "account_id": account_id,
                    "device_id": "DEV-FRAUD-01",
                    "ip_address": "10.0.0.99",
                    "country": "IN",
                    "merchant_category": "ONLINE_SERVICES",
                    "amount": 25000.00 + index * 1000,
                    "timestamp": (
                        f"2026-01-02T10:0{index}:00+00:00"
                    ),
                    "is_fraud": True,
                    "fraud_scenario": "VELOCITY_ATTACK",
                }
            )

        return transactions + scenario

    def inject_new_device_high_value(
        self,
        transactions: list[dict[str, Any]],
        account_id: str = "ACC8888",
    ) -> list[dict[str, Any]]:

        transaction = {
            "transaction_id": "FRAUD-DEVICE-0001",
            "account_id": account_id,
            "device_id": "DEV-NEW-999",
            "ip_address": "10.0.0.88",
            "country": "IN",
            "merchant_category": "ELECTRONICS",
            "amount": 95000.00,
            "timestamp": "2026-01-03T14:30:00+00:00",
            "is_fraud": True,
            "fraud_scenario": "NEW_DEVICE_HIGH_VALUE",
        }

        return transactions + [transaction]
