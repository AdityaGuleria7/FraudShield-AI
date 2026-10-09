from datetime import datetime, timedelta, timezone
import random
from typing import Any


class TransactionGenerator:

    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)

    def generate(self, count: int = 100) -> list[dict[str, Any]]:

        transactions = []

        base_time = datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        )

        for index in range(count):

            transaction = {
                "transaction_id": f"TX{index + 1:06d}",
                "account_id": f"ACC{self.random.randint(1, 20):04d}",
                "device_id": f"DEV{self.random.randint(1, 30):04d}",
                "ip_address": (
                    f"192.168.{self.random.randint(1, 10)}."
                    f"{self.random.randint(1, 254)}"
                ),
                "country": self.random.choice(
                    ["IN", "US", "GB", "SG"]
                ),
                "merchant_category": self.random.choice(
                    [
                        "GROCERY",
                        "ELECTRONICS",
                        "TRAVEL",
                        "FOOD",
                        "ONLINE_SERVICES",
                    ]
                ),
                "amount": round(
                    self.random.uniform(100, 50000),
                    2,
                ),
                "timestamp": (
                    base_time
                    + timedelta(
                        minutes=self.random.randint(0, 100000)
                    )
                ).isoformat(),
                "is_fraud": False,
            }

            transactions.append(transaction)

        return transactions
