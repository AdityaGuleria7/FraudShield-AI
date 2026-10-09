from datetime import datetime
from typing import Any


class TemporalFeatureBuilder:
    def build(
        self, transactions: list[dict[str, Any]]
    ) -> dict[str, dict[str, int]]:
        features: dict[str, dict[str, int]] = {}

        for transaction in transactions:
            transaction_id = transaction["transaction_id"]
            timestamp = datetime.fromisoformat(transaction["timestamp"])

            hour = timestamp.hour
            day_of_week = timestamp.weekday()

            features[transaction_id] = {
                "hour_of_day": hour,
                "day_of_week": day_of_week,
                "is_weekend": int(day_of_week >= 5),
                "is_night": int(hour < 6 or hour >= 22),
            }

        return features
