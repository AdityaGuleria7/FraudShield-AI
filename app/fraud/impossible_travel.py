from datetime import datetime
from math import atan2, cos, radians, sin, sqrt
from typing import Any


class ImpossibleTravelDetector:

    # Approximate country coordinates.
    # These are used only for deterministic test/demo logic.
    COUNTRY_COORDINATES = {
        "IN": (20.5937, 78.9629),
        "US": (37.0902, -95.7129),
        "GB": (55.3781, -3.4360),
        "SG": (1.3521, 103.8198),
    }

    def __init__(self, max_speed_kmh: float = 900.0):
        self.max_speed_kmh = max_speed_kmh

    def _distance_km(
        self,
        first_country: str,
        second_country: str,
    ) -> float:

        first = self.COUNTRY_COORDINATES[first_country]
        second = self.COUNTRY_COORDINATES[second_country]

        lat1, lon1 = radians(first[0]), radians(first[1])
        lat2, lon2 = radians(second[0]), radians(second[1])

        earth_radius_km = 6371.0

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            sin(dlat / 2) ** 2
            + cos(lat1)
            * cos(lat2)
            * sin(dlon / 2) ** 2
        )

        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return earth_radius_km * c

    def detect(
        self,
        transactions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        grouped: dict[str, list[dict[str, Any]]] = {}

        for transaction in transactions:

            account_id = transaction["account_id"]

            if transaction["country"] not in self.COUNTRY_COORDINATES:
                continue

            grouped.setdefault(
                account_id,
                [],
            ).append(transaction)

        findings = []

        for account_id, account_transactions in grouped.items():

            ordered = sorted(
                account_transactions,
                key=lambda transaction: datetime.fromisoformat(
                    transaction["timestamp"]
                ),
            )

            for first, second in zip(
                ordered,
                ordered[1:],
            ):

                if first["country"] == second["country"]:
                    continue

                first_time = datetime.fromisoformat(
                    first["timestamp"]
                )

                second_time = datetime.fromisoformat(
                    second["timestamp"]
                )

                elapsed_hours = (
                    second_time - first_time
                ).total_seconds() / 3600

                if elapsed_hours <= 0:
                    continue

                distance_km = self._distance_km(
                    first["country"],
                    second["country"],
                )

                required_speed = (
                    distance_km / elapsed_hours
                )

                if required_speed > self.max_speed_kmh:

                    findings.append(
                        {
                            "pattern": "IMPOSSIBLE_TRAVEL",
                            "provenance": {
                                "detector": "impossible_travel",
                                "method": "deterministic_rule",
                                "version": "1",
                            },
                            "account_id": account_id,
                            "from_country": first["country"],
                            "to_country": second["country"],
                            "from_transaction_id": (
                                first["transaction_id"]
                            ),
                            "to_transaction_id": (
                                second["transaction_id"]
                            ),
                            "distance_km": round(
                                distance_km,
                                2,
                            ),
                            "elapsed_hours": round(
                                elapsed_hours,
                                4,
                            ),
                            "required_speed_kmh": round(
                                required_speed,
                                2,
                            ),
                            "max_allowed_speed_kmh": (
                                self.max_speed_kmh
                            ),
                            "evidence": (
                                f"Account {account_id} moved from "
                                f"{first['country']} to "
                                f"{second['country']} requiring "
                                f"{required_speed:.2f} km/h, "
                                f"above the configured maximum of "
                                f"{self.max_speed_kmh:.2f} km/h."
                            ),
                        }
                    )

        return findings
