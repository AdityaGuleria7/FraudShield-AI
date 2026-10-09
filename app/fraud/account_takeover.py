from typing import Any


class AccountTakeoverDetector:

    def __init__(
        self,
        high_value_threshold: float = 50000.0,
    ):
        self.high_value_threshold = high_value_threshold

    def detect(
        self,
        transactions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        known_devices: dict[str, set[str]] = {}
        known_ips: dict[str, set[str]] = {}

        findings = []

        for transaction in transactions:

            account_id = transaction["account_id"]
            device_id = transaction["device_id"]
            ip_address = transaction["ip_address"]
            amount = transaction["amount"]

            devices = known_devices.setdefault(
                account_id,
                set(),
            )

            ips = known_ips.setdefault(
                account_id,
                set(),
            )

            is_new_device = device_id not in devices
            is_new_ip = ip_address not in ips
            is_high_value = amount >= self.high_value_threshold

            if (
                is_new_device
                and is_new_ip
                and is_high_value
            ):
                findings.append(
                    {
                        "pattern": "ACCOUNT_TAKEOVER",
                        "account_id": account_id,
                        "transaction_id": transaction[
                            "transaction_id"
                        ],
                        "device_id": device_id,
                        "ip_address": ip_address,
                        "amount": amount,
                        "threshold": self.high_value_threshold,
                        "signals": {
                            "new_device": True,
                            "new_ip": True,
                            "high_value": True,
                        },
                        "evidence": (
                            f"Account {account_id} used a new device "
                            f"and new IP address for a high-value "
                            f"transaction of {amount:.2f}."
                        ),
                    }
                )

            devices.add(device_id)
            ips.add(ip_address)

        return findings
