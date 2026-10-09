from typing import Any


class RelationshipFeatureBuilder:
    def build(
        self, transactions: list[dict[str, Any]]
    ) -> dict[str, dict[str, int]]:
        ordered_transactions = sorted(
            transactions,
            key=lambda transaction: transaction["transaction_id"],
        )

        account_devices: dict[str, set[str]] = {}
        account_ips: dict[str, set[str]] = {}
        device_accounts: dict[str, set[str]] = {}

        features: dict[str, dict[str, int]] = {}

        for transaction in ordered_transactions:
            transaction_id = transaction["transaction_id"]
            account_id = transaction["account_id"]
            device_id = transaction["device_id"]
            ip_address = transaction["ip_address"]

            account_devices.setdefault(account_id, set())
            account_ips.setdefault(account_id, set())
            device_accounts.setdefault(device_id, set())

            features[transaction_id] = {
                "account_unique_device_count": len(
                    account_devices[account_id] | {device_id}
                ),
                "account_unique_ip_count": len(
                    account_ips[account_id] | {ip_address}
                ),
                "device_unique_account_count": len(
                    device_accounts[device_id] | {account_id}
                ),
                "is_new_device_for_account": int(
                    device_id not in account_devices[account_id]
                ),
                "is_new_ip_for_account": int(
                    ip_address not in account_ips[account_id]
                ),
            }

            account_devices[account_id].add(device_id)
            account_ips[account_id].add(ip_address)
            device_accounts[device_id].add(account_id)

        return features
