"""Explainable relationship-graph analysis for fraud investigations.

Graph connections are investigative signals, not proof of fraud.
A shared IP address or device alone is never sufficient to flag a cluster.
"""

from collections import defaultdict
from typing import Any

import networkx as nx


IDENTIFIER_FIELDS = (
    "device_id",
    "ip_address",
    "payment_id",
)


class FraudGraphDetector:
    """Build account-identifier graphs and return explainable cluster evidence."""

    def __init__(
        self,
        min_accounts: int = 3,
        min_shared_identifiers: int = 2,
    ) -> None:
        if min_accounts < 2:
            raise ValueError("min_accounts must be at least 2.")
        if min_shared_identifiers < 1:
            raise ValueError("min_shared_identifiers must be at least 1.")

        self.min_accounts = min_accounts
        self.min_shared_identifiers = min_shared_identifiers

    def analyze(
        self,
        transactions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Analyze shared identifiers and return clusters with supporting evidence.

        Expected transaction fields:
        transaction_id, account_id, and any of device_id, ip_address,
        payment_id. Missing or empty identifiers are ignored.
        """
        graph = nx.Graph()
        account_identifiers: dict[str, set[tuple[str, str]]] = defaultdict(set)
        identifier_accounts: dict[tuple[str, str], set[str]] = defaultdict(set)
        transaction_accounts: dict[str, str] = {}

        for transaction in transactions:
            transaction_id = transaction.get("transaction_id")
            account_value = transaction.get("account_id")

            if account_value is None or not str(account_value).strip():
                continue

            account_id = str(account_value).strip()
            graph.add_node(account_id, node_type="account")

            if transaction_id is not None:
                transaction_accounts[str(transaction_id)] = account_id

            for field in IDENTIFIER_FIELDS:
                value = transaction.get(field)
                if value is None or not str(value).strip():
                    continue

                identifier = (field, str(value).strip())
                account_identifiers[account_id].add(identifier)
                identifier_accounts[identifier].add(account_id)

        # Add account-account edges through shared identifiers.
        for identifier, accounts in identifier_accounts.items():
            ordered_accounts = sorted(accounts)
            for index, account_a in enumerate(ordered_accounts):
                for account_b in ordered_accounts[index + 1:]:
                    if not graph.has_edge(account_a, account_b):
                        graph.add_edge(
                            account_a,
                            account_b,
                            shared_identifiers=[],
                        )
                    graph.edges[account_a, account_b][
                        "shared_identifiers"
                    ].append({
                        "type": identifier[0],
                        "value": identifier[1],
                    })

        clusters = []
        for members in nx.connected_components(graph):
            accounts = sorted(members)
            if len(accounts) < 2:
                continue

            evidence = []
            shared_identifier_count = 0
            shared_identifier_types = set()

            for index, account_a in enumerate(accounts):
                for account_b in accounts[index + 1:]:
                    if not graph.has_edge(account_a, account_b):
                        continue

                    for item in graph.edges[account_a, account_b][
                        "shared_identifiers"
                    ]:
                        evidence.append({
                            "account_a": account_a,
                            "account_b": account_b,
                            "identifier_type": item["type"],
                            "identifier_value": item["value"],
                        })
                        shared_identifier_count += 1
                        shared_identifier_types.add(item["type"])

            # Require multiple identifier types and sufficient evidence.
            # A high-degree shared IP or device alone is not enough.
            suspicious = (
                len(accounts) >= self.min_accounts
                and len(shared_identifier_types) >= 2
                and shared_identifier_count >= self.min_shared_identifiers
            )

            clusters.append({
                "accounts": accounts,
                "account_count": len(accounts),
                "shared_identifier_types": sorted(shared_identifier_types),
                "shared_identifier_count": shared_identifier_count,
                "evidence": evidence,
                "status": (
                    "REVIEW_CANDIDATE" if suspicious
                    else "INSUFFICIENT_EVIDENCE"
                ),
                "explanation": (
                    "Multiple accounts are connected through more than one "
                    "identifier type. Review the evidence and context; this "
                    "does not establish fraud."
                    if suspicious
                    else
                    "Accounts share identifiers, but the configured evidence "
                    "rule was not met. Shared infrastructure can be legitimate."
                ),
            })

        clusters.sort(key=lambda cluster: cluster["accounts"])

        transaction_cluster = {}
        for transaction_id, account_id in transaction_accounts.items():
            for index, cluster in enumerate(clusters):
                if account_id in cluster["accounts"]:
                    transaction_cluster[transaction_id] = index
                    break

        return {
            "graph_type": "account_shared_identifier_graph",
            "account_count": graph.number_of_nodes(),
            "connection_count": graph.number_of_edges(),
            "cluster_count": len(clusters),
            "review_candidate_count": sum(
                cluster["status"] == "REVIEW_CANDIDATE"
                for cluster in clusters
            ),
            "clusters": clusters,
            "transaction_cluster": transaction_cluster,
            "limitations": [
                "Shared identifiers are investigative signals, not proof.",
                "Connected components can combine unrelated accounts through "
                "common infrastructure.",
                "Identifier values should be protected in logs and reports.",
                "Only the supplied transactions and identifier fields are used.",
            ],
        }
