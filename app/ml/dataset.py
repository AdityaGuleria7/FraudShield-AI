from typing import Any


class FraudDatasetBuilder:
    def build(
        self,
        feature_vectors: dict[str, dict[str, Any]],
        transactions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not feature_vectors:
            raise ValueError("No feature vectors supplied")

        feature_names = sorted(
            next(iter(feature_vectors.values())).keys()
        )

        X: list[list[Any]] = []
        y: list[int] = []

        for transaction in transactions:
            transaction_id = transaction["transaction_id"]

            if transaction_id not in feature_vectors:
                raise ValueError(
                    f"Missing feature vector for transaction {transaction_id}"
                )

            label = transaction.get("is_fraud")

            if not isinstance(label, bool):
                raise ValueError(
                    f"Invalid fraud label for transaction {transaction_id}"
                )

            vector = feature_vectors[transaction_id]

            missing_features = set(feature_names) - vector.keys()

            if missing_features:
                raise ValueError(
                    f"Missing features for transaction {transaction_id}"
                )

            X.append([vector[name] for name in feature_names])
            y.append(int(label))

        return {
            "feature_names": feature_names,
            "X": X,
            "y": y,
        }
