from typing import Any


class MLEvidenceBuilder:
    def build(
        self,
        ml_result: dict[str, Any],
        transaction_id: str,
    ) -> dict[str, Any]:
        transaction_ids = ml_result.get("transaction_ids", [])

        if transaction_id not in transaction_ids:
            raise ValueError(f"No ML prediction for transaction {transaction_id}")

        transaction_index = transaction_ids.index(transaction_id)

        predictions = ml_result.get("predictions", [])
        probabilities = ml_result.get("probabilities", [])

        if transaction_index >= len(predictions):
            raise ValueError(f"No ML prediction for transaction {transaction_id}")

        if transaction_index >= len(probabilities):
            raise ValueError(f"No ML probability for transaction {transaction_id}")

        validation = ml_result.get("validation", {})

        return {
            "evidence_type": "ml_model_prediction",
            "transaction_id": transaction_id,
            "model_type": ml_result["model_type"],
            "predicted_fraud": bool(predictions[transaction_index]),
            "fraud_probability": float(probabilities[transaction_index]),
            "validation_status": validation.get("status"),
            "model_approved": bool(validation.get("approved", False)),
            "metrics": ml_result.get("metrics", {}),
        }
