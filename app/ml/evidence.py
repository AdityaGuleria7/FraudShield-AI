from typing import Any


class MLEvidenceBuilder:
    def build(
        self,
        ml_result: dict[str, Any],
        transaction_id: str,
    ) -> dict[str, Any]:
        validation_provenance = ml_result.get(
            "validation_prediction_provenance"
        )

        if validation_provenance != "out_of_sample":
            raise ValueError(
                "ML evidence requires out_of_sample prediction provenance"
            )

        validation_transaction_ids = ml_result.get(
            "validation_transaction_ids",
            [],
        )

        if transaction_id not in validation_transaction_ids:
            raise ValueError(
                f"No out_of_sample ML prediction for transaction "
                f"{transaction_id}"
            )

        transaction_index = validation_transaction_ids.index(transaction_id)

        predictions = ml_result.get("validation_predictions", [])
        probabilities = ml_result.get("validation_probabilities", [])

        if transaction_index >= len(predictions):
            raise ValueError(
                f"No out_of_sample ML prediction for transaction "
                f"{transaction_id}"
            )

        if transaction_index >= len(probabilities):
            raise ValueError(
                f"No out_of_sample ML probability for transaction "
                f"{transaction_id}"
            )

        validation = ml_result.get("validation", {})

        return {
            "evidence_type": "ml_model_prediction",
            "transaction_id": transaction_id,
            "model_type": ml_result["model_type"],
            "predicted_fraud": bool(predictions[transaction_index]),
            "fraud_probability": float(probabilities[transaction_index]),
            "prediction_provenance": validation_provenance,
            "validation_status": validation.get("status"),
            "model_approved": bool(validation.get("approved", False)),
            "metrics": ml_result.get("metrics", {}),
        }
