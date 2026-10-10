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
        decision_threshold = ml_result.get("optimized_threshold")

        if decision_threshold is None:
            raise ValueError(
                "ML evidence requires an optimized decision threshold"
            )

        probability = float(probabilities[transaction_index])
        predicted_fraud = probability >= float(decision_threshold)

        return {
            "evidence_type": "ml_model_prediction",
            "transaction_id": transaction_id,
            "model_type": ml_result["model_type"],
            "predicted_fraud": predicted_fraud,
            "fraud_probability": probability,
            "decision_threshold": float(decision_threshold),
            "prediction_provenance": validation_provenance,
            "validation_status": validation.get("status"),
            "model_approved": bool(validation.get("approved", False)),
            "metrics": ml_result.get("metrics", {}),
        }
