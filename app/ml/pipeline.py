from typing import Any

from app.ml.baseline import MLBaselineModel
from app.ml.dataset import FraudDatasetBuilder
from app.ml.evaluator import MLEvaluator
from app.ml.validation import ModelValidationGate


class MLPipeline:
    def __init__(self) -> None:
        self.dataset_builder = FraudDatasetBuilder()
        self.model = MLBaselineModel()
        self.evaluator = MLEvaluator()
        self.validation_gate = ModelValidationGate()

    def run(
        self,
        feature_vectors: dict[str, dict[str, Any]],
        transactions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        dataset = self.dataset_builder.build(
            feature_vectors,
            transactions,
        )

        training_result = self.model.train(dataset)

        predictions = self.model.predict(dataset["X"])
        probabilities = self.model.predict_proba(dataset["X"])

        metrics = self.evaluator.evaluate(
            y_true=dataset["y"],
            y_pred=predictions,
            y_score=probabilities,
        )

        validation = self.validation_gate.validate(metrics)

        return {
            **training_result,
            "predictions": predictions,
            "probabilities": probabilities,
            "metrics": metrics,
            "validation": validation,
            "dataset_size": len(dataset["y"]),
            "fraud_count": sum(dataset["y"]),
            "non_fraud_count": len(dataset["y"]) - sum(dataset["y"]),
        }
