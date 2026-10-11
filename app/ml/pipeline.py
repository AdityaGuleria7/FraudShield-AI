from typing import Any

from app.ml.baseline import MLBaselineModel
from app.ml.dataset import FraudDatasetBuilder
from app.ml.evaluator import MLEvaluator
from app.ml.validation import ModelValidationGate
from app.ml.threshold_optimizer import CostSensitiveThresholdOptimizer
from app.ml.threshold_policy import ThresholdPolicyGate


class MLPipeline:
    def __init__(self) -> None:
        self.dataset_builder = FraudDatasetBuilder()
        self.model = MLBaselineModel()
        self.evaluator = MLEvaluator()
        self.validation_gate = ModelValidationGate()
        self.threshold_optimizer = CostSensitiveThresholdOptimizer(
            false_positive_cost=1.0,
            false_negative_cost=5.0,
        )
        self.threshold_policy = ThresholdPolicyGate()

    def run(
        self,
        feature_vectors: dict[str, dict[str, Any]],
        transactions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        dataset = self.dataset_builder.build(
            feature_vectors,
            transactions,
        )

        transaction_ids = [
            transaction["transaction_id"]
            for transaction in transactions
        ]

        if len(dataset["y"]) < 5:
            raise ValueError(
                "Dataset is too small for a temporal holdout split."
            )

        timestamp_by_index = {}

        for index, transaction in enumerate(transactions):
            timestamp = transaction.get("timestamp")

            if timestamp is None:
                raise ValueError(
                    "Every transaction must contain a timestamp for "
                    "temporal holdout validation."
                )

            timestamp_by_index[index] = timestamp

        ordered_indices = sorted(
            range(len(dataset["y"])),
            key=lambda index: timestamp_by_index[index],
        )

        validation_size = max(
            1,
            int(len(dataset["y"]) * 0.20 + 0.999999),
        )

        if validation_size >= len(dataset["y"]):
            raise ValueError(
                "Dataset is too small for a temporal holdout split."
            )

        split_index = len(ordered_indices) - validation_size

        train_indices = ordered_indices[:split_index]
        validation_indices = ordered_indices[split_index:]

        X_train = [dataset["X"][index] for index in train_indices]
        y_train = [dataset["y"][index] for index in train_indices]

        X_validation = [
            dataset["X"][index]
            for index in validation_indices
        ]
        y_validation = [
            dataset["y"][index]
            for index in validation_indices
        ]

        training_dataset = {
            "feature_names": dataset["feature_names"],
            "X": X_train,
            "y": y_train,
        }

        training_result = self.model.train(training_dataset)

        validation_probabilities = self.model.predict_proba(X_validation)

        threshold_optimization = self.threshold_optimizer.optimize(
            y_true=y_validation,
            probabilities=validation_probabilities,
            prediction_provenance="out_of_sample",
        )

        optimized_threshold = threshold_optimization["selected_threshold"]

        validation_predictions = [
            int(probability >= optimized_threshold)
            for probability in validation_probabilities
        ]

        metrics = self.evaluator.evaluate(
            y_true=y_validation,
            y_pred=validation_predictions,
            y_score=validation_probabilities,
        )

        predictions = self.model.predict(dataset["X"])
        probabilities = self.model.predict_proba(dataset["X"])

        validation = self.validation_gate.validate(metrics)

        threshold_policy = self.threshold_policy.validate(
            threshold=threshold_optimization["selected_threshold"],
            validation=validation,
            optimization_basis=threshold_optimization["optimization_basis"],
            false_positive_cost=self.threshold_optimizer.false_positive_cost,
            false_negative_cost=self.threshold_optimizer.false_negative_cost,
        )

        train_transaction_ids = [
            transaction_ids[index]
            for index in train_indices
        ]

        validation_transaction_ids = [
            transaction_ids[index]
            for index in validation_indices
        ]

        return {
            **training_result,
            "transaction_ids": transaction_ids,
            "train_transaction_ids": train_transaction_ids,
            "validation_transaction_ids": validation_transaction_ids,
            "predictions": predictions,
            "probabilities": probabilities,
            "prediction_provenance": "in_sample",
            "validation_predictions": validation_predictions,
            "validation_probabilities": validation_probabilities,
            "validation_prediction_provenance": "out_of_sample",
            "metrics": metrics,
            "validation": validation,
            "optimized_threshold": threshold_optimization["selected_threshold"],
            "threshold_optimization": threshold_optimization,
            "threshold_policy": threshold_policy,
            "dataset_size": len(dataset["y"]),
            "train_dataset_size": len(y_train),
            "validation_dataset_size": len(y_validation),
            "fraud_count": sum(dataset["y"]),
            "non_fraud_count": len(dataset["y"]) - sum(dataset["y"]),
        }
