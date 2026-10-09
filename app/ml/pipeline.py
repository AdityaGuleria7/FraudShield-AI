from typing import Any

from sklearn.model_selection import train_test_split

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

        transaction_ids = [
            transaction["transaction_id"]
            for transaction in transactions
        ]

        class_counts = {
            label: dataset["y"].count(label)
            for label in set(dataset["y"])
        }

        if len(class_counts) < 2 or min(class_counts.values()) < 2:
            raise ValueError(
                "Dataset is too small for a stratified holdout split; "
                "each class requires at least 2 samples."
            )

        validation_size = max(
            len(class_counts),
            int(len(dataset["y"]) * 0.20 + 0.999999),
        )

        if validation_size >= len(dataset["y"]):
            raise ValueError(
                "Dataset is too small for a stratified holdout split."
            )

        train_indices, validation_indices = train_test_split(
            list(range(len(dataset["y"]))),
            test_size=validation_size,
            random_state=42,
            stratify=dataset["y"],
        )

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

        validation_predictions = self.model.predict(X_validation)
        validation_probabilities = self.model.predict_proba(X_validation)

        metrics = self.evaluator.evaluate(
            y_true=y_validation,
            y_pred=validation_predictions,
            y_score=validation_probabilities,
        )

        predictions = self.model.predict(dataset["X"])
        probabilities = self.model.predict_proba(dataset["X"])

        validation = self.validation_gate.validate(metrics)

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
            "dataset_size": len(dataset["y"]),
            "train_dataset_size": len(y_train),
            "validation_dataset_size": len(y_validation),
            "fraud_count": sum(dataset["y"]),
            "non_fraud_count": len(dataset["y"]) - sum(dataset["y"]),
        }
