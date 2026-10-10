from typing import Any


class CostSensitiveThresholdOptimizer:
    def __init__(
        self,
        false_positive_cost: float,
        false_negative_cost: float,
    ) -> None:
        if false_positive_cost < 0:
            raise ValueError("false_positive_cost must be non-negative")

        if false_negative_cost < 0:
            raise ValueError("false_negative_cost must be non-negative")

        self.false_positive_cost = false_positive_cost
        self.false_negative_cost = false_negative_cost

    def optimize(
        self,
        y_true: list[int],
        probabilities: list[float],
        prediction_provenance: str,
    ) -> dict[str, Any]:
        if prediction_provenance != "out_of_sample":
            raise ValueError(
                "Threshold optimization requires out_of_sample "
                "prediction provenance"
            )
        if len(y_true) != len(probabilities):
            raise ValueError(
                "y_true and probabilities must have the same length"
            )

        if not y_true:
            raise ValueError("Cannot optimize threshold on empty data")

        if any(
            probability < 0.0 or probability > 1.0
            for probability in probabilities
        ):
            raise ValueError(
                "Probabilities must be between 0.0 and 1.0"
            )

        thresholds = sorted(set(probabilities))

        if 0.0 not in thresholds:
            thresholds.insert(0, 0.0)

        if 1.0 not in thresholds:
            thresholds.append(1.0)

        evaluated = []

        for threshold in thresholds:
            predictions = [
                int(probability >= threshold)
                for probability in probabilities
            ]

            false_positives = sum(
                1
                for actual, predicted in zip(y_true, predictions)
                if actual == 0 and predicted == 1
            )

            false_negatives = sum(
                1
                for actual, predicted in zip(y_true, predictions)
                if actual == 1 and predicted == 0
            )

            cost = (
                false_positives * self.false_positive_cost
                + false_negatives * self.false_negative_cost
            )

            evaluated.append(
                {
                    "threshold": threshold,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                    "cost": cost,
                }
            )

        best = min(
            evaluated,
            key=lambda candidate: (
                candidate["cost"],
                candidate["threshold"],
            ),
        )

        return {
            "selected_threshold": best["threshold"],
            "selected_cost": best["cost"],
            "evaluated_thresholds": [
                candidate["threshold"]
                for candidate in evaluated
            ],
            "false_positives": best["false_positives"],
            "false_negatives": best["false_negatives"],
            "optimization_basis": "validation_out_of_sample",
            "evaluations": evaluated,
        }
