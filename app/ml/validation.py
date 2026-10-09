from typing import Any


class ModelValidationGate:
    REQUIRED_METRICS = (
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    )

    def __init__(
        self,
        minimum_precision: float = 0.70,
        minimum_recall: float = 0.70,
        minimum_f1: float = 0.70,
        minimum_roc_auc: float = 0.80,
        minimum_pr_auc: float = 0.60,
    ) -> None:
        self.thresholds = {
            "precision": minimum_precision,
            "recall": minimum_recall,
            "f1": minimum_f1,
            "roc_auc": minimum_roc_auc,
            "pr_auc": minimum_pr_auc,
        }

    def validate(self, metrics: dict[str, Any]) -> dict[str, Any]:
        for metric in self.REQUIRED_METRICS:
            if metric not in metrics:
                raise ValueError(f"Missing required metric: {metric}")

        issues = [
            metric
            for metric, minimum in self.thresholds.items()
            if metrics[metric] < minimum
        ]

        if issues:
            return {
                "status": "REJECT",
                "approved": False,
                "issues": issues,
            }

        return {
            "status": "PASS",
            "approved": True,
            "issues": [],
        }
