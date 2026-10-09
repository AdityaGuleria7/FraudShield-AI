from typing import Any

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MLEvaluator:
    def evaluate(
        self,
        y_true: list[int],
        y_pred: list[int],
        y_score: list[float] | None = None,
    ) -> dict[str, Any]:
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        if y_score is not None and len(y_true) != len(y_score):
            raise ValueError("y_true and y_score must have the same length")

        result = {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

        if y_score is not None:
            result["roc_auc"] = roc_auc_score(y_true, y_score)
            result["pr_auc"] = average_precision_score(y_true, y_score)

        return result
