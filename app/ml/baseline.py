from typing import Any

from sklearn.linear_model import LogisticRegression


class MLBaselineModel:
    def __init__(self) -> None:
        self.model = LogisticRegression(random_state=42)
        self.feature_names: list[str] = []
        self.trained = False

    def train(self, dataset: dict[str, Any]) -> dict[str, Any]:
        self.feature_names = dataset["feature_names"]
        X = dataset["X"]
        y = dataset["y"]

        self.model.fit(X, y)
        self.trained = True

        return {
            "model_type": "logistic_regression",
            "feature_names": self.feature_names,
            "trained": True,
        }

    def predict(self, X: list[list[Any]]) -> list[int]:
        if not self.trained:
            raise RuntimeError("Model has not been trained")

        return self.model.predict(X).tolist()

    def predict_proba(self, X: list[list[Any]]) -> list[float]:
        if not self.trained:
            raise RuntimeError("Model has not been trained")

        return self.model.predict_proba(X)[:, 1].tolist()
