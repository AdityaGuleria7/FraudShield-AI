import pytest

from app.ml.baseline import MLBaselineModel


def sample_dataset():
    return {
        "feature_names": [
            "account_unique_device_count",
            "transaction_amount",
            "transactions_last_5m",
        ],
        "X": [
            [1, 1000.0, 0],
            [2, 5000.0, 3],
            [1, 1200.0, 0],
            [3, 9000.0, 5],
        ],
        "y": [0, 1, 0, 1],
    }


def test_baseline_model_trains_and_returns_metadata():
    dataset = sample_dataset()

    model = MLBaselineModel()

    result = model.train(dataset)

    assert result["model_type"] == "logistic_regression"
    assert result["feature_names"] == dataset["feature_names"]
    assert result["trained"] is True


def test_baseline_model_can_predict():
    dataset = sample_dataset()

    model = MLBaselineModel()
    model.train(dataset)

    predictions = model.predict(dataset["X"])

    assert len(predictions) == len(dataset["X"])
    assert all(prediction in [0, 1] for prediction in predictions)


def test_prediction_requires_trained_model():
    model = MLBaselineModel()

    with pytest.raises(RuntimeError, match="Model has not been trained"):
        model.predict([[1, 1000.0, 0]])


def test_baseline_model_returns_fraud_probabilities():
    dataset = sample_dataset()

    model = MLBaselineModel()
    model.train(dataset)

    probabilities = model.predict_proba(dataset["X"])

    assert len(probabilities) == len(dataset["X"])
    assert all(0.0 <= probability <= 1.0 for probability in probabilities)
