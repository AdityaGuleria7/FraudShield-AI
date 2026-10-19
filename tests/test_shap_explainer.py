"""Unit tests for the SHAP explainability engine."""
import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from app.explainability.shap_explainer import (
    explain_with_shap,
    explain_batch_with_shap,
    _label,
    _plain_driver_description,
)


@pytest.fixture
def trained_tree_artifact():
    # Train a minimal tree model on synthetic data
    X = np.array([
        [10.0, 0, 1],
        [500.0, 1, 0],
        [25.0, 0, 1],
        [1200.0, 1, 0],
        [50.0, 0, 1],
        [3000.0, 1, 0],
    ])
    y = np.array([0, 1, 0, 1, 0, 1])
    features = ["TransactionAmt", "card1", "ProductCD"]

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)

    return {
        "model": model,
        "model_type": "DecisionTreeClassifier",
        "features": features,
        "numeric_features": ["TransactionAmt", "card1", "ProductCD"],
        "categorical_features": [],
    }


def test_feature_label_fallback():
    assert _label("TransactionAmt") == "Transaction Amount"
    assert _label("unknown_custom_feat") == "Unknown Custom Feat"


def test_plain_driver_description():
    desc_pos = _plain_driver_description("TransactionAmt", 500.0, 0.25)
    assert "Transaction amount $500.00" in desc_pos
    assert "pushes the model score higher" in desc_pos

    desc_neg = _plain_driver_description("identity_available", 1, -0.15)
    assert "Identity data available" in desc_neg
    assert "pushes the model score lower" in desc_neg


def test_explain_with_shap_single_row(trained_tree_artifact):
    row = {"TransactionAmt": 1500.0, "card1": 1, "ProductCD": 0}
    explanation = explain_with_shap(trained_tree_artifact, row, top_n=3)

    assert explanation["explanation_type"] == "shap_tree_explainer"
    assert "top_n_drivers" in explanation
    assert len(explanation["top_n_drivers"]) == 3

    driver = explanation["top_n_drivers"][0]
    assert "feature" in driver
    assert "shap_value" in driver
    assert "direction" in driver
    assert "impact_magnitude" in driver
    assert "plain_description" in driver


def test_explain_batch_with_shap(trained_tree_artifact):
    rows = [
        {"TransactionAmt": 15.0, "card1": 0, "ProductCD": 1},
        {"TransactionAmt": 2500.0, "card1": 1, "ProductCD": 0},
    ]
    batch_explanations = explain_batch_with_shap(trained_tree_artifact, rows, top_n=2)

    assert len(batch_explanations) == 2
    for exp in batch_explanations:
        assert exp["explanation_type"] == "shap_tree_explainer"
        assert len(exp["top_n_drivers"]) == 2


@pytest.mark.parametrize(
    ("shap_value", "expected_direction", "expected_phrase"),
    [
        (0.0, "neutral", "does not change the model score"),
        (0.15, "increases_risk", "pushes the model score higher"),
        (-0.15, "decreases_risk", "pushes the model score lower"),
    ],
)
def test_shap_driver_direction_handles_zero_and_signed_values(
    shap_value, expected_direction, expected_phrase
):
    from app.explainability.shap_explainer import _driver_direction

    assert _driver_direction(shap_value) == expected_direction
    description = _plain_driver_description(
        "TransactionAmt", 100.0, shap_value
    )
    assert expected_phrase in description
