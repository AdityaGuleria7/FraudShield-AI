import pytest

from app.ml.validation import ModelValidationGate


def test_validation_passes_when_model_meets_minimum_requirements():
    metrics = {
        "precision": 0.80,
        "recall": 0.75,
        "f1": 0.77,
        "roc_auc": 0.90,
        "pr_auc": 0.70,
    }

    gate = ModelValidationGate(
        minimum_precision=0.70,
        minimum_recall=0.70,
        minimum_f1=0.70,
        minimum_roc_auc=0.80,
        minimum_pr_auc=0.60,
    )

    result = gate.validate(metrics)

    assert result["status"] == "PASS"
    assert result["approved"] is True
    assert result["issues"] == []


def test_validation_rejects_model_below_threshold():
    metrics = {
        "precision": 0.60,
        "recall": 0.75,
        "f1": 0.67,
        "roc_auc": 0.90,
        "pr_auc": 0.70,
    }

    gate = ModelValidationGate(
        minimum_precision=0.70,
        minimum_recall=0.70,
        minimum_f1=0.70,
        minimum_roc_auc=0.80,
        minimum_pr_auc=0.60,
    )

    result = gate.validate(metrics)

    assert result["status"] == "REJECT"
    assert result["approved"] is False
    assert "precision" in result["issues"]
    assert "f1" in result["issues"]


def test_validation_rejects_missing_metric():
    metrics = {
        "precision": 0.80,
        "recall": 0.75,
        "f1": 0.77,
        "roc_auc": 0.90,
    }

    gate = ModelValidationGate()

    with pytest.raises(ValueError, match="Missing required metric"):
        gate.validate(metrics)
