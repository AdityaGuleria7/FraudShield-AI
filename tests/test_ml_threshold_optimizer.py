from app.ml.threshold_optimizer import CostSensitiveThresholdOptimizer


def test_threshold_optimizer_selects_lower_cost_threshold():
    y_true = [0, 0, 1, 1]
    probabilities = [0.10, 0.40, 0.60, 0.90]

    optimizer = CostSensitiveThresholdOptimizer(
        false_positive_cost=1.0,
        false_negative_cost=5.0,
    )

    result = optimizer.optimize(
        y_true=y_true,
        probabilities=probabilities,
        prediction_provenance="out_of_sample",
    )

    assert result["selected_threshold"] in result["evaluated_thresholds"]
    assert result["selected_threshold"] >= 0.0
    assert result["selected_threshold"] <= 1.0
    assert result["selected_cost"] >= 0.0
    assert result["optimization_basis"] == "validation_out_of_sample"


def test_threshold_optimizer_rejects_mismatched_lengths():
    optimizer = CostSensitiveThresholdOptimizer(
        false_positive_cost=1.0,
        false_negative_cost=5.0,
    )

    try:
        optimizer.optimize(
            y_true=[0, 1],
            probabilities=[0.2],
            prediction_provenance="out_of_sample",
        )
    except ValueError as exc:
        assert "same length" in str(exc).lower()
        return

    raise AssertionError(
        "Mismatched inputs were incorrectly accepted"
    )


def test_threshold_optimizer_rejects_invalid_probabilities():
    optimizer = CostSensitiveThresholdOptimizer(
        false_positive_cost=1.0,
        false_negative_cost=5.0,
    )

    try:
        optimizer.optimize(
            y_true=[0, 1],
            probabilities=[-0.1, 1.2],
            prediction_provenance="out_of_sample",
        )
    except ValueError as exc:
        assert "between 0.0 and 1.0" in str(exc)
        return

    raise AssertionError(
        "Invalid probabilities were incorrectly accepted"
    )


def test_threshold_optimizer_rejects_empty_data():
    optimizer = CostSensitiveThresholdOptimizer(
        false_positive_cost=1.0,
        false_negative_cost=5.0,
    )

    try:
        optimizer.optimize(
            y_true=[],
            probabilities=[],
            prediction_provenance="out_of_sample",
        )
    except ValueError as exc:
        assert "empty" in str(exc).lower()
        return

    raise AssertionError(
        "Empty validation data was incorrectly accepted"
    )


def test_threshold_optimizer_minimizes_false_negative_cost():
    optimizer = CostSensitiveThresholdOptimizer(
        false_positive_cost=1.0,
        false_negative_cost=10.0,
    )

    result = optimizer.optimize(
        y_true=[0, 0, 1, 1],
        probabilities=[0.10, 0.40, 0.60, 0.90],
        prediction_provenance="out_of_sample",
    )

    assert result["selected_threshold"] == 0.60
    assert result["selected_cost"] == 0.0
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0


def test_threshold_optimizer_prefers_lower_threshold_when_fraud_loss_is_expensive():
    optimizer = CostSensitiveThresholdOptimizer(
        false_positive_cost=1.0,
        false_negative_cost=20.0,
    )

    result = optimizer.optimize(
        y_true=[0, 0, 1, 1],
        probabilities=[0.20, 0.45, 0.55, 0.70],
        prediction_provenance="out_of_sample",
    )

    assert result["selected_threshold"] == 0.55
    assert result["false_negatives"] == 0
    assert result["false_positives"] == 0
    assert result["selected_cost"] == 0.0


def test_threshold_optimizer_rejects_in_sample_optimization():
    optimizer = CostSensitiveThresholdOptimizer(
        false_positive_cost=1.0,
        false_negative_cost=5.0,
    )

    try:
        optimizer.optimize(
            y_true=[0, 1],
            probabilities=[0.20, 0.80],
            prediction_provenance="in_sample",
        )
    except ValueError as exc:
        assert "out_of_sample" in str(exc).lower()
        return

    raise AssertionError(
        "In-sample predictions were incorrectly accepted for "
        "threshold optimization"
    )


def test_ml_pipeline_returns_cost_optimized_threshold():
    from app.ml.pipeline import MLPipeline

    transactions = [
        {
            "transaction_id": f"TX{index + 1}",
            "timestamp": f"2026-01-{index + 1:02d}T00:00:00+00:00",
            "is_fraud": index % 2 == 0,
        }
        for index in range(10)
    ]

    feature_vectors = {
        transaction["transaction_id"]: {
            "feature_a": float(index),
        }
        for index, transaction in enumerate(transactions)
    }

    pipeline = MLPipeline()
    result = pipeline.run(feature_vectors, transactions)

    assert "optimized_threshold" in result
    assert "threshold_optimization" in result
    assert result["threshold_optimization"]["optimization_basis"] == "validation_out_of_sample"



def test_ml_evidence_uses_optimized_threshold():
    from app.ml.evidence import MLEvidenceBuilder

    ml_result = {
        "model_type": "logistic_regression",
        "validation_prediction_provenance": "out_of_sample",
        "validation_transaction_ids": ["TX100"],
        "validation_predictions": [0],
        "validation_probabilities": [0.70],
        "optimized_threshold": 0.60,
        "validation": {
            "status": "PASS",
            "approved": True,
        },
        "threshold_policy": {
            "status": "PASS",
            "approved": True,
            "issues": [],
            "threshold": 0.60,
            "minimum_threshold": 0.30,
            "maximum_threshold": 0.80,
            "optimization_basis": "validation_out_of_sample",
            "false_positive_cost": 1.0,
            "false_negative_cost": 5.0,
        },
        "metrics": {},
    }

    evidence = MLEvidenceBuilder().build(
        ml_result=ml_result,
        transaction_id="TX100",
    )

    assert evidence["fraud_probability"] == 0.70
    assert evidence["decision_threshold"] == 0.60
    assert evidence["predicted_fraud"] is True


def test_ml_pipeline_validation_metrics_use_optimized_threshold():
    from app.ml.pipeline import MLPipeline

    transactions = [
        {
            "transaction_id": f"TX{index + 1}",
            "timestamp": f"2026-01-{index + 1:02d}T00:00:00+00:00",
            "is_fraud": index % 2 == 0,
        }
        for index in range(10)
    ]

    feature_vectors = {
        transaction["transaction_id"]: {
            "feature_a": float(index),
        }
        for index, transaction in enumerate(transactions)
    }

    result = MLPipeline().run(feature_vectors, transactions)

    assert result["optimized_threshold"] < 0.50
    assert result["metrics"]["precision"] > 0.0
    assert result["metrics"]["recall"] > 0.0
    assert result["metrics"]["f1"] > 0.0


def test_threshold_optimizer_includes_probability_at_selected_boundary():
    from app.ml.threshold_optimizer import CostSensitiveThresholdOptimizer

    optimizer = CostSensitiveThresholdOptimizer(
        false_positive_cost=1.0,
        false_negative_cost=5.0,
    )

    result = optimizer.optimize(
        y_true=[1, 0],
        probabilities=[0.3100264421804111, 0.2733332703060586],
        prediction_provenance="out_of_sample",
    )

    threshold = result["selected_threshold"]

    predictions = [
        int(probability >= threshold)
        for probability in [0.3100264421804111, 0.2733332703060586]
    ]

    assert predictions == [1, 0]
