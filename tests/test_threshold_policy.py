from app.ml.threshold_policy import ThresholdPolicyGate


def test_threshold_policy_rejects_threshold_outside_allowed_range():
    policy = ThresholdPolicyGate(
        minimum_threshold=0.30,
        maximum_threshold=0.80,
    )

    result = policy.validate(
        threshold=0.90,
        validation={
            "status": "PASS",
            "approved": True,
        },
        optimization_basis="validation_out_of_sample",
    )

    assert result["status"] == "REJECT"
    assert result["approved"] is False
    assert "threshold_range" in result["issues"]


def test_threshold_policy_rejects_unvalidated_model():
    policy = ThresholdPolicyGate(
        minimum_threshold=0.30,
        maximum_threshold=0.80,
    )

    result = policy.validate(
        threshold=0.60,
        validation={
            "status": "REJECT",
            "approved": False,
        },
        optimization_basis="validation_out_of_sample",
    )

    assert result["status"] == "REJECT"
    assert result["approved"] is False
    assert "model_validation_status" in result["issues"]
    assert "model_validation_approval" in result["issues"]


def test_threshold_policy_rejects_in_sample_threshold():
    policy = ThresholdPolicyGate(
        minimum_threshold=0.30,
        maximum_threshold=0.80,
    )

    result = policy.validate(
        threshold=0.60,
        validation={
            "status": "PASS",
            "approved": True,
        },
        optimization_basis="in_sample",
    )

    assert result["status"] == "REJECT"
    assert result["approved"] is False
    assert "optimization_provenance" in result["issues"]


def test_threshold_policy_rejects_invalid_threshold_value():
    policy = ThresholdPolicyGate(
        minimum_threshold=0.30,
        maximum_threshold=0.80,
    )

    result = policy.validate(
        threshold=1.20,
        validation={
            "status": "PASS",
            "approved": True,
        },
        optimization_basis="validation_out_of_sample",
    )

    assert result["status"] == "REJECT"
    assert result["approved"] is False
    assert "threshold_range" in result["issues"]


def test_threshold_policy_rejects_missing_cost_configuration():
    policy = ThresholdPolicyGate(
        minimum_threshold=0.30,
        maximum_threshold=0.80,
    )

    result = policy.validate(
        threshold=0.60,
        validation={
            "status": "PASS",
            "approved": True,
        },
        optimization_basis="validation_out_of_sample",
        false_positive_cost=None,
        false_negative_cost=None,
    )

    assert result["status"] == "REJECT"
    assert result["approved"] is False
    assert "cost_configuration" in result["issues"]


def test_threshold_policy_records_decision_context():
    policy = ThresholdPolicyGate(
        minimum_threshold=0.30,
        maximum_threshold=0.80,
    )

    result = policy.validate(
        threshold=0.60,
        validation={
            "status": "PASS",
            "approved": True,
        },
        optimization_basis="validation_out_of_sample",
        false_positive_cost=1.0,
        false_negative_cost=5.0,
    )

    assert result["status"] == "PASS"
    assert result["approved"] is True
    assert result["threshold"] == 0.60
    assert result["minimum_threshold"] == 0.30
    assert result["maximum_threshold"] == 0.80
    assert result["optimization_basis"] == "validation_out_of_sample"
    assert result["false_positive_cost"] == 1.0
    assert result["false_negative_cost"] == 5.0


def test_ml_pipeline_returns_threshold_policy_decision():
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

    assert "threshold_policy" in result
    assert result["threshold_policy"]["status"] == "PASS"
    assert result["threshold_policy"]["approved"] is True
    assert (
        result["threshold_policy"]["optimization_basis"]
        == "validation_out_of_sample"
    )
