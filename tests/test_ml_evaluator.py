from app.ml.evaluator import MLEvaluator


def sample_predictions():
    return {
        "y_true": [0, 0, 1, 1],
        "y_pred": [0, 1, 1, 1],
    }


def test_evaluator_returns_core_metrics():
    data = sample_predictions()

    evaluator = MLEvaluator()

    result = evaluator.evaluate(
        y_true=data["y_true"],
        y_pred=data["y_pred"],
    )

    assert result["precision"] == 2 / 3
    assert result["recall"] == 1.0
    assert result["f1"] == 0.8


def test_evaluator_returns_confusion_matrix():
    data = sample_predictions()

    evaluator = MLEvaluator()

    result = evaluator.evaluate(
        y_true=data["y_true"],
        y_pred=data["y_pred"],
    )

    assert result["confusion_matrix"] == [
        [1, 1],
        [0, 2],
    ]


def test_evaluator_rejects_mismatched_lengths():
    evaluator = MLEvaluator()

    try:
        evaluator.evaluate(
            y_true=[0, 1],
            y_pred=[0],
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "same length" in str(exc)


def test_evaluator_returns_probability_metrics():
    data = sample_predictions()

    evaluator = MLEvaluator()

    result = evaluator.evaluate(
        y_true=data["y_true"],
        y_pred=data["y_pred"],
        y_score=[0.1, 0.6, 0.8, 0.9],
    )

    assert "roc_auc" in result
    assert "pr_auc" in result
    assert 0.0 <= result["roc_auc"] <= 1.0
    assert 0.0 <= result["pr_auc"] <= 1.0


def test_evaluator_rejects_mismatched_score_lengths():
    evaluator = MLEvaluator()

    try:
        evaluator.evaluate(
            y_true=[0, 1],
            y_pred=[0, 1],
            y_score=[0.2],
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "same length" in str(exc)
