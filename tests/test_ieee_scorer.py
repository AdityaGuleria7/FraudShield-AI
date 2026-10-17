import numpy as np
import pytest

from app.ml import ieee_scorer


class FakeModel:
    def predict_proba(self, frame):
        assert len(frame) > 0
        return np.array([[0.6, 0.4] for _ in range(len(frame))])


@pytest.fixture
def mock_artifact(monkeypatch):
    artifact = {
        "model": FakeModel(),
        "model_type": "test_model",
        "features": ["TransactionAmt", "card1", "ProductCD"],
        "numeric_features": ["TransactionAmt"],
        "categorical_features": ["card1", "ProductCD"],
        "target": "isFraud",
        "training_rows": 100,
        "training_fraud_rate": 0.03,
        "threshold": None,
        "score_semantics": "Test score; not a calibrated fraud probability.",
        "validation_note": "Test artifact only.",
    }
    monkeypatch.setattr(ieee_scorer, "load_artifact", lambda: artifact)
    return artifact


def valid_transaction():
    return {
        "TransactionID": 999001,
        "TransactionAmt": 125.5,
        "card1": 12345,
        "ProductCD": "W",
        "identity_available": 0,
    }


def test_valid_transaction_is_scored_without_a_decision(mock_artifact):
    result = ieee_scorer.score_transactions([valid_transaction()])

    assert result["scored_count"] == 1
    assert result["status"] == "EXPERIMENTAL_NOT_APPROVED"
    assert result["decision_made"] is False
    assert result["results"][0]["TransactionID"] == 999001
    assert result["results"][0]["experimental_model_score"] == 0.4
    assert result["results"][0]["input_coverage"] == {
        "available_features": 3,
        "expected_features": 3,
        "missing_features": 0,
    }


def test_missing_optional_model_features_are_counted(mock_artifact):
    transaction = valid_transaction()
    del transaction["card1"]

    result = ieee_scorer.score_transactions([transaction])
    coverage = result["results"][0]["input_coverage"]

    assert coverage == {
        "available_features": 2,
        "expected_features": 3,
        "missing_features": 1,
    }


def test_missing_transaction_amount_is_rejected(mock_artifact):
    transaction = valid_transaction()
    del transaction["TransactionAmt"]

    with pytest.raises(ValueError):
        ieee_scorer.score_transactions([transaction])


def test_invalid_identity_availability_is_rejected(mock_artifact):
    transaction = valid_transaction()
    transaction["identity_available"] = 2

    with pytest.raises(ValueError):
        ieee_scorer.score_transactions([transaction])


def test_unexpected_fields_are_rejected(mock_artifact):
    transaction = valid_transaction()
    transaction["isFraud"] = 1

    with pytest.raises(ValueError):
        ieee_scorer.score_transactions([transaction])


def test_batches_over_100_are_rejected(mock_artifact):
    transactions = [valid_transaction() for _ in range(101)]

    with pytest.raises(ValueError):
        ieee_scorer.score_transactions(transactions)


def test_empty_batch_is_rejected(mock_artifact):
    with pytest.raises(ValueError):
        ieee_scorer.score_transactions([])


def test_negative_transaction_amount_is_rejected(mock_artifact):
    transaction = valid_transaction()
    transaction["TransactionAmt"] = -1

    with pytest.raises(ValueError):
        ieee_scorer.score_transactions([transaction])



def test_pandas_nan_is_treated_as_missing():
    assert ieee_scorer._is_missing(float("nan")) is True
    assert ieee_scorer._is_missing(float("inf")) is False
