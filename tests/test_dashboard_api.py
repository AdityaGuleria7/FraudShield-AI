import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import dashboard
from app.ml import ieee_scorer


class FakeModel:
    def predict_proba(self, frame):
        return np.array([[0.6, 0.4] for _ in range(len(frame))])


@pytest.fixture
def client():
    return TestClient(dashboard.app)


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


def test_health_endpoint(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["application"] == "FraudShield AI"


def test_dashboard_homepage(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "FraudShield AI" in response.text


def test_ieee_score_endpoint_returns_experimental_score(
    client, mock_artifact
):
    response = client.post(
        "/api/ieee-score",
        json={"transactions": [valid_transaction()]},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "EXPERIMENTAL_NOT_APPROVED"
    assert result["scored_count"] == 1
    assert result["decision_made"] is False
    assert result["results"][0]["experimental_model_score"] == 0.4


def test_ieee_score_rejects_missing_transactions_array(client):
    response = client.post("/api/ieee-score", json={})

    assert response.status_code == 400
    assert "transactions" in response.json()["detail"]


def test_ieee_score_rejects_invalid_transaction(client, mock_artifact):
    transaction = valid_transaction()
    transaction["identity_available"] = 2

    response = client.post(
        "/api/ieee-score",
        json={"transactions": [transaction]},
    )

    assert response.status_code == 422


def test_ieee_score_returns_503_when_artifact_is_missing(
    client, monkeypatch
):
    def missing_artifact():
        raise FileNotFoundError("Test artifact unavailable.")

    monkeypatch.setattr(ieee_scorer, "load_artifact", missing_artifact)

    response = client.post(
        "/api/ieee-score",
        json={"transactions": [valid_transaction()]},
    )

    assert response.status_code == 503
    assert "Test artifact unavailable" in response.json()["detail"]


def test_csv_endpoint_rejects_non_csv_file(client):
    response = client.post(
        "/api/analyze-csv",
        files={"file": ("transactions.txt", "not a CSV", "text/plain")},
    )

    assert response.status_code == 400
    assert "csv" in response.json()["detail"].lower()
