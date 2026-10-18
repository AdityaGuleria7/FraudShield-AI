import base64
import numpy as np
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app import dashboard
from app.ml import ieee_scorer


class FakeModel:
    def predict_proba(self, frame):
        return np.array([[0.6, 0.4] for _ in range(len(frame))])


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("FRAUDSHIELD_AUTH_USERNAME", "fraudshield-test-user")
    monkeypatch.setenv("FRAUDSHIELD_AUTH_PASSWORD", "fraudshield-test-password")
    token = base64.b64encode(b"fraudshield-test-user:fraudshield-test-password").decode()
    return TestClient(
        dashboard.app, headers={"Authorization": f"Basic {token}"}
    )


def test_dashboard_explains_shap_values_as_model_score_contributions(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "Top model-score contributions" in response.text
    assert "Positive values push the model score higher" in response.text
    assert "not fraud probabilities and do not prove fraud" in response.text
    assert "const color = 'var(--muted)'" in response.text
    assert "const isRisk = d.direction" not in response.text
    assert "Top SHAP Risk Drivers" not in response.text
    assert "sparse-input demonstration" in response.text
    assert "may affect the score" in response.text
    assert "does not supply" in response.text
    assert "not a calibrated fraud" in response.text
    assert "investigation signal" in response.text
    assert "does not confirm fraud" in response.text


def test_dashboard_requires_authentication(monkeypatch):
    monkeypatch.setenv("FRAUDSHIELD_AUTH_USERNAME", "fraudshield-test-user")
    monkeypatch.setenv("FRAUDSHIELD_AUTH_PASSWORD", "fraudshield-test-password")

    with TestClient(dashboard.app) as unauthenticated_client:
        response = unauthenticated_client.get("/")
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Basic"


def test_dashboard_rejects_invalid_credentials(monkeypatch):
    monkeypatch.setenv("FRAUDSHIELD_AUTH_USERNAME", "fraudshield-test-user")
    monkeypatch.setenv("FRAUDSHIELD_AUTH_PASSWORD", "fraudshield-test-password")

    token = base64.b64encode(b"fraudshield-test-user:wrong-password").decode()
    with TestClient(
        dashboard.app, headers={"Authorization": f"Basic {token}"}
    ) as invalid_client:
        response = invalid_client.get("/")
    assert response.status_code == 401


def test_dashboard_fails_closed_without_auth_configuration(monkeypatch):
    monkeypatch.delenv("FRAUDSHIELD_AUTH_USERNAME", raising=False)
    monkeypatch.delenv("FRAUDSHIELD_AUTH_PASSWORD", raising=False)

    token = base64.b64encode(b"test-user:test-password").decode()
    with TestClient(
        dashboard.app, headers={"Authorization": f"Basic {token}"}
    ) as configured_client:
        response = configured_client.get("/")
    assert response.status_code == 503


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



def valid_analysis_csv(amount="125.50"):
    return (
        "transaction_id,account_id,device_id,ip_address,country,"
        "merchant_category,amount,timestamp,is_fraud\n"
        f"txn-1,acct-1,dev-1,192.0.2.1,IN,retail,{amount},"
        "2026-01-01T10:00:00Z,false\n"
    )


@pytest.mark.parametrize("amount", ["NaN", "Infinity", "-Infinity"])
def test_parse_csv_rejects_non_finite_amounts(amount):
    with pytest.raises(dashboard.HTTPException) as exc:
        dashboard.parse_csv(valid_analysis_csv(amount).encode())

    assert exc.value.status_code == 400
    assert "amount" in exc.value.detail.lower()


def test_parse_csv_accepts_valid_transaction():
    transactions = dashboard.parse_csv(valid_analysis_csv().encode())

    assert len(transactions) == 1
    assert transactions[0]["amount"] == 125.5
    assert transactions[0]["is_fraud"] is False


@pytest.mark.parametrize(
    "header",
    [
        "transaction_id,account_id,device_id,ip_address,country,"
        "merchant_category,amount,timestamp,is_fraud,is_fraud",
        ",transaction_id,account_id,device_id,ip_address,country,"
        "merchant_category,amount,timestamp,is_fraud",
    ],
)
def test_parse_csv_rejects_duplicate_or_blank_headers(header):
    content = header + "\n" + ",".join(["x"] * len(header.split(","))) + "\n"

    with pytest.raises(dashboard.HTTPException) as exc:
        dashboard.parse_csv(content.encode())

    assert exc.value.status_code == 400


@pytest.mark.parametrize(
    "row",
    [
        "txn-1,acct-1,dev-1,192.0.2.1,IN,retail,125.50,"
        "2026-01-01T10:00:00Z,false,EXTRA",
        "txn-1,acct-1,dev-1,192.0.2.1,IN,retail,125.50,"
        "2026-01-01T10:00:00Z",
    ],
)
def test_parse_csv_rejects_rows_with_wrong_field_count(row):
    header = (
        "transaction_id,account_id,device_id,ip_address,country,"
        "merchant_category,amount,timestamp,is_fraud\n"
    )

    with pytest.raises(dashboard.HTTPException) as exc:
        dashboard.parse_csv((header + row + "\n").encode())

    assert exc.value.status_code == 400

def test_health_endpoint(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["application"] == "FraudShield AI"


def test_dashboard_homepage(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "FraudShield AI" in response.text
    assert 'id="ieeeCsvScoringPanel"' in response.text
    assert 'id="ieeeScoringPanel"' in response.text
    assert response.text.index('id="ieeeCsvScoringPanel"') < response.text.index(
        'id="ieeeScoringPanel"'
    )
    assert "<summary>Advanced: score a single transaction</summary>" in response.text


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
    assert response.json()["detail"] == "IEEE-CIS model artifact unavailable."
    assert "Test artifact unavailable" not in response.json()["detail"]


def test_csv_endpoint_rejects_non_csv_file(client):
    response = client.post(
        "/api/analyze-csv",
        files={"file": ("transactions.txt", "not a CSV", "text/plain")},
    )

    assert response.status_code == 400
    assert "csv" in response.json()["detail"].lower()



def test_ieee_csv_endpoint_scores_rows_and_preserves_global_row_numbers(
    client, mock_artifact
):
    header = "TransactionID,TransactionAmt,identity_available,card1,ProductCD\n"
    rows = [
        f"{100000 + index},{10 + index},0,12345,W"
        for index in range(101)
    ]
    response = client.post(
        "/api/ieee-score-csv",
        files={"file": ("transactions.csv", header + "\n".join(rows), "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["scored_count"] == 101
    assert payload["results"][0]["row"] == 1
    assert payload["results"][99]["row"] == 100
    assert payload["results"][100]["row"] == 101
    assert payload["decision_made"] is False


def test_ieee_csv_endpoint_rejects_duplicate_headers(client):
    content = (
        "TransactionAmt,identity_available,card1,card1,ProductCD\n"
        "125,0,123,456,W\n"
    )
    response = client.post(
        "/api/ieee-score-csv",
        files={"file": ("duplicates.csv", content, "text/csv")},
    )

    assert response.status_code == 422
    assert "duplicate column" in response.json()["detail"].lower()


def test_ieee_csv_endpoint_rejects_missing_identity_metadata(client):
    content = "TransactionAmt,card1,ProductCD\n125,12345,W\n"
    response = client.post(
        "/api/ieee-score-csv",
        files={"file": ("missing-identity.csv", content, "text/csv")},
    )

    assert response.status_code == 422
    assert "identity_available" in response.json()["detail"]



def test_ieee_csv_endpoint_reports_correct_row_after_batch_boundary(
    client, mock_artifact
):
    header = "TransactionAmt,identity_available,card1,ProductCD\n"
    rows = ["10,0,12345,W"] * 100 + ["not-a-number,0,12345,W"]
    response = client.post(
        "/api/ieee-score-csv",
        files={"file": ("bad-row.csv", header + "\n".join(rows), "text/csv")},
    )

    assert response.status_code == 422
    assert "Row 101" in response.json()["detail"]

def test_ieee_csv_endpoint_rejects_rows_with_extra_fields(client):
    content = (
        "TransactionID,TransactionAmt,identity_available\n"
        "999401,125.50,0,EXTRA_FIELD\n"
    )
    response = client.post(
        "/api/ieee-score-csv",
        files={"file": ("malformed.csv", content, "text/csv")},
    )

    assert response.status_code == 422
    assert "row 1" in response.json()["detail"].lower()
    assert "field" in response.json()["detail"].lower()

def test_ieee_csv_endpoint_rejects_unclosed_quotes(client):
    content = (
        "TransactionAmt,identity_available,ProductCD\n"
        '125.50,0,"W\n'
    )
    response = client.post(
        "/api/ieee-score-csv",
        files={"file": ("unclosed_quote.csv", content, "text/csv")},
    )

    assert response.status_code == 422
    assert "Could not parse the uploaded CSV" in response.json()["detail"]


def test_dashboard_audit_logs_persist_with_unique_task_ids(
    tmp_path, monkeypatch
):
    audit_dir = tmp_path / "audit"
    monkeypatch.setenv("FRAUDSHIELD_AUDIT_DIR", str(audit_dir))
    task_ids = []

    class FakeOrchestrator:
        def __init__(self, audit_path):
            self.audit_path = Path(audit_path)

        def run(self, task_id, data):
            task_ids.append(task_id)
            ledger = dashboard.AuditLedger(str(self.audit_path))
            ledger.record(
                agent_id="TEST_AGENT",
                agent_role="Test Agent",
                task_id=task_id,
                action="TEST_AUDIT",
                status="PASS",
                confidence=1.0,
                data={"transaction_count": len(data["transactions"])},
            )
            return {"status": "PASS"}

    monkeypatch.setattr(
        dashboard, "FraudShieldOrchestrator", FakeOrchestrator
    )

    first = dashboard.analyze([{"transaction_id": "txn-1"}])
    second = dashboard.analyze([{"transaction_id": "txn-2"}])

    assert len(task_ids) == 2
    assert task_ids[0] != task_ids[1]
    assert all(task_id.startswith("FRAUDSHIELD-") for task_id in task_ids)

    for task_id, response in zip(task_ids, (first, second)):
        audit_path = audit_dir / f"{task_id}.jsonl"
        assert audit_path.is_file()
        assert dashboard.AuditLedger(str(audit_path)).verify_chain() is True
        assert response["audit_integrity"]["valid"] is True
        assert response["audit_integrity"]["events"] == 1
        assert response["transaction_count"] == 1


def test_rate_limit_returns_429_and_retry_after(client, monkeypatch):
    monkeypatch.setattr(dashboard, "RATE_LIMIT_REQUESTS", 1)
    monkeypatch.setattr(dashboard, "RATE_LIMIT_WINDOW_SECONDS", 60)
    dashboard._rate_limit_buckets.clear()

    first = client.get("/api/demo")
    second = client.get("/api/demo")

    assert first.status_code == 200
    assert second.status_code == 429
    assert int(second.headers["retry-after"]) >= 1

    dashboard._rate_limit_buckets.clear()
