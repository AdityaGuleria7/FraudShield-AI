"""Every shipped example CSV must pass the data-quality gate and be useful."""
import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import dashboard
from app.data_quality.validator import DataQualityValidator
from app.fraud.impossible_travel import ImpossibleTravelDetector

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
EXPECTED_PATTERN = {
    "sample_transactions.csv": "VELOCITY_ATTACK",
    "sample_transactions_labelled.csv": "VELOCITY_ATTACK",
    "sample_impossible_travel.csv": "IMPOSSIBLE_TRAVEL",
}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUDSHIELD_AUTH_USERNAME", "example-user")
    monkeypatch.setenv("FRAUDSHIELD_AUTH_PASSWORD", "example-password")
    monkeypatch.setenv("FRAUDSHIELD_AUDIT_DIR", str(tmp_path))
    token = base64.b64encode(b"example-user:example-password").decode()
    return TestClient(dashboard.app, headers={"Authorization": f"Basic {token}"})


def test_every_example_file_is_covered():
    assert {p.name for p in EXAMPLES.glob("*.csv")} == set(EXPECTED_PATTERN)


@pytest.mark.parametrize("name,pattern", sorted(EXPECTED_PATTERN.items()))
def test_example_passes_quality_gate_and_detects_pattern(client, name, pattern):
    with open(EXAMPLES / name, "rb") as fh:
        response = client.post("/api/analyze-csv", files={"file": (name, fh, "text/csv")})

    assert response.status_code == 200
    body = response.json()
    assert body["data_quality"]["status"] == "PASS", body["data_quality"]["issues"][:3]
    assert pattern in {f["pattern"] for f in body["pattern_findings"]}
    assert body["audit_integrity"]["valid"] is True


def test_valid_countries_all_have_travel_coordinates():
    assert DataQualityValidator.VALID_COUNTRIES == frozenset(ImpossibleTravelDetector.COUNTRY_COORDINATES)
