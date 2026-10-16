import hashlib

import pytest

from app.ml.artifact_integrity import (
    ArtifactIntegrityError,
    expected_sha256,
    verify_artifact,
)


def test_verify_artifact_accepts_matching_hash(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"trusted test artifact")
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()

    assert verify_artifact(artifact, expected) == expected


def test_verify_artifact_rejects_tampered_file(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"changed artifact")
    wrong_hash = hashlib.sha256(b"original artifact").hexdigest()

    with pytest.raises(ArtifactIntegrityError, match="mismatch"):
        verify_artifact(artifact, wrong_hash)


def test_verify_artifact_rejects_missing_file(tmp_path):
    with pytest.raises(ArtifactIntegrityError, match="cannot be read"):
        verify_artifact(tmp_path / "missing.bin", "0" * 64)


def test_expected_hash_rejects_invalid_environment_value(monkeypatch):
    monkeypatch.setenv("FRAUDSHIELD_MODEL_SHA256", "not-a-hash")

    with pytest.raises(ArtifactIntegrityError, match="64 hexadecimal"):
        expected_sha256()


def test_scorer_rejects_bad_hash_before_joblib_load(monkeypatch, tmp_path):
    from app.ml import ieee_scorer

    artifact_path = tmp_path / "untrusted.joblib"
    artifact_path.write_bytes(b"not a trusted model")

    monkeypatch.setattr(ieee_scorer, "DEFAULT_ARTIFACT", artifact_path)
    monkeypatch.setattr(
        ieee_scorer.joblib,
        "load",
        lambda *_args, **_kwargs: pytest.fail("joblib.load must not be called"),
    )

    with pytest.raises(ArtifactIntegrityError, match="mismatch"):
        ieee_scorer.load_artifact.__wrapped__()
