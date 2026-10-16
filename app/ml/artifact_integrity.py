"""Integrity checks for locally trusted, serialized model artifacts."""

from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path

# Pin for the currently inspected, locally trusted experimental artifact.
DEFAULT_MODEL_SHA256 = (
    "3b6bfe977214c9ba75433d5871c430c49792a02e30518c3a792a71a08cee6ce7"
)
MODEL_SHA256_ENV = "FRAUDSHIELD_MODEL_SHA256"


class ArtifactIntegrityError(RuntimeError):
    """Raised when a model artifact cannot be verified."""


def expected_sha256() -> str:
    """Return a configured SHA-256 pin, falling back to the checked-in default."""
    value = os.environ.get(MODEL_SHA256_ENV, DEFAULT_MODEL_SHA256).strip().lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ArtifactIntegrityError(
            f"{MODEL_SHA256_ENV} must contain exactly 64 hexadecimal characters."
        )
    return value


def sha256_file(path: Path) -> str:
    """Hash a file incrementally without loading it into memory."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactIntegrityError("Model artifact cannot be read.") from exc
    return digest.hexdigest()


def verify_artifact(path: Path, expected: str | None = None) -> str:
    """Verify an artifact's digest before any pickle/joblib deserialization."""
    expected = expected.lower() if expected is not None else expected_sha256()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ArtifactIntegrityError("Expected model SHA-256 is invalid.")

    actual = sha256_file(path)
    if not hmac.compare_digest(actual, expected):
        raise ArtifactIntegrityError(
            "Model artifact SHA-256 mismatch; refusing to load the artifact."
        )
    return actual
