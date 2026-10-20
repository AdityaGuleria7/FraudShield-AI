"""Experimental IEEE-CIS scoring service for FraudShield AI."""

import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.ml.artifact_integrity import verify_artifact
from app.ml.frame import build_model_frame


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = (
    PROJECT_ROOT / "artifacts" / "ieee_extratrees_identity_enriched.joblib"
)

ALLOWED_EXTRA_FIELDS = {"TransactionID", "TransactionDT", "identity_available"}
MAX_BATCH_SIZE = 100

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_artifact() -> dict[str, Any]:
    """Load and validate the local model artifact once per process."""
    if not DEFAULT_ARTIFACT.is_file():
        raise FileNotFoundError(
            "IEEE-CIS model artifact is unavailable. "
            "Place the trained artifact in the artifacts directory."
        )

    # Verify bytes before deserializing the trusted local artifact.
    verify_artifact(DEFAULT_ARTIFACT)
    artifact = joblib.load(DEFAULT_ARTIFACT)

    required_metadata = {
        "model",
        "features",
        "numeric_features",
        "categorical_features",
        "score_semantics",
        "threshold",
    }
    if not isinstance(artifact, dict):
        raise ValueError("Model artifact must be a dictionary.")

    missing = required_metadata - artifact.keys()
    if missing:
        raise ValueError(
            f"Model artifact is missing metadata: {sorted(missing)}"
        )

    features = artifact["features"]
    numeric = artifact["numeric_features"]
    categorical = artifact["categorical_features"]

    if (
        not isinstance(features, list)
        or not features
        or len(features) != len(set(features))
        or set(features) != set(numeric) | set(categorical)
        or set(numeric) & set(categorical)
    ):
        raise ValueError("Model artifact feature metadata is invalid.")

    if not callable(getattr(artifact["model"], "predict_proba", None)):
        raise ValueError("Model does not support predict_proba.")

    return artifact


def _is_missing(value: Any) -> bool:
    """Recognize common CSV nulls, including pandas/numpy scalar NaNs."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "na", "nan", "null", "none"}

    # pandas.read_csv represents blank numeric cells as NaN. Treat those as
    # missing so the trained pipeline's imputer can handle them.
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool):
            return missing
        # numpy.bool_ is scalar but isn't a built-in bool.
        if getattr(missing, "ndim", None) == 0:
            return bool(missing)
    except (TypeError, ValueError):
        pass
    return False


def score_transactions(
    rows: list[dict[str, Any]],
    explain: bool = False,
) -> dict[str, Any]:
    """Score compatible feature rows without making fraud decisions."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("Provide a non-empty list of transaction rows.")

    if len(rows) > MAX_BATCH_SIZE:
        raise ValueError(f"A maximum of {MAX_BATCH_SIZE} rows is allowed.")

    artifact = load_artifact()
    features = artifact["features"]
    numeric_features = set(artifact["numeric_features"])
    categorical_features = list(artifact["categorical_features"])
    expected_features = set(features)

    cleaned_rows = []
    input_coverages = []

    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} must be a JSON object.")

        unexpected = set(row) - expected_features - ALLOWED_EXTRA_FIELDS
        if unexpected:
            raise ValueError(
                f"Row {index} has unsupported fields: "
                f"{sorted(unexpected)}"
            )

        if "TransactionAmt" not in row or _is_missing(row["TransactionAmt"]):
            raise ValueError(
                f"Row {index} requires a valid TransactionAmt."
            )

        if "identity_available" not in row:
            raise ValueError(
                f"Row {index} must explicitly provide identity_available "
                "(0 or 1). Do not infer identity availability from missing "
                "feature values."
            )

        availability = row["identity_available"]
        if isinstance(availability, bool):
            availability = int(availability)
        if availability not in (0, 1, 0.0, 1.0, "0", "1"):
            raise ValueError(
                f"Row {index}: identity_available must be 0 or 1."
            )

        clean = {}
        available_features = 0
        missing_features = 0

        for feature in features:
            value = row.get(feature)

            if _is_missing(value):
                missing_features += 1
                clean[feature] = float("nan")
                continue

            available_features += 1
            if feature in numeric_features:
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Row {index}: {feature} must be numeric."
                    ) from exc

                if not math.isfinite(numeric_value):
                    raise ValueError(
                        f"Row {index}: {feature} must be finite."
                    )
                clean[feature] = numeric_value
            else:
                if isinstance(value, (dict, list)):
                    raise ValueError(
                        f"Row {index}: {feature} must be a scalar value."
                    )
                clean[feature] = value

        if clean["TransactionAmt"] < 0:
            raise ValueError(f"Row {index}: TransactionAmt cannot be negative.")

        clean["identity_available"] = int(availability)
        cleaned_rows.append(clean)
        input_coverages.append({
            "available_features": available_features,
            "expected_features": len(features),
            "missing_features": missing_features,
        })

    frame = build_model_frame(cleaned_rows, features, categorical_features)
    scores = artifact["model"].predict_proba(frame)[:, 1]

    shap_explanations = None
    if explain:
        try:
            from app.explainability.shap_explainer import explain_batch_with_shap
            shap_explanations = explain_batch_with_shap(artifact, cleaned_rows)
        except Exception:
            logger.warning(
                "SHAP explanation generation failed; returning scores without explanations",
                exc_info=True,
            )
            shap_explanations = None

    results = []
    for index, (row, score) in enumerate(zip(rows, scores), start=1):
        result = {
            "row": index,
            "experimental_model_score": round(float(score), 6),
            "score_interpretation": "Uncalibrated model score, not a calibrated fraud probability",
            "input_coverage": input_coverages[index - 1],
        }
        if "TransactionID" in row:
            result["TransactionID"] = row["TransactionID"]
        explanation = None
        if explain:
            if shap_explanations and index - 1 < len(shap_explanations):
                explanation = shap_explanations[index - 1]

            if explanation is not None:
                result["shap_explanation"] = explanation
                result["explanation_status"] = "available"
            else:
                result["explanation_status"] = "unavailable"
        else:
            result["explanation_status"] = "not_requested"

        results.append(result)

    return {
        "model_type": artifact.get("model_type", "unknown"),
        "status": "EXPERIMENTAL_NOT_APPROVED",
        "scored_count": len(results),
        "training_rows": artifact.get("training_rows"),
        "training_fraud_rate": artifact.get("training_fraud_rate"),
        "threshold": artifact.get("threshold"),
        "score_semantics": artifact["score_semantics"],
        "validation_note": artifact.get("validation_note"),
        "decision_made": False,
        "results": results,
        "warning": (
            "Scores are experimental and uncalibrated. They do not confirm "
            "fraud, constitute approved ML evidence, or authorize adverse "
            "actions. Use the existing evidence and policy workflow."
        ),
    }
