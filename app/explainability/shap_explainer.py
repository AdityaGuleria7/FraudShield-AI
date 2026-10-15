"""
SHAP-based explainability for the IEEE-CIS scoring model.

Provides per-transaction feature attribution — translating model internals
into ranked risk drivers that analysts can understand and audit.

SHAP (SHapley Additive exPlanations) is the industry standard for explainable
AI in regulated fraud detection environments.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


# ── Human-readable feature name mapping ──────────────────────────────────────
# Maps raw IEEE-CIS feature names → analyst-friendly labels.

FEATURE_LABELS: dict[str, str] = {
    "TransactionAmt": "Transaction Amount",
    "card1": "Card Fingerprint (card1)",
    "card2": "Card Group Code (card2)",
    "card4": "Card Network (card4)",
    "card6": "Card Type (card6)",
    "ProductCD": "Product Category",
    "identity_available": "Identity Data Available",
    "TransactionDT": "Transaction Timestamp Offset",
    # Add more as feature set grows
}


def _label(feature: str) -> str:
    """Return a human-readable label for a feature name."""
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


import numpy as np


def _extract_positive_shap_matrix(shap_values: Any) -> np.ndarray:
    """Extract a 2D matrix of shape (n_samples, n_features) for the positive class."""
    if isinstance(shap_values, list):
        matrix = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif hasattr(shap_values, "ndim"):
        if shap_values.ndim == 3:
            matrix = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
        elif shap_values.ndim == 2:
            matrix = shap_values
        elif shap_values.ndim == 1:
            matrix = shap_values.reshape(1, -1)
        else:
            matrix = np.asarray(shap_values)
    else:
        matrix = np.asarray(shap_values)
    return np.asarray(matrix, dtype=float)


def explain_with_shap(
    artifact: dict[str, Any],
    row: dict[str, Any],
    top_n: int = 5,
) -> dict[str, Any]:
    """
    Compute SHAP values for a single transaction row and return structured
    risk-driver cards ready for display in the dashboard.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "shap is required for model explainability. "
            "Install it with: pip install shap"
        ) from exc

    model = artifact["model"]
    features = artifact["features"]

    frame = pd.DataFrame([row], columns=features)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(frame)
    values_matrix = _extract_positive_shap_matrix(shap_values)
    values = values_matrix[0]

    # Build driver cards
    drivers = []
    for feature, value, shap_val in zip(features, frame.iloc[0], values):
        direction = "increases_risk" if shap_val > 0 else "decreases_risk"
        drivers.append(
            {
                "feature": feature,
                "label": _label(feature),
                "raw_value": value,
                "shap_value": round(float(shap_val), 6),
                "direction": direction,
                "impact_magnitude": abs(float(shap_val)),
                "plain_description": _plain_driver_description(
                    feature, value, shap_val
                ),
            }
        )

    # Sort by absolute SHAP value (most impactful first)
    drivers.sort(key=lambda d: d["impact_magnitude"], reverse=True)

    top_drivers = drivers[:top_n]
    positive_drivers = [d for d in top_drivers if d["direction"] == "increases_risk"]
    negative_drivers = [d for d in top_drivers if d["direction"] == "decreases_risk"]

    return {
        "explanation_type": "shap_tree_explainer",
        "model_type": artifact.get("model_type", "unknown"),
        "top_n_drivers": top_drivers,
        "top_positive_driver": positive_drivers[0] if positive_drivers else None,
        "top_negative_driver": negative_drivers[0] if negative_drivers else None,
        "driver_count": len(drivers),
        "explainability_note": (
            "SHAP values show how each feature pushes the model score higher "
            "(toward fraud) or lower (toward legitimate). They do not confirm fraud."
        ),
    }


def _plain_driver_description(
    feature: str,
    value: Any,
    shap_val: float,
) -> str:
    """Generate a one-line plain-language description of a SHAP driver."""
    label = _label(feature)
    direction = "increases fraud risk" if shap_val > 0 else "reduces fraud risk"
    magnitude = abs(shap_val)

    if magnitude > 0.1:
        strength = "strongly"
    elif magnitude > 0.03:
        strength = "moderately"
    else:
        strength = "slightly"

    # Feature-specific descriptions where we can add domain context
    if feature == "TransactionAmt":
        try:
            amt = float(value)
            return f"Transaction amount ${amt:,.2f} {strength} {direction}"
        except (TypeError, ValueError):
            pass
    elif feature == "identity_available":
        avail = "available" if value in (1, 1.0, "1") else "unavailable"
        return f"Identity data {avail} — {strength} {direction}"

    return f"{label} = {value!r} — {strength} {direction}"


def explain_batch_with_shap(
    artifact: dict[str, Any],
    rows: list[dict[str, Any]],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """
    Compute SHAP explanations for multiple transactions.

    Batches the computation for efficiency.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is required: pip install shap") from exc

    model = artifact["model"]
    features = artifact["features"]

    frame = pd.DataFrame(rows, columns=features)
    explainer = shap.TreeExplainer(model)
    shap_values_all = explainer.shap_values(frame)

    # Positive class SHAP values for all rows
    values_matrix = _extract_positive_shap_matrix(shap_values_all)

    results = []
    for row_idx, (row, feature_row) in enumerate(zip(rows, frame.to_dict("records"))):
        shap_row = values_matrix[row_idx]
        drivers = []

        for feature, shap_val in zip(features, shap_row):
            raw_val = feature_row.get(feature)
            direction = "increases_risk" if shap_val > 0 else "decreases_risk"
            drivers.append(
                {
                    "feature": feature,
                    "label": _label(feature),
                    "raw_value": raw_val,
                    "shap_value": round(float(shap_val), 6),
                    "direction": direction,
                    "impact_magnitude": abs(float(shap_val)),
                    "plain_description": _plain_driver_description(
                        feature, raw_val, shap_val
                    ),
                }
            )

        drivers.sort(key=lambda d: d["impact_magnitude"], reverse=True)
        top_drivers = drivers[:top_n]

        positive = [d for d in top_drivers if d["direction"] == "increases_risk"]
        negative = [d for d in top_drivers if d["direction"] == "decreases_risk"]

        results.append(
            {
                "explanation_type": "shap_tree_explainer",
                "top_n_drivers": top_drivers,
                "top_positive_driver": positive[0] if positive else None,
                "top_negative_driver": negative[0] if negative else None,
            }
        )

    return results
