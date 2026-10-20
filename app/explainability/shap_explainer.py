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


def _driver_direction(shap_val: float) -> str:
    """Classify a SHAP contribution without treating zero as negative."""
    if shap_val > 0:
        return "increases_risk"
    if shap_val < 0:
        return "decreases_risk"
    return "neutral"


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



# ── Pipeline-aware Shapley attribution ───────────────────────────────────────
# The production artifact is a scikit-learn Pipeline (imputer + one-hot encoder
# + ExtraTrees, ~1,600 transformed columns). shap.TreeExplainer cannot take a
# Pipeline and is unreliable/slow on that tree ensemble, so for pipelines we
# compute Shapley values directly over the *original* input features:
#
#   value(S) = pipeline score when only the features in S are supplied and
#              every other feature is treated as "not provided" (missing).
#
# Contributions therefore sum exactly to  score(row) - score(nothing provided).
# Exact enumeration is used for small inputs; permutation sampling (fixed seed,
# deterministic) is used when many features are supplied.

EXACT_MAX_FEATURES = 8
PERMUTATION_SAMPLES = 24
BASELINE_NOTE = "Baseline = the model score when none of the supplied features are provided."


def _is_pipeline(model: Any) -> bool:
    return hasattr(model, "named_steps") and hasattr(model, "predict_proba")


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _masked_frame(
    rows: list[dict[str, Any]],
    features: list[str],
    categorical: set[str],
    masks: list[tuple[int, frozenset[str]]],
) -> pd.DataFrame:
    from app.ml.frame import build_model_frame

    records = [
        {f: rows[row_idx].get(f) for f in features if f in keep}
        for row_idx, keep in masks
    ]
    return build_model_frame(records, features, categorical)


def _pipeline_shapley(
    artifact: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return per-row {feature: contribution} plus baseline and score."""
    from itertools import combinations
    from math import factorial

    model = artifact["model"]
    features = list(artifact["features"])
    categorical = set(artifact.get("categorical_features", []))

    # Which features did the caller actually supply for each row?
    provided = [
        [f for f in features if not _is_missing_value(r.get(f))] for r in rows
    ]

    # Build every coalition we need, across all rows, then predict once.
    plans: list[tuple[int, list[str], str, list[tuple[int, frozenset[str]]]]] = []
    masks: list[tuple[int, frozenset[str]]] = []
    index_of: dict[tuple[int, frozenset[str]], int] = {}

    def register(row_idx: int, subset: frozenset[str]) -> None:
        key = (row_idx, subset)
        if key not in index_of:
            index_of[key] = len(masks)
            masks.append(key)

    rng = np.random.default_rng(20240601)
    perms_by_row: dict[int, list[list[str]]] = {}
    for row_idx, feats in enumerate(provided):
        k = len(feats)
        register(row_idx, frozenset())
        register(row_idx, frozenset(feats))
        if k == 0:
            continue
        if k <= EXACT_MAX_FEATURES:
            for size in range(k + 1):
                for combo in combinations(feats, size):
                    register(row_idx, frozenset(combo))
        else:
            perms = []
            for _ in range(PERMUTATION_SAMPLES):
                order = list(rng.permutation(feats))
                perms.append(order)
                for pos in range(k + 1):
                    register(row_idx, frozenset(order[:pos]))
            perms_by_row[row_idx] = perms

    frame = _masked_frame(rows, features, categorical, masks)
    probs = model.predict_proba(frame)[:, 1]

    def v(row_idx: int, subset: frozenset[str]) -> float:
        return float(probs[index_of[(row_idx, subset)]])

    out = []
    for row_idx, feats in enumerate(provided):
        k = len(feats)
        contrib = {f: 0.0 for f in feats}
        if k and k <= EXACT_MAX_FEATURES:
            for f in feats:
                others = [x for x in feats if x != f]
                for size in range(len(others) + 1):
                    weight = factorial(size) * factorial(k - size - 1) / factorial(k)
                    for combo in combinations(others, size):
                        base = frozenset(combo)
                        contrib[f] += weight * (v(row_idx, base | {f}) - v(row_idx, base))
        elif k:
            for order in perms_by_row[row_idx]:
                for pos, f in enumerate(order):
                    before = frozenset(order[:pos])
                    contrib[f] += v(row_idx, before | {f}) - v(row_idx, before)
            for f in feats:
                contrib[f] /= len(perms_by_row[row_idx])
        out.append(
            {
                "contributions": contrib,
                "baseline_score": v(row_idx, frozenset()),
                "score": v(row_idx, frozenset(feats)),
                "method": "exact" if k <= EXACT_MAX_FEATURES else "permutation_sampling",
            }
        )
    return out


def _driver_cards(
    features: list[str],
    raw_row: dict[str, Any],
    contributions: dict[str, float],
) -> list[dict[str, Any]]:
    cards = []
    for feature in features:
        if feature not in contributions:
            continue
        value = raw_row.get(feature)
        shap_val = contributions[feature]
        cards.append(
            {
                "feature": feature,
                "label": _label(feature),
                "raw_value": value,
                "shap_value": round(float(shap_val), 6),
                "direction": _driver_direction(shap_val),
                "impact_magnitude": abs(float(shap_val)),
                "plain_description": _plain_driver_description(feature, value, shap_val),
            }
        )
    cards.sort(key=lambda d: d["impact_magnitude"], reverse=True)
    return cards


def _pipeline_explanations(
    artifact: dict[str, Any],
    rows: list[dict[str, Any]],
    top_n: int,
) -> list[dict[str, Any]]:
    features = list(artifact["features"])
    results = []
    for raw_row, res in zip(rows, _pipeline_shapley(artifact, rows)):
        cards = _driver_cards(features, raw_row, res["contributions"])
        top = cards[:top_n]
        positive = [d for d in top if d["direction"] == "increases_risk"]
        negative = [d for d in top if d["direction"] == "decreases_risk"]
        results.append(
            {
                "explanation_type": "shapley_pipeline_attribution",
                "model_type": artifact.get("model_type", "unknown"),
                "attribution_method": res["method"],
                "baseline_score": round(res["baseline_score"], 6),
                "model_score": round(res["score"], 6),
                "top_n_drivers": top,
                "top_positive_driver": positive[0] if positive else None,
                "top_negative_driver": negative[0] if negative else None,
                "driver_count": len(cards),
                "explainability_note": (
                    "Contributions are Shapley values over the supplied input "
                    "features and sum to (model score - baseline score). "
                    + BASELINE_NOTE
                    + " They describe this model's uncalibrated score and do not "
                    "confirm fraud or establish real-world risk."
                ),
            }
        )
    return results


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

    if _is_pipeline(model):
        return _pipeline_explanations(artifact, [row], top_n)[0]

    frame = pd.DataFrame([row], columns=features)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(frame)
    values_matrix = _extract_positive_shap_matrix(shap_values)
    values = values_matrix[0]

    # Build driver cards
    drivers = []
    for feature, value, shap_val in zip(features, frame.iloc[0], values):
        direction = _driver_direction(shap_val)
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
            "SHAP values describe how each feature contributes to this model's "
            "uncalibrated score. They do not confirm fraud or establish real-world risk."
        ),
    }


def _plain_driver_description(
    feature: str,
    value: Any,
    shap_val: float,
) -> str:
    """Generate a one-line plain-language description of a SHAP driver."""
    label = _label(feature)
    if shap_val > 0:
        direction = "pushes the model score higher"
    elif shap_val < 0:
        direction = "pushes the model score lower"
    else:
        return f"{label} does not change the model score direction"
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
    model = artifact["model"]
    features = artifact["features"]

    if _is_pipeline(model):
        return _pipeline_explanations(artifact, rows, top_n)

    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is required: pip install shap") from exc

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
            direction = _driver_direction(shap_val)
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
