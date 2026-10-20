"""Shapley attribution through a real scikit-learn Pipeline.

Regression tests for the bug where explanations silently failed because
shap.TreeExplainer was handed a Pipeline instead of a tree model.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.explainability import shap_explainer as se
from app.explainability.shap_explainer import explain_batch_with_shap, explain_with_shap

NUMERIC = ["TransactionAmt", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9"]
CATEGORICAL = ["ProductCD", "card4"]
FEATURES = NUMERIC + CATEGORICAL


def _artifact():
    rng = np.random.default_rng(7)
    n = 600
    df = pd.DataFrame({f: rng.normal(size=n) for f in NUMERIC})
    df["TransactionAmt"] = rng.gamma(2.0, 60.0, size=n)
    df["ProductCD"] = rng.choice(["W", "C", "H"], size=n, p=[0.6, 0.2, 0.2])  # W most frequent -> imputed baseline
    df["card4"] = rng.choice(["visa", "mastercard"], size=n)
    y = ((df["TransactionAmt"] > 200) | (df["ProductCD"] == "C")).astype(int)
    pre = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), NUMERIC),
            (
                "categorical",
                __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )
    model = Pipeline(
        [("preprocessing", pre), ("classifier", ExtraTreesClassifier(40, min_samples_leaf=3, random_state=0))]
    )
    model.fit(df[FEATURES], y)
    return {
        "model": model,
        "model_type": "test_pipeline",
        "features": FEATURES,
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
    }


@pytest.fixture(scope="module")
def artifact():
    return _artifact()


def _row(**overrides):
    row = {f: None for f in FEATURES}
    row.update(TransactionAmt=450.0, ProductCD="C", card4="visa")
    row.update(overrides)
    return row


def test_pipeline_explanation_is_additive_and_exact(artifact):
    row = _row()
    exp = explain_with_shap(artifact, row, top_n=10)

    assert exp["explanation_type"] == "shapley_pipeline_attribution"
    assert exp["attribution_method"] == "exact"
    total = sum(d["shap_value"] for d in exp["top_n_drivers"])
    assert total == pytest.approx(exp["model_score"] - exp["baseline_score"], abs=1e-4)

    score = artifact["model"].predict_proba(pd.DataFrame([row], columns=FEATURES))[0, 1]
    assert exp["model_score"] == pytest.approx(score, abs=1e-5)


def test_only_supplied_features_get_drivers(artifact):
    exp = explain_with_shap(artifact, _row(), top_n=10)
    assert {d["feature"] for d in exp["top_n_drivers"]} == {"TransactionAmt", "ProductCD", "card4"}


def test_high_amount_and_product_push_score_up(artifact):
    exp = explain_with_shap(artifact, _row(), top_n=3)
    by_feature = {d["feature"]: d for d in exp["top_n_drivers"]}
    assert by_feature["TransactionAmt"]["direction"] == "increases_risk"
    assert by_feature["ProductCD"]["direction"] == "increases_risk"
    assert exp["top_positive_driver"] is not None


def test_many_features_use_deterministic_permutation_sampling(artifact):
    wide = _row(**{f: 0.3 for f in NUMERIC[1:]})
    first = explain_with_shap(artifact, wide, top_n=12)
    second = explain_with_shap(artifact, wide, top_n=12)

    assert first["attribution_method"] == "permutation_sampling"
    assert first["top_n_drivers"] == second["top_n_drivers"]
    total = sum(d["shap_value"] for d in first["top_n_drivers"])
    assert total == pytest.approx(first["model_score"] - first["baseline_score"], abs=1e-4)


def test_batch_matches_single_row_results(artifact):
    rows = [_row(), _row(TransactionAmt=20.0, ProductCD="W")]
    batch = explain_batch_with_shap(artifact, rows, top_n=5)
    assert len(batch) == 2
    for row, from_batch in zip(rows, batch):
        single = explain_with_shap(artifact, row, top_n=5)
        assert from_batch["top_n_drivers"] == single["top_n_drivers"]


def test_row_with_nothing_supplied_has_no_drivers(artifact):
    exp = explain_with_shap(artifact, _row(TransactionAmt=None, ProductCD=None, card4=None))
    assert exp["top_n_drivers"] == []
    assert exp["model_score"] == pytest.approx(exp["baseline_score"])


REAL_ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "ieee_extratrees_identity_enriched.joblib"


@pytest.mark.skipif(not REAL_ARTIFACT.exists(), reason="trained IEEE artifact not present")
def test_real_ieee_artifact_produces_explanations():
    from app.ml.ieee_scorer import score_transactions

    result = score_transactions(
        [
            {
                "TransactionAmt": 25.0, "card1": 13926, "card2": 360, "ProductCD": "W",
                "card4": "visa", "card6": "debit", "identity_available": 0,
            }
        ],
        explain=True,
    )
    row = result["results"][0]
    assert row["explanation_status"] == "available"
    drivers = row["shap_explanation"]["top_n_drivers"]
    assert drivers and all("plain_description" in d for d in drivers)


def test_score_is_identical_alone_and_in_a_batch(artifact, monkeypatch):
    """Regression: a row's score must not depend on other rows in the batch."""
    from app.ml import ieee_scorer

    art = dict(artifact)
    art.update(
        score_semantics="test",
        features=FEATURES + ["identity_available"],
    )
    # Build a scorer-compatible artifact around the synthetic pipeline.
    rng = np.random.default_rng(3)
    df = pd.DataFrame({f: rng.normal(size=400) for f in NUMERIC})
    df["ProductCD"] = rng.choice(["W", "C"], size=400)
    df["card4"] = rng.choice(["visa", "mastercard"], size=400)
    df["identity_available"] = rng.integers(0, 2, size=400)
    y = ((df["TransactionAmt"] > 0.5) | (df["ProductCD"] == "C")).astype(int)
    feats = FEATURES + ["identity_available"]
    pre = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), NUMERIC + ["identity_available"]),
            (
                "categorical",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore"))]),
                CATEGORICAL,
            ),
        ]
    )
    model = Pipeline([("preprocessing", pre),
                      ("classifier", ExtraTreesClassifier(30, min_samples_leaf=3, random_state=0))])
    model.fit(df[feats], y)
    art.update(model=model, features=feats, numeric_features=NUMERIC + ["identity_available"],
               categorical_features=CATEGORICAL, training_rows=400, training_fraud_rate=0.4,
               target="y", model_type="t")
    monkeypatch.setattr(ieee_scorer, "load_artifact", lambda *a, **k: art)

    row_a = {"TransactionAmt": 25.0, "ProductCD": "W", "identity_available": 0}
    row_b = {"TransactionAmt": 90.0, "ProductCD": "C", "card4": "visa", "identity_available": 1}
    alone = ieee_scorer.score_transactions([row_a], explain=False)["results"][0]["experimental_model_score"]
    batched = ieee_scorer.score_transactions([row_a, row_b], explain=False)["results"][0]["experimental_model_score"]
    assert alone == batched
