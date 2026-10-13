from pathlib import Path
import json
import time

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

DATA_FILE = (
    Path.home() / "Downloads" / "ieee-fraud-detection"
    / "train_transaction.csv"
)
FEATURES = [
    "TransactionAmt", "card1", "card2",
    "ProductCD", "card4", "card6",
]
NUMERIC = ["TransactionAmt"]
CATEGORICAL = ["card1", "card2", "ProductCD", "card4", "card6"]


def main():
    print("Loading selected IEEE-CIS columns...")
    df = pd.read_csv(DATA_FILE, usecols=FEATURES + ["isFraud"])
    df = df.dropna(subset=["isFraud"])
    y = df["isFraud"].astype(int)
    X = df[FEATURES]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=50,
        )),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", ExtraTreesClassifier(
            n_estimators=100,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print(
        f"Training Extra Trees on {len(X_train):,} rows; "
        f"validating on {len(X_valid):,} rows..."
    )
    start = time.time()
    model.fit(X_train, y_train)
    print(f"Training time: {time.time() - start:.1f} seconds")

    scores = model.predict_proba(X_valid)[:, 1]
    predictions = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_valid, predictions, labels=[0, 1]
    ).ravel()

    results = {
        "model": "ExtraTreesClassifier",
        "features": FEATURES,
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "precision_at_0_5": float(
            precision_score(y_valid, predictions, zero_division=0)
        ),
        "recall_at_0_5": float(
            recall_score(y_valid, predictions, zero_division=0)
        ),
        "f1_at_0_5": float(
            f1_score(y_valid, predictions, zero_division=0)
        ),
        "average_precision": float(
            average_precision_score(y_valid, scores)
        ),
        "roc_auc": float(roc_auc_score(y_valid, scores)),
        "confusion_matrix_at_0_5": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "review_capacity": {},
    }

    # Capacity-based review: rank transactions by score.
    ranked = pd.DataFrame({
        "score": scores,
        "actual": y_valid.to_numpy(),
    }).sort_values("score", ascending=False)

    for pct in [1, 2, 5]:
        n_review = max(1, int(len(ranked) * pct / 100))
        top = ranked.head(n_review)
        results["review_capacity"][f"top_{pct}_percent"] = {
            "transactions_reviewed": n_review,
            "frauds_found": int(top["actual"].sum()),
            "precision": float(top["actual"].mean()),
            "recall": float(top["actual"].sum() / y_valid.sum()),
        }

    Path("artifacts").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    joblib.dump(model, "artifacts/ieee_extratrees_baseline.joblib")
    Path("reports/ieee_extratrees_metrics.json").write_text(
        json.dumps(results, indent=2)
    )

    print("\n=== EXTRA TREES RESULTS ===")
    print(f"Precision at 0.5: {results['precision_at_0_5']:.4f}")
    print(f"Recall at 0.5:    {results['recall_at_0_5']:.4f}")
    print(f"F1 at 0.5:        {results['f1_at_0_5']:.4f}")
    print(f"Average precision:{results['average_precision']:.4f}")
    print(f"ROC-AUC:          {results['roc_auc']:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(results["confusion_matrix_at_0_5"])

    print("\n=== TOP-RISK REVIEW CAPACITY ===")
    for name, values in results["review_capacity"].items():
        print(
            f"{name}: review {values['transactions_reviewed']:,}; "
            f"frauds found {values['frauds_found']:,}; "
            f"precision {values['precision']:.4f}; "
            f"recall {values['recall']:.4f}"
        )

    print("\nSaved model: artifacts/ieee_extratrees_baseline.joblib")
    print("Saved report: reports/ieee_extratrees_metrics.json")


if __name__ == "__main__":
    main()
