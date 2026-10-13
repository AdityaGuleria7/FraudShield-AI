from pathlib import Path
import json
import time

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_FILE = Path.home() / "Downloads" / "ieee-fraud-detection" / "train_transaction.csv"
ARTIFACT_DIR = Path("artifacts")
REPORT_DIR = Path("reports")

NUMERIC_FEATURES = ["TransactionAmt", "card1", "card2"]
CATEGORICAL_FEATURES = ["ProductCD", "card4", "card6"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "isFraud"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    ARTIFACT_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    print("Loading selected IEEE-CIS columns...")
    df = pd.read_csv(DATA_FILE, usecols=FEATURES + [TARGET])
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    print(f"Training on {len(X_train):,} rows; validating on {len(X_valid):,} rows...")
    started = time.time()
    model.fit(X_train, y_train)
    print(f"Training completed in {time.time() - started:.1f} seconds.")

    predictions = model.predict(X_valid)
    probabilities = model.predict_proba(X_valid)[:, 1]

    metrics = {
        "dataset": "IEEE-CIS Fraud Detection",
        "model": "LogisticRegression",
        "features": FEATURES,
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "fraud_rate_validation_pct": round(float(y_valid.mean() * 100), 4),
        "precision": float(precision_score(y_valid, predictions, zero_division=0)),
        "recall": float(recall_score(y_valid, predictions, zero_division=0)),
        "f1": float(f1_score(y_valid, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y_valid, probabilities)),
        "roc_auc": float(roc_auc_score(y_valid, probabilities)),
        "confusion_matrix": confusion_matrix(y_valid, predictions).tolist(),
        "classification_report": classification_report(
            y_valid, predictions, output_dict=True, zero_division=0
        ),
    }

    joblib.dump(model, ARTIFACT_DIR / "ieee_logistic_baseline.joblib")
    (REPORT_DIR / "ieee_baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    print("\n=== VALIDATION RESULTS ===")
    print(f"Precision:        {metrics['precision']:.4f}")
    print(f"Recall:           {metrics['recall']:.4f}")
    print(f"F1:               {metrics['f1']:.4f}")
    print(f"Average precision:{metrics['average_precision']:.4f}")
    print(f"ROC-AUC:          {metrics['roc_auc']:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(metrics["confusion_matrix"])
    print("\nSaved model: artifacts/ieee_logistic_baseline.joblib")
    print("Saved report: reports/ieee_baseline_metrics.json")
    print("\nNote: This is a baseline, not a production-ready fraud detector.")


if __name__ == "__main__":
    main()
