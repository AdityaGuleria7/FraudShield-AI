from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

DATA_FILE = (
    Path.home() / "Downloads" / "ieee-fraud-detection"
    / "train_transaction.csv"
)
# Set FRAUDSHIELD_CHALLENGER_OUTPUT to a new path for each experiment.
# Existing reports are never overwritten by default.
import os

OUTPUT_FILE = Path(
    os.environ.get(
        "FRAUDSHIELD_CHALLENGER_OUTPUT",
        "reports/ieee_challenger_comparison.json",
    )
)

FEATURES = [
    "TransactionAmt", "card1", "card2",
    "ProductCD", "card4", "card6",
]
NUMERIC = ["TransactionAmt", "card1", "card2"]
CATEGORICAL = ["ProductCD", "card4", "card6"]
TREE_NUMERIC = ["TransactionAmt"]
TREE_CATEGORICAL = ["card1", "card2", "ProductCD", "card4", "card6"]


def make_preprocessor(numeric, categorical, scale=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=50,
            )),
        ]), categorical),
    ])


def make_models(y_train):
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    if positives == 0:
        raise RuntimeError("Training partition has no fraud cases.")
    scale_pos_weight = negatives / positives

    logistic = Pipeline([
        ("preprocessing", make_preprocessor(NUMERIC, CATEGORICAL, scale=True)),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=1000,
            random_state=42, solver="lbfgs",
        )),
    ])

    extra_trees = Pipeline([
        ("preprocessing", make_preprocessor(
            TREE_NUMERIC, TREE_CATEGORICAL
        )),
        ("classifier", ExtraTreesClassifier(
            n_estimators=100, min_samples_leaf=5,
            max_features="sqrt", class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])

    xgboost = Pipeline([
        ("preprocessing", make_preprocessor(
            TREE_NUMERIC, TREE_CATEGORICAL
        )),
        ("classifier", XGBClassifier(
            n_estimators=250, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="aucpr",
            tree_method="hist", scale_pos_weight=scale_pos_weight,
            random_state=42, n_jobs=4,
        )),
    ])

    lightgbm = Pipeline([
        ("preprocessing", make_preprocessor(
            TREE_NUMERIC, TREE_CATEGORICAL
        )),
        ("classifier", LGBMClassifier(
            n_estimators=250, max_depth=-1, num_leaves=31,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0,
            objective="binary", class_weight="balanced",
            random_state=42, n_jobs=4, verbosity=-1,
        )),
    ])

    return {
        "logistic_regression": logistic,
        "extra_trees": extra_trees,
        "xgboost": xgboost,
        "lightgbm": lightgbm,
    }


def evaluate(name, y, scores, threshold):
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    ranked = pd.DataFrame({
        "score": np.asarray(scores),
        "actual": np.asarray(y),
    }).sort_values("score", ascending=False)

    capacity = {}
    for pct in (1, 2, 5):
        count = max(1, int(len(ranked) * pct / 100))
        top = ranked.head(count)
        found = int(top["actual"].sum())
        capacity[f"top_{pct}_percent"] = {
            "transactions_reviewed": count,
            "frauds_found": found,
            "precision": float(top["actual"].mean()),
            "recall": float(found / np.asarray(y).sum())
            if np.asarray(y).sum() else 0.0,
        }

    return {
        "model": name,
        "selected_threshold": float(threshold),
        "threshold_rule": "98th percentile of calibration scores",
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "flagged_count": int(predictions.sum()),
        "flagged_rate": float(predictions.mean()),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "review_capacity": capacity,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def require_clean_working_tree():
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=True,
    )
    if result.stdout.strip():
        raise RuntimeError(
            "Working tree is not clean. Commit or remove changes before "
            "running the experiment so source_revision identifies the code."
        )


def dependency_versions():
    packages = ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def main():
    # Fail before loading data or training if the report already exists.
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Report already exists: {OUTPUT_FILE}. "
            "Choose a new path with FRAUDSHIELD_CHALLENGER_OUTPUT."
        )
    require_clean_working_tree()
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}. Update DATA_FILE first."
        )

    print("Loading selected IEEE-CIS columns...")
    df = pd.read_csv(
        DATA_FILE, usecols=FEATURES + ["TransactionDT", "isFraud"]
    )
    df = df.dropna(subset=["TransactionDT", "isFraud"])
    df["isFraud"] = df["isFraud"].astype(int)
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # Strict chronological boundaries: tied timestamps stay in one period.
    n = len(df)
    train_boundary = df.loc[int(n * 0.70), "TransactionDT"]
    test_boundary = df.loc[int(n * 0.85), "TransactionDT"]

    train = df[df["TransactionDT"] < train_boundary]
    calibration = df[
        (df["TransactionDT"] >= train_boundary)
        & (df["TransactionDT"] < test_boundary)
    ]
    test = df[df["TransactionDT"] >= test_boundary]

    if any(part.empty for part in (train, calibration, test)):
        raise RuntimeError("A chronological partition is empty.")

    X_train, y_train = train[FEATURES], train["isFraud"]
    X_cal, y_cal = calibration[FEATURES], calibration["isFraud"]
    X_test, y_test = test[FEATURES], test["isFraud"]

    for label, target in (
        ("training", y_train),
        ("calibration", y_cal),
        ("test", y_test),
    ):
        if target.nunique() < 2:
            raise RuntimeError(f"{label} partition has only one target class.")

    print("\n=== CHRONOLOGICAL PARTITIONS ===")
    for label, part in (
        ("Training", train),
        ("Calibration", calibration),
        ("Test", test),
    ):
        print(
            f"{label}: {len(part):,} rows; "
            f"fraud rate {part['isFraud'].mean():.4%}; "
            f"fraud cases {int(part['isFraud'].sum()):,}"
        )

    report = {
        "evaluation_status": "exploratory_reused_test_period_not_independent_validation",
        "test_period_is_untouched": False,
        "dataset": {
            "filename": DATA_FILE.name,
            "sha256": sha256_file(DATA_FILE),
            "rows_after_required_field_filter": int(len(df)),
        },
        "environment": {
            "python": sys.version.split()[0],
            "dependencies": dependency_versions(),
        },
        "source_revision": git_revision(),
        "experiment": "IEEE-CIS chronological four-model comparison",
        "features": FEATURES,
        "partition_rule": (
            "Earlier 70% train, next 15% calibration, latest 15% test; "
            "strict TransactionDT boundaries"
        ),
        "threshold_rule": (
            "Per-model 98th percentile of calibration scores; "
            "thresholds fixed before test evaluation"
        ),
        "limitations": [
            "Only six selected features are used.",
            "The dataset period may have been used in earlier experiments; "
            "the latest 15% is not claimed to be a pristine holdout.",
            "A percentile threshold is a ranking rule, not a cost-optimized "
            "business decision threshold.",
            "Metrics on this benchmark do not establish production readiness.",
        ],
        "partitions": {
            "training_rows": int(len(train)),
            "calibration_rows": int(len(calibration)),
            "test_rows": int(len(test)),
            "training_boundary": float(train_boundary),
            "test_boundary": float(test_boundary),
            "test_fraud_cases": int(y_test.sum()),
        },
        "models": {},
    }

    for name, model in make_models(y_train).items():
        print(f"\n=== {name.upper()} ===")
        started = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            model.fit(X_train, y_train)
        print(f"Training time: {time.time() - started:.1f} seconds")

        # Calibrate the threshold on calibration data, never on test labels.
        calibration_scores = model.predict_proba(X_cal)[:, 1]
        threshold = float(np.quantile(calibration_scores, 0.98))
        test_scores = model.predict_proba(X_test)[:, 1]

        result = evaluate(name, y_test, test_scores, threshold)
        report["models"][name] = result

        for metric in (
            "average_precision", "roc_auc", "precision",
            "recall", "f1", "flagged_rate",
        ):
            print(f"{metric}: {result[metric]:.4f}")
        print(f"Threshold: {threshold:.6f}")
        print(f"Flagged transactions: {result['flagged_count']:,}")
        print("Review capacity:", json.dumps(result["review_capacity"]))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of historical results.
    with OUTPUT_FILE.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(f"\nSaved report: {OUTPUT_FILE}")
    print("Compare average precision and review-capacity results first.")
    print("These are experimental benchmark results, not production approval.")


if __name__ == "__main__":
    main()
