from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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

DATA_FILE = (
    Path.home() / "Downloads" / "ieee-fraud-detection"
    / "train_transaction.csv"
)

FEATURES = [
    "TransactionAmt", "card1", "card2",
    "ProductCD", "card4", "card6",
]

LOGISTIC_NUMERIC = ["TransactionAmt", "card1", "card2"]
LOGISTIC_CATEGORICAL = ["ProductCD", "card4", "card6"]

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


def make_models():
    logistic = Pipeline([
        ("preprocessing", make_preprocessor(
            LOGISTIC_NUMERIC, LOGISTIC_CATEGORICAL, scale=True
        )),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    extra_trees = Pipeline([
        ("preprocessing", make_preprocessor(
            TREE_NUMERIC, TREE_CATEGORICAL
        )),
        ("classifier", ExtraTreesClassifier(
            n_estimators=100,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    return {
        "logistic_regression": logistic,
        "extra_trees": extra_trees,
    }


def evaluate(name, model, X, y, scores, threshold):
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    ranked = pd.DataFrame({
        "score": scores,
        "actual": y.to_numpy(),
    }).sort_values("score", ascending=False)

    capacity = {}
    for pct in [1, 2, 5]:
        count = max(1, int(len(ranked) * pct / 100))
        top = ranked.head(count)
        found = int(top["actual"].sum())
        capacity[f"top_{pct}_percent"] = {
            "transactions_reviewed": count,
            "frauds_found": found,
            "precision": float(top["actual"].mean()),
            "recall": float(found / y.sum()) if y.sum() else 0.0,
        }

    return {
        "model": name,
        "selected_threshold": float(threshold),
        "precision_at_selected_threshold": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall_at_selected_threshold": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1_at_selected_threshold": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "flagged_count": int(predictions.sum()),
        "flagged_rate": float(predictions.mean()),
        "confusion_matrix": [
            [int(tn), int(fp)], [int(fn), int(tp)]
        ],
        "review_capacity": capacity,
    }


def main():
    print("Loading selected IEEE-CIS columns...")
    df = pd.read_csv(
        DATA_FILE,
        usecols=FEATURES + ["TransactionDT", "isFraud"],
    )
    df = df.dropna(subset=["TransactionDT", "isFraud"])
    df["isFraud"] = df["isFraud"].astype(int)
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # Use strict time boundaries so identical timestamps cannot straddle splits.
    n = len(df)
    train_boundary = df.loc[int(n * 0.70), "TransactionDT"]
    test_boundary = df.loc[int(n * 0.85), "TransactionDT"]

    train = df[df["TransactionDT"] < train_boundary].copy()
    calibration = df[
        (df["TransactionDT"] >= train_boundary)
        & (df["TransactionDT"] < test_boundary)
    ].copy()
    test = df[df["TransactionDT"] >= test_boundary].copy()

    if train.empty or calibration.empty or test.empty:
        raise RuntimeError("A time-based partition is empty.")

    X_train, y_train = train[FEATURES], train["isFraud"]
    X_cal, y_cal = calibration[FEATURES], calibration["isFraud"]
    X_test, y_test = test[FEATURES], test["isFraud"]

    for label, target in [
        ("training", y_train),
        ("calibration", y_cal),
        ("final test", y_test),
    ]:
        if target.nunique() < 2:
            raise RuntimeError(
                f"The {label} partition contains only one target class."
            )

    print("\n=== THREE CHRONOLOGICAL PERIODS ===")
    print(f"Training:    {len(train):,} rows; fraud rate {y_train.mean():.4%}")
    print(
        f"Calibration: {len(calibration):,} rows; "
        f"fraud rate {y_cal.mean():.4%}"
    )
    print(
        f"Final test:  {len(test):,} rows; "
        f"fraud rate {y_test.mean():.4%}; "
        f"fraud cases {int(y_test.sum()):,}"
    )
    print(f"Training ends before TransactionDT {train_boundary}")
    print(f"Final test starts at TransactionDT {test_boundary}")

    report = {
        "experiment": "IEEE-CIS three-period chronological evaluation",
        "features": FEATURES,
        "partition_rule": (
            "Earlier 70% for training, next 15% for threshold selection, "
            "latest 15% for final testing; strict TransactionDT boundaries"
        ),
        "partitions": {
            "training_rows": int(len(train)),
            "calibration_rows": int(len(calibration)),
            "test_rows": int(len(test)),
            "training_fraud_rate": float(y_train.mean()),
            "calibration_fraud_rate": float(y_cal.mean()),
            "test_fraud_rate": float(y_test.mean()),
            "test_fraud_cases": int(y_test.sum()),
            "training_boundary": float(train_boundary),
            "test_boundary": float(test_boundary),
        },
        "models": {},
    }

    for name, model in make_models().items():
        print(f"\n=== {name.upper()} ===")
        started = time.time()
        model.fit(X_train, y_train)
        print(f"Training time: {time.time() - started:.1f} seconds")

        # Threshold is chosen on calibration data only.
        calibration_scores = model.predict_proba(X_cal)[:, 1]
        threshold = float(np.quantile(calibration_scores, 0.98))

        # The threshold is now fixed before evaluating the final test.
        test_scores = model.predict_proba(X_test)[:, 1]
        result = evaluate(
            name, model, X_test, y_test, test_scores, threshold
        )
        report["models"][name] = result

        print(f"Threshold selected on calibration: {threshold:.6f}")
        print(
            f"Final-test average precision: "
            f"{result['average_precision']:.4f}"
        )
        print(f"Final-test ROC-AUC: {result['roc_auc']:.4f}")
        print(
            f"Precision at selected threshold: "
            f"{result['precision_at_selected_threshold']:.4f}"
        )
        print(
            f"Recall at selected threshold: "
            f"{result['recall_at_selected_threshold']:.4f}"
        )
        print(
            f"F1 at selected threshold: "
            f"{result['f1_at_selected_threshold']:.4f}"
        )
        print(
            f"Flagged on final test: {result['flagged_count']:,} "
            f"({result['flagged_rate']:.2%})"
        )
        print("Confusion matrix [[TN, FP], [FN, TP]]:")
        print(result["confusion_matrix"])
        print("Top-risk review capacity:")
        print(json.dumps(result["review_capacity"], indent=2))

    Path("reports").mkdir(exist_ok=True)
    output = Path("reports/ieee_three_period_validation.json")
    output.write_text(json.dumps(report, indent=2))
    print(f"\nSaved report: {output}")
    print(
        "\nInterpretation: thresholds were chosen using the calibration "
        "period only. The final test was not used for threshold selection."
    )
    print(
        "The 2% calibration quantile targets roughly 2% flagged there; "
        "the flagged percentage may differ on the final test."
    )


if __name__ == "__main__":
    main()
