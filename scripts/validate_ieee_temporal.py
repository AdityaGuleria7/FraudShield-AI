from pathlib import Path
import json
import time

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

NUMERIC_LOGISTIC = ["TransactionAmt", "card1", "card2"]
CATEGORICAL_LOGISTIC = ["ProductCD", "card4", "card6"]
NUMERIC_TREE = ["TransactionAmt"]
CATEGORICAL_TREE = ["card1", "card2", "ProductCD", "card4", "card6"]


def make_preprocessor(numeric, categorical, scale_numeric=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
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


def evaluate(name, model, X_train, y_train, X_valid, y_valid):
    print(f"\nTraining {name}...")
    started = time.time()
    model.fit(X_train, y_train)
    print(f"Training time: {time.time() - started:.1f} seconds")

    scores = model.predict_proba(X_valid)[:, 1]
    predictions = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_valid, predictions, labels=[0, 1]
    ).ravel()

    ranked = pd.DataFrame({
        "score": scores,
        "actual": y_valid.to_numpy(),
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
            "recall": float(found / y_valid.sum()) if y_valid.sum() else 0.0,
        }

    result = {
        "model": name,
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
        "confusion_matrix_at_0_5": [
            [int(tn), int(fp)], [int(fn), int(tp)]
        ],
        "review_capacity": capacity,
    }

    print(f"Precision at 0.5: {result['precision_at_0_5']:.4f}")
    print(f"Recall at 0.5:    {result['recall_at_0_5']:.4f}")
    print(f"F1 at 0.5:        {result['f1_at_0_5']:.4f}")
    print(f"Average precision:{result['average_precision']:.4f}")
    print(f"ROC-AUC:          {result['roc_auc']:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(result["confusion_matrix_at_0_5"])
    print("Review capacity:")
    print(json.dumps(capacity, indent=2))
    return result


def main():
    columns = [
        "TransactionDT", "TransactionAmt", "card1", "card2",
        "ProductCD", "card4", "card6", "isFraud",
    ]
    print("Loading selected columns...")
    df = pd.read_csv(DATA_FILE, usecols=columns)
    df = df.dropna(subset=["TransactionDT", "isFraud"])
    df["isFraud"] = df["isFraud"].astype(int)
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # Use a chronological boundary, not a random split.
    split_index = int(len(df) * 0.80)
    cutoff = df.loc[split_index, "TransactionDT"]
    train = df[df["TransactionDT"] < cutoff].copy()
    valid = df[df["TransactionDT"] >= cutoff].copy()

    if train.empty or valid.empty:
        raise RuntimeError("Temporal split produced an empty partition.")

    features = [
        "TransactionAmt", "card1", "card2",
        "ProductCD", "card4", "card6",
    ]
    X_train, y_train = train[features], train["isFraud"]
    X_valid, y_valid = valid[features], valid["isFraud"]

    print("\n=== CHRONOLOGICAL SPLIT ===")
    print(f"Boundary TransactionDT: {cutoff}")
    print(f"Training rows:          {len(train):,}")
    print(f"Validation rows:        {len(valid):,}")
    print(f"Training fraud rate:    {y_train.mean():.4%}")
    print(f"Validation fraud rate:  {y_valid.mean():.4%}")
    print(f"Validation fraud cases: {int(y_valid.sum()):,}")

    if y_valid.nunique() < 2:
        raise RuntimeError(
            "Temporal validation has only one target class; "
            "ROC-AUC cannot be computed."
        )

    logistic = Pipeline([
        ("preprocessing", make_preprocessor(
            NUMERIC_LOGISTIC, CATEGORICAL_LOGISTIC,
            scale_numeric=True,
        )),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    extra_trees = Pipeline([
        ("preprocessing", make_preprocessor(
            NUMERIC_TREE, CATEGORICAL_TREE,
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

    results = {
        "experiment": "IEEE-CIS chronological 80/20 holdout",
        "split_method": "TransactionDT ascending; earlier rows train, later rows validate",
        "boundary_transaction_dt": float(cutoff),
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "train_fraud_rate": float(y_train.mean()),
        "validation_fraud_rate": float(y_valid.mean()),
        "models": {},
    }

    results["models"]["logistic_regression"] = evaluate(
        "Logistic Regression", logistic,
        X_train, y_train, X_valid, y_valid,
    )
    results["models"]["extra_trees"] = evaluate(
        "Extra Trees", extra_trees,
        X_train, y_train, X_valid, y_valid,
    )

    Path("reports").mkdir(exist_ok=True)
    output = Path("reports/ieee_temporal_validation.json")
    output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved temporal validation report: {output}")
    print("\nCompare average precision first, then review-capacity recall.")
    print("This is a temporal holdout result, not a production guarantee.")


if __name__ == "__main__":
    main()
