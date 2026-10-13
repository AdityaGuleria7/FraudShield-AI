from pathlib import Path
import json
import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

DATA_DIR = Path.home() / "Downloads" / "ieee-fraud-detection"
TRANSACTION_FILE = DATA_DIR / "train_transaction.csv"
IDENTITY_FILE = DATA_DIR / "train_identity.csv"
TARGET = "isFraud"

BASE_NUMERIC = ["TransactionAmt"]
BASE_CATEGORICAL = ["card1", "card2", "ProductCD", "card4", "card6"]

ID_NUMERIC = [
    "id_01", "id_02", "id_05", "id_06", "id_11",
    "id_13", "id_17", "id_19", "id_20",
]
ID_CATEGORICAL = [
    "id_12", "id_15", "id_16", "id_28", "id_29",
    "id_31", "id_35", "id_36", "id_37", "id_38",
    "DeviceType", "DeviceInfo",
]


def build_model(numeric, categorical):
    transformers = []

    if numeric:
        transformers.append((
            "numeric",
            SimpleImputer(strategy="median"),
            numeric,
        ))

    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=50,
                )),
            ]),
            categorical,
        ))

    return Pipeline([
        ("preprocessing", ColumnTransformer(transformers)),
        ("classifier", ExtraTreesClassifier(
            n_estimators=100,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def evaluate(name, numeric, categorical, train, test):
    features = numeric + categorical
    model = build_model(numeric, categorical)

    print(f"\nTraining {name}...")
    started = time.time()
    model.fit(train[features], train[TARGET])
    print(f"Training time: {time.time() - started:.1f} seconds")

    scores = model.predict_proba(test[features])[:, 1]
    y = test[TARGET]

    ranked = pd.DataFrame({
        "score": scores,
        "actual": y.to_numpy(),
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
            "recall": float(found / y.sum()) if y.sum() else 0.0,
        }

    result = {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "review_capacity": capacity,
    }

    print(f"Average precision: {result['average_precision']:.4f}")
    print(f"ROC-AUC: {result['roc_auc']:.4f}")
    print("Top-capacity results:")
    print(json.dumps(capacity, indent=2))

    del model
    return result


def main():
    for path in (TRANSACTION_FILE, IDENTITY_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

    print("Loading transaction data...")
    transactions = pd.read_csv(
        TRANSACTION_FILE,
        usecols=[
            "TransactionID", "TransactionDT", TARGET,
            *BASE_NUMERIC, *BASE_CATEGORICAL,
        ],
    )

    print("Loading selected identity/device features...")
    identity = pd.read_csv(
        IDENTITY_FILE,
        usecols=["TransactionID", *ID_NUMERIC, *ID_CATEGORICAL],
    )

    if transactions["TransactionID"].duplicated().any():
        raise ValueError("Duplicate transaction IDs detected.")
    if identity["TransactionID"].duplicated().any():
        raise ValueError("Duplicate identity IDs detected.")

    df = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    df["identity_available"] = (df["_merge"] == "both").astype("int8")
    df.drop(columns="_merge", inplace=True)
    df.dropna(subset=["TransactionDT", TARGET], inplace=True)
    df[TARGET] = df[TARGET].astype(int)
    df.sort_values("TransactionDT", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Match the existing chronological 70/15/15 experiment.
    n = len(df)
    train_boundary = df.loc[int(n * 0.70), "TransactionDT"]
    test_boundary = df.loc[int(n * 0.85), "TransactionDT"]

    train = df[df["TransactionDT"] < train_boundary].copy()
    test = df[df["TransactionDT"] >= test_boundary].copy()

    print("\n=== SHARED DATA SPLIT ===")
    print(f"Rows: {len(df):,}")
    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Test frauds: {int(test[TARGET].sum()):,}")
    print(f"Identity available overall: {df['identity_available'].mean():.2%}")
    print(f"Identity available in training: {train['identity_available'].mean():.2%}")
    print(f"Identity available in test: {test['identity_available'].mean():.2%}")

    # Use a fresh model for each experiment; identical test rows and settings.
    experiments = [
        (
            "baseline",
            BASE_NUMERIC,
            BASE_CATEGORICAL,
        ),
        (
            "baseline_plus_identity_availability",
            BASE_NUMERIC + ["identity_available"],
            BASE_CATEGORICAL,
        ),
        (
            "identity_device_enriched",
            BASE_NUMERIC + ID_NUMERIC + ["identity_available"],
            BASE_CATEGORICAL + ID_CATEGORICAL,
        ),
    ]

    results = {
        "experiment": "Identity/device feature ablation",
        "split": "Chronological 70/15/15; train on first 70%, evaluate final 15%",
        "note": (
            "Exploratory iterative benchmark, not an untouched test. "
            "Observational associations do not establish causation."
        ),
        "identity_available_rate": float(df["identity_available"].mean()),
        "models": {},
    }

    for name, numeric, categorical in experiments:
        results["models"][name] = evaluate(
            name, numeric, categorical, train, test
        )

    baseline_ap = results["models"]["baseline"]["average_precision"]
    for name, result in results["models"].items():
        result["average_precision_change_vs_baseline"] = (
            result["average_precision"] - baseline_ap
        )

    output = Path("reports/ieee_identity_ablation_metrics.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved report: {output}")


if __name__ == "__main__":
    main()
