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

BASE_NUMERIC = ["TransactionAmt"]
BASE_CATEGORICAL = ["card1", "card2", "ProductCD", "card4", "card6"]

IDENTITY_NUMERIC = [
    "id_01", "id_02", "id_05", "id_06", "id_11",
    "id_13", "id_17", "id_19", "id_20",
]
IDENTITY_CATEGORICAL = [
    "id_12", "id_15", "id_16", "id_28", "id_29",
    "id_31", "id_35", "id_36", "id_37", "id_38",
    "DeviceType", "DeviceInfo",
]

TARGET = "isFraud"


def build_model(numeric, categorical):
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=50,
            )),
        ]), categorical),
    ])

    return Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", ExtraTreesClassifier(
            n_estimators=100,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def evaluate(name, model, X_train, y_train, X_valid, y_valid):
    print(f"\nTraining {name}...")
    started = time.time()
    model.fit(X_train, y_train)
    print(f"Training time: {time.time() - started:.1f} seconds")

    scores = model.predict_proba(X_valid)[:, 1]
    ap = average_precision_score(y_valid, scores)
    auc = roc_auc_score(y_valid, scores)

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
            "recall": float(found / y_valid.sum()) if y_valid.sum() else 0,
        }

    result = {
        "model": name,
        "average_precision": float(ap),
        "roc_auc": float(auc),
        "review_capacity": capacity,
    }

    print(f"Average precision: {ap:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print("Review capacity:")
    print(json.dumps(capacity, indent=2))
    return result


def main():
    for path in [TRANSACTION_FILE, IDENTITY_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

    transaction_columns = [
        "TransactionID", "TransactionDT", TARGET,
        *BASE_NUMERIC, *BASE_CATEGORICAL,
    ]
    identity_columns = [
        "TransactionID", *IDENTITY_NUMERIC, *IDENTITY_CATEGORICAL,
    ]

    print("Loading transaction features...")
    transactions = pd.read_csv(
        TRANSACTION_FILE,
        usecols=transaction_columns,
    )

    print("Loading selected identity/device features...")
    identity = pd.read_csv(
        IDENTITY_FILE,
        usecols=identity_columns,
    )

    if transactions["TransactionID"].duplicated().any():
        raise ValueError("Duplicate TransactionID in transaction data.")
    if identity["TransactionID"].duplicated().any():
        raise ValueError("Duplicate TransactionID in identity data.")

    # Left join retains transactions that have no identity record.
    df = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    df["identity_available"] = (
        df["_merge"] == "both"
    ).astype("int8")
    df = df.drop(columns=["_merge"])
    df = df.dropna(subset=["TransactionDT", TARGET])
    df[TARGET] = df[TARGET].astype(int)
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    base_numeric = BASE_NUMERIC
    base_categorical = BASE_CATEGORICAL

    enriched_numeric = BASE_NUMERIC + IDENTITY_NUMERIC + [
        "identity_available"
    ]
    enriched_categorical = BASE_CATEGORICAL + IDENTITY_CATEGORICAL

    # Same chronological 70/15/15 split rule as the prior experiment.
    n = len(df)
    train_boundary = df.loc[int(n * 0.70), "TransactionDT"]
    test_boundary = df.loc[int(n * 0.85), "TransactionDT"]

    train = df[df["TransactionDT"] < train_boundary].copy()
    calibration = df[
        (df["TransactionDT"] >= train_boundary)
        & (df["TransactionDT"] < test_boundary)
    ].copy()
    test = df[df["TransactionDT"] >= test_boundary].copy()

    print("\n=== DATA AND PARTITIONS ===")
    print(f"All rows after join: {len(df):,}")
    print(f"Identity availability: {df['identity_available'].mean():.2%}")
    print(f"Training rows: {len(train):,}")
    print(f"Calibration rows: {len(calibration):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Test fraud cases: {int(test[TARGET].sum()):,}")

    # Drop identifier, time and label from model inputs.
    # They are not passed to either model as features.
    base_features = base_numeric + base_categorical
    enriched_features = (
        enriched_numeric + enriched_categorical
    )

    results = {
        "experiment": "Extra Trees with and without identity enrichment",
        "join": "Left join on TransactionID; one-to-one validation",
        "identity_available_rate": float(
            df["identity_available"].mean()
        ),
        "split": "Chronological 70/15/15",
        "note": (
            "Iterative benchmark: this time period was evaluated in "
            "earlier experiments and is not a pristine untouched test."
        ),
        "models": {},
    }

    for name, numeric, categorical, features in [
        (
            "extra_trees_base",
            base_numeric, base_categorical, base_features,
        ),
        (
            "extra_trees_identity_enriched",
            enriched_numeric, enriched_categorical, enriched_features,
        ),
    ]:
        model = build_model(numeric, categorical)
        result = evaluate(
            name,
            model,
            train[features],
            train[TARGET],
            test[features],
            test[TARGET],
        )
        results["models"][name] = result

    Path("reports").mkdir(exist_ok=True)
    output = Path("reports/ieee_identity_enriched_metrics.json")
    output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved report: {output}")
    print(
        "\nImportant: identity features are optional and missing for many "
        "transactions. These scores are experimental, not production "
        "performance guarantees."
    )


if __name__ == "__main__":
    main()
