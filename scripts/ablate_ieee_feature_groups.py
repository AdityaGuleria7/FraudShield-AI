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
TX_FILE = DATA_DIR / "train_transaction.csv"
ID_FILE = DATA_DIR / "train_identity.csv"
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
    for path in (TX_FILE, ID_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

    print("Loading transaction data...")
    tx = pd.read_csv(
        TX_FILE,
        usecols=[
            "TransactionID", "TransactionDT", TARGET,
            *BASE_NUMERIC, *BASE_CATEGORICAL,
        ],
    )

    print("Loading identity/device features...")
    identity = pd.read_csv(
        ID_FILE,
        usecols=["TransactionID", *ID_NUMERIC, *ID_CATEGORICAL],
    )

    if tx["TransactionID"].duplicated().any():
        raise ValueError("Duplicate transaction IDs found.")
    if identity["TransactionID"].duplicated().any():
        raise ValueError("Duplicate identity IDs found.")

    df = tx.merge(
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

    # Match the existing chronological 70/15/15 partition.
    n = len(df)
    train_boundary = df.loc[int(n * 0.70), "TransactionDT"]
    test_boundary = df.loc[int(n * 0.85), "TransactionDT"]

    train = df[df["TransactionDT"] < train_boundary].copy()
    test = df[df["TransactionDT"] >= test_boundary].copy()

    print("\n=== SHARED PARTITION ===")
    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Test fraud cases: {int(test[TARGET].sum()):,}")

    base_num = BASE_NUMERIC
    base_cat = BASE_CATEGORICAL
    available_num = BASE_NUMERIC + ["identity_available"]
    numeric_enriched = BASE_NUMERIC + ID_NUMERIC + ["identity_available"]
    categorical_enriched = BASE_CATEGORICAL + ID_CATEGORICAL

    experiments = [
        ("baseline", base_num, base_cat),
        ("numeric_identity_plus_availability", numeric_enriched, base_cat),
        ("categorical_identity_device_plus_availability", available_num, categorical_enriched),
        ("all_identity_device_features", numeric_enriched, categorical_enriched),
    ]

    results = {
        "experiment": "IEEE-CIS identity feature-group ablation",
        "split": "Chronological 70/15/15; first 70% train, final 15% test",
        "note": (
            "Exploratory iterative benchmark; test period has been examined "
            "in earlier experiments and is not an untouched final test. "
            "Feature associations do not establish causation."
        ),
        "models": {},
    }

    for name, numeric, categorical in experiments:
        results["models"][name] = evaluate(
            name, numeric, categorical, train, test
        )

    baseline_ap = results["models"]["baseline"]["average_precision"]
    for result in results["models"].values():
        result["average_precision_change_vs_baseline"] = (
            result["average_precision"] - baseline_ap
        )

    output = Path("reports/ieee_feature_group_ablation.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved report: {output}")


if __name__ == "__main__":
    main()
