from pathlib import Path
import json
import time

import joblib
import pandas as pd

from scripts.train_ieee_identity_enriched import (
    BASE_NUMERIC,
    BASE_CATEGORICAL,
    IDENTITY_NUMERIC,
    IDENTITY_CATEGORICAL,
    TARGET,
    build_model,
)

DATA_DIR = Path.home() / "Downloads" / "ieee-fraud-detection"
TRANSACTION_FILE = DATA_DIR / "train_transaction.csv"
IDENTITY_FILE = DATA_DIR / "train_identity.csv"

OUTPUT = Path("artifacts/ieee_extratrees_identity_enriched.joblib")

def main():
    for path in (TRANSACTION_FILE, IDENTITY_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

    transaction_columns = [
        "TransactionID", "TransactionDT", TARGET,
        *BASE_NUMERIC, *BASE_CATEGORICAL,
    ]
    identity_columns = [
        "TransactionID", *IDENTITY_NUMERIC, *IDENTITY_CATEGORICAL,
    ]

    print("Loading selected transaction features...")
    transactions = pd.read_csv(
        TRANSACTION_FILE, usecols=transaction_columns
    )
    print("Loading selected identity/device features...")
    identity = pd.read_csv(
        IDENTITY_FILE, usecols=identity_columns
    )

    if transactions["TransactionID"].duplicated().any():
        raise ValueError("Duplicate transaction IDs found.")
    if identity["TransactionID"].duplicated().any():
        raise ValueError("Duplicate identity IDs found.")

    df = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    df["identity_available"] = (df["_merge"] == "both").astype("int8")
    df = df.drop(columns=["_merge"])
    df = df.dropna(subset=["TransactionDT", TARGET])
    df[TARGET] = df[TARGET].astype(int)
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    numeric = BASE_NUMERIC + IDENTITY_NUMERIC + ["identity_available"]
    categorical = BASE_CATEGORICAL + IDENTITY_CATEGORICAL
    features = numeric + categorical

    # Train on the first 70% chronologically, matching the prior experiment.
    boundary = df.loc[int(len(df) * 0.70), "TransactionDT"]
    train = df[df["TransactionDT"] < boundary].copy()

    if train[TARGET].nunique() != 2:
        raise ValueError("Training partition must contain both target classes.")

    print(f"Training rows: {len(train):,}")
    print(f"Training fraud rate: {train[TARGET].mean():.4%}")
    print(f"Identity available: {train['identity_available'].mean():.2%}")

    model = build_model(numeric, categorical)
    started = time.time()
    model.fit(train[features], train[TARGET])
    print(f"Training completed in {time.time() - started:.1f} seconds.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "model_type": "extra_trees_identity_enriched",
        "features": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "target": TARGET,
        "training_rows": len(train),
        "training_max_transaction_dt": float(train["TransactionDT"].max()),
        "training_fraud_rate": float(train[TARGET].mean()),
        "identity_available_rate": float(train["identity_available"].mean()),
        "random_state": 42,
        "threshold": None,
        "score_semantics": (
            "Uncalibrated model score; not a calibrated fraud probability. "
            "No production decision threshold has been approved."
        ),
        "validation_note": (
            "Experimental artifact trained on the first 70% of the "
            "chronologically ordered IEEE-CIS training dataset."
        ),
    }
    joblib.dump(artifact, OUTPUT)
    print(f"Saved model artifact: {OUTPUT}")
    print(f"Artifact size: {OUTPUT.stat().st_size / (1024 * 1024):.1f} MB")

    # Verify the saved artifact can be loaded and score a few rows.
    loaded = joblib.load(OUTPUT)
    sample_scores = loaded["model"].predict_proba(
        train[loaded["features"]].head(3)
    )[:, 1]
    if len(sample_scores) != 3:
        raise RuntimeError("Artifact smoke test failed.")

    print("Artifact load and scoring smoke test: PASS")
    print("Sample model scores:", [round(float(s), 4) for s in sample_scores])
    print("These sample scores are not validation metrics.")

if __name__ == "__main__":
    main()
