from pathlib import Path
import json

import pandas as pd

DATA_DIR = Path.home() / "Downloads" / "ieee-fraud-detection"
TRANSACTION_FILE = DATA_DIR / "train_transaction.csv"
IDENTITY_FILE = DATA_DIR / "train_identity.csv"


def main():
    for path in [TRANSACTION_FILE, IDENTITY_FILE]:
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {path}\n"
                "Check the extraction folder and update DATA_DIR if needed."
            )

    print("Reading transaction identifiers and labels...")
    transactions = pd.read_csv(
        TRANSACTION_FILE,
        usecols=["TransactionID", "TransactionDT", "isFraud"],
    )

    print("Reading identity data...")
    identity = pd.read_csv(IDENTITY_FILE)

    transaction_ids = transactions["TransactionID"]
    identity_ids = identity["TransactionID"]

    duplicate_transaction_ids = int(transaction_ids.duplicated().sum())
    duplicate_identity_ids = int(identity_ids.duplicated().sum())

    transaction_id_set = set(transaction_ids)
    identity_id_set = set(identity_ids)

    matched_ids = transaction_id_set.intersection(identity_id_set)
    unmatched_identity_ids = identity_id_set - transaction_id_set

    merged = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    has_identity = merged["_merge"].eq("both")
    fraud_rate_with_identity = float(
        merged.loc[has_identity, "isFraud"].mean()
    )
    fraud_rate_without_identity = float(
        merged.loc[~has_identity, "isFraud"].mean()
    )

    missing = identity.drop(columns=["TransactionID"]).isna().mean()
    missing_report = (
        (missing * 100)
        .sort_values(ascending=False)
        .round(2)
        .to_dict()
    )

    feature_types = {
        column: str(dtype)
        for column, dtype in identity.dtypes.items()
        if column != "TransactionID"
    }

    report = {
        "dataset": "IEEE-CIS train identity",
        "transaction_rows": int(len(transactions)),
        "identity_rows": int(len(identity)),
        "transaction_columns_inspected": [
            "TransactionID", "TransactionDT", "isFraud"
        ],
        "identity_feature_count_excluding_id": int(
            len(identity.columns) - 1
        ),
        "duplicate_transaction_ids": duplicate_transaction_ids,
        "duplicate_identity_ids": duplicate_identity_ids,
        "transaction_ids_with_identity": int(len(matched_ids)),
        "transactions_with_identity_percent": round(
            100 * len(matched_ids) / len(transaction_ids), 2
        ),
        "identity_ids_not_in_transactions": int(
            len(unmatched_identity_ids)
        ),
        "fraud_rate_with_identity": fraud_rate_with_identity,
        "fraud_rate_without_identity": fraud_rate_without_identity,
        "missing_percent_by_identity_feature": missing_report,
        "identity_feature_types": feature_types,
    }

    Path("reports").mkdir(exist_ok=True)
    output = Path("reports/ieee_identity_inspection.json")
    output.write_text(json.dumps(report, indent=2))

    print("\n=== IDENTITY DATA SUMMARY ===")
    print(f"Transaction rows: {len(transactions):,}")
    print(f"Identity rows: {len(identity):,}")
    print(
        "Identity features (excluding TransactionID): "
        f"{len(identity.columns) - 1}"
    )
    print(f"Duplicate transaction IDs: {duplicate_transaction_ids:,}")
    print(f"Duplicate identity IDs: {duplicate_identity_ids:,}")
    print(
        f"Transactions matched to identity: {len(matched_ids):,} "
        f"({report['transactions_with_identity_percent']:.2f}%)"
    )
    print(
        "Identity IDs absent from transaction file: "
        f"{len(unmatched_identity_ids):,}"
    )
    print(
        "Fraud rate with identity: "
        f"{fraud_rate_with_identity:.4%}"
    )
    print(
        "Fraud rate without identity: "
        f"{fraud_rate_without_identity:.4%}"
    )

    print("\n=== IDENTITY FEATURES WITH MOST MISSING VALUES ===")
    for feature, pct in list(missing_report.items())[:20]:
        print(f"{feature}: {pct:.2f}% missing")

    print("\n=== IDENTITY FEATURE TYPES ===")
    for feature, dtype in feature_types.items():
        print(f"{feature}: {dtype}")

    print(f"\nSaved report: {output}")
    print(
        "\nNote: differences in fraud rate are associations, not proof "
        "that identity data causes or prevents fraud."
    )


if __name__ == "__main__":
    main()
