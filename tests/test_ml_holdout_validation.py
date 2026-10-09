from app.ml.pipeline import MLPipeline


def make_feature_vectors():
    vectors = {}

    for index in range(1, 21):
        transaction_id = f"TX{index:03d}"
        vectors[transaction_id] = {
            "transaction_amount": float(index * 100),
            "account_transaction_count": float(index),
            "account_total_amount": float(index * 1000),
            "account_average_amount": float(index * 100),
            "account_max_amount": float(index * 150),
            "velocity_5m_count": 0.0,
            "velocity_5m_amount": 0.0,
            "velocity_10m_count": 0.0,
            "velocity_10m_amount": 0.0,
            "velocity_1h_count": 0.0,
            "velocity_1h_amount": 0.0,
            "account_unique_device_count": 1.0,
            "account_unique_ip_count": 1.0,
            "device_unique_account_count": 1.0,
            "is_new_device_for_account": 0.0,
            "is_new_ip_for_account": 0.0,
            "amount_vs_account_average": 1.0,
            "amount_vs_account_max": 1.0,
            "is_above_account_average": 0.0,
            "is_above_account_max": 0.0,
            "hour_of_day": 12.0,
            "day_of_week": 1.0,
            "is_weekend": 0.0,
            "is_night": 0.0,
            "category_transaction_count": 1.0,
            "category_total_amount": float(index * 100),
            "is_new_category_for_account": 0.0,
            "category_share_of_account_spend": 1.0,
        }

    return vectors


def make_transactions():
    transactions = []

    for index in range(1, 21):
        transactions.append(
            {
                "transaction_id": f"TX{index:03d}",
                "timestamp": (
                    f"2026-01-{index:02d}T10:00:00+00:00"
                ),
                "is_fraud": index % 2 == 0,
            }
        )

    return transactions


def test_ml_pipeline_uses_separate_holdout_evaluation():
    result = MLPipeline().run(
        make_feature_vectors(),
        make_transactions(),
    )

    assert "train_transaction_ids" in result
    assert "validation_transaction_ids" in result
    assert result["train_transaction_ids"]
    assert result["validation_transaction_ids"]
    assert set(result["train_transaction_ids"]).isdisjoint(
        result["validation_transaction_ids"]
    )


def test_ml_pipeline_rejects_dataset_too_small_for_holdout():
    feature_vectors = make_feature_vectors()
    transactions = make_transactions()[:4]

    feature_vectors = {
        transaction["transaction_id"]: feature_vectors[transaction["transaction_id"]]
        for transaction in transactions
    }

    try:
        MLPipeline().run(feature_vectors, transactions)
    except ValueError as exc:
        assert "holdout" in str(exc).lower() or "split" in str(exc).lower()
        return

    raise AssertionError("Tiny dataset was accepted without a clear holdout error")


def test_ml_pipeline_reports_prediction_provenance():
    pipeline = MLPipeline()

    result = pipeline.run(
        make_feature_vectors(),
        make_transactions(),
    )

    assert result["prediction_provenance"] == "in_sample"
    assert result["validation_prediction_provenance"] == "out_of_sample"


def test_ml_evidence_builder_rejects_in_sample_prediction_evidence():
    from app.ml.evidence import MLEvidenceBuilder

    result = MLPipeline().run(
        make_feature_vectors(),
        make_transactions(),
    )

    transaction_id = result["train_transaction_ids"][0]

    try:
        MLEvidenceBuilder().build(result, transaction_id)
    except ValueError as exc:
        assert "out_of_sample" in str(exc).lower()
        return

    raise AssertionError(
        "In-sample prediction was incorrectly accepted as ML evidence"
    )


def test_ml_pipeline_temporal_validation_uses_future_transactions_as_holdout():
    feature_vectors = make_feature_vectors()

    transactions = [
        {
            "transaction_id": "TX003",
            "timestamp": "2026-01-03T10:00:00+00:00",
            "is_fraud": False,
        },
        {
            "transaction_id": "TX001",
            "timestamp": "2026-01-01T10:00:00+00:00",
            "is_fraud": False,
        },
        {
            "transaction_id": "TX004",
            "timestamp": "2026-01-04T10:00:00+00:00",
            "is_fraud": True,
        },
        {
            "transaction_id": "TX002",
            "timestamp": "2026-01-02T10:00:00+00:00",
            "is_fraud": False,
        },
        {
            "transaction_id": "TX005",
            "timestamp": "2026-01-05T10:00:00+00:00",
            "is_fraud": False,
        },
        {
            "transaction_id": "TX006",
            "timestamp": "2026-01-06T10:00:00+00:00",
            "is_fraud": True,
        },
        {
            "transaction_id": "TX007",
            "timestamp": "2026-01-07T10:00:00+00:00",
            "is_fraud": False,
        },
        {
            "transaction_id": "TX008",
            "timestamp": "2026-01-08T10:00:00+00:00",
            "is_fraud": True,
        },
        {
            "transaction_id": "TX009",
            "timestamp": "2026-01-09T10:00:00+00:00",
            "is_fraud": False,
        },
        {
            "transaction_id": "TX010",
            "timestamp": "2026-01-10T10:00:00+00:00",
            "is_fraud": True,
        },
        {
            "transaction_id": "TX011",
            "timestamp": "2026-01-11T10:00:00+00:00",
            "is_fraud": False,
        },
        {
            "transaction_id": "TX012",
            "timestamp": "2026-01-12T10:00:00+00:00",
            "is_fraud": True,
        },
        {
            "transaction_id": "TX013",
            "timestamp": "2026-01-13T10:00:00+00:00",
            "is_fraud": False,
        },
        {
            "transaction_id": "TX014",
            "timestamp": "2026-01-14T10:00:00+00:00",
            "is_fraud": True,
        },
        {
            "transaction_id": "TX015",
            "timestamp": "2026-01-15T10:00:00+00:00",
            "is_fraud": False,
        },
        {
            "transaction_id": "TX016",
            "timestamp": "2026-01-16T10:00:00+00:00",
            "is_fraud": True,
        },
        {
            "transaction_id": "TX017",
            "timestamp": "2026-01-17T10:00:00+00:00",
            "is_fraud": False,
        },
        {
            "transaction_id": "TX018",
            "timestamp": "2026-01-18T10:00:00+00:00",
            "is_fraud": True,
        },
        {
            "transaction_id": "TX019",
            "timestamp": "2026-01-19T10:00:00+00:00",
            "is_fraud": False,
        },
        {
            "transaction_id": "TX020",
            "timestamp": "2026-01-20T10:00:00+00:00",
            "is_fraud": True,
        },
    ]

    result = MLPipeline().run(feature_vectors, transactions)

    train_ids = set(result["train_transaction_ids"])
    validation_ids = set(result["validation_transaction_ids"])

    assert train_ids
    assert validation_ids

    timestamps = {
        transaction["transaction_id"]: transaction["timestamp"]
        for transaction in transactions
    }

    latest_train_timestamp = max(
        timestamps[transaction_id]
        for transaction_id in train_ids
    )

    earliest_validation_timestamp = min(
        timestamps[transaction_id]
        for transaction_id in validation_ids
    )

    assert latest_train_timestamp < earliest_validation_timestamp
