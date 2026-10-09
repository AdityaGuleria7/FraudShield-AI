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
                "is_fraud": index > 10,
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
