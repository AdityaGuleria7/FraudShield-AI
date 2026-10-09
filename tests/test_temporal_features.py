from app.features.temporal import TemporalFeatureBuilder


def sample_transactions():
    return [
        {
            "transaction_id": "TX001",
            "timestamp": "2026-01-05T10:30:00+00:00",
        },
        {
            "transaction_id": "TX002",
            "timestamp": "2026-01-10T23:15:00+00:00",
        },
        {
            "transaction_id": "TX003",
            "timestamp": "2026-01-11T02:45:00+00:00",
        },
    ]


def test_extracts_time_features():
    builder = TemporalFeatureBuilder()

    features = builder.build(sample_transactions())

    assert features["TX001"]["hour_of_day"] == 10
    assert features["TX001"]["day_of_week"] == 0
    assert features["TX001"]["is_weekend"] == 0
    assert features["TX001"]["is_night"] == 0


def test_identifies_weekend_transactions():
    builder = TemporalFeatureBuilder()

    features = builder.build(sample_transactions())

    assert features["TX002"]["day_of_week"] == 5
    assert features["TX002"]["is_weekend"] == 1

    assert features["TX003"]["day_of_week"] == 6
    assert features["TX003"]["is_weekend"] == 1


def test_identifies_night_transactions():
    builder = TemporalFeatureBuilder()

    features = builder.build(sample_transactions())

    assert features["TX002"]["is_night"] == 1
    assert features["TX003"]["is_night"] == 1
