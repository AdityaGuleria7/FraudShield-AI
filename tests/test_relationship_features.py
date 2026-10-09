from app.features.relationships import RelationshipFeatureBuilder


def sample_transactions():
    return [
        {
            "transaction_id": "TX001",
            "account_id": "ACC001",
            "device_id": "DEV001",
            "ip_address": "192.168.1.10",
        },
        {
            "transaction_id": "TX002",
            "account_id": "ACC001",
            "device_id": "DEV001",
            "ip_address": "192.168.1.10",
        },
        {
            "transaction_id": "TX003",
            "account_id": "ACC001",
            "device_id": "DEV002",
            "ip_address": "192.168.1.11",
        },
        {
            "transaction_id": "TX004",
            "account_id": "ACC002",
            "device_id": "DEV002",
            "ip_address": "192.168.1.12",
        },
    ]


def test_account_relationship_features():
    builder = RelationshipFeatureBuilder()

    features = builder.build(sample_transactions())

    assert features["TX001"]["account_unique_device_count"] == 1
    assert features["TX001"]["account_unique_ip_count"] == 1

    assert features["TX003"]["account_unique_device_count"] == 2
    assert features["TX003"]["account_unique_ip_count"] == 2


def test_device_account_relationship():
    builder = RelationshipFeatureBuilder()

    features = builder.build(sample_transactions())

    assert features["TX001"]["device_unique_account_count"] == 1
    assert features["TX003"]["device_unique_account_count"] == 1
    assert features["TX004"]["device_unique_account_count"] == 2


def test_new_device_and_ip_flags():
    builder = RelationshipFeatureBuilder()

    features = builder.build(sample_transactions())

    assert features["TX001"]["is_new_device_for_account"] == 1
    assert features["TX001"]["is_new_ip_for_account"] == 1

    assert features["TX002"]["is_new_device_for_account"] == 0
    assert features["TX002"]["is_new_ip_for_account"] == 0

    assert features["TX003"]["is_new_device_for_account"] == 1
    assert features["TX003"]["is_new_ip_for_account"] == 1

    assert features["TX004"]["is_new_device_for_account"] == 1
    assert features["TX004"]["is_new_ip_for_account"] == 1
