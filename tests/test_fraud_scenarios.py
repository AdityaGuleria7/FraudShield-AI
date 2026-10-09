from app.data.transaction_generator import TransactionGenerator
from app.fraud.scenario_generator import FraudScenarioGenerator


def test_velocity_attack_injection():

    transactions = TransactionGenerator(seed=42).generate(10)

    generator = FraudScenarioGenerator()

    result = generator.inject_velocity_attack(transactions)

    fraud_transactions = [
        transaction
        for transaction in result
        if transaction.get("fraud_scenario") == "VELOCITY_ATTACK"
    ]

    assert len(result) == 15
    assert len(fraud_transactions) == 5
    assert all(
        transaction["is_fraud"] is True
        for transaction in fraud_transactions
    )


def test_new_device_high_value_injection():

    transactions = TransactionGenerator(seed=42).generate(10)

    generator = FraudScenarioGenerator()

    result = generator.inject_new_device_high_value(
        transactions
    )

    fraud_transactions = [
        transaction
        for transaction in result
        if transaction.get("fraud_scenario")
        == "NEW_DEVICE_HIGH_VALUE"
    ]

    assert len(result) == 11
    assert len(fraud_transactions) == 1

    transaction = fraud_transactions[0]

    assert transaction["is_fraud"] is True
    assert transaction["device_id"] == "DEV-NEW-999"
    assert transaction["amount"] == 95000.00
