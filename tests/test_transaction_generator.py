from app.data.transaction_generator import TransactionGenerator


def test_transaction_generator_creates_requested_count():

    generator = TransactionGenerator(seed=42)

    transactions = generator.generate(count=25)

    assert len(transactions) == 25


def test_transaction_has_required_fields():

    generator = TransactionGenerator(seed=42)

    transactions = generator.generate(count=1)

    transaction = transactions[0]

    required_fields = {
        "transaction_id",
        "account_id",
        "device_id",
        "ip_address",
        "country",
        "merchant_category",
        "amount",
        "timestamp",
        "is_fraud",
    }

    assert required_fields.issubset(transaction.keys())


def test_transaction_ids_are_unique():

    generator = TransactionGenerator(seed=42)

    transactions = generator.generate(count=100)

    ids = [
        transaction["transaction_id"]
        for transaction in transactions
    ]

    assert len(ids) == len(set(ids))


def test_generator_is_reproducible():

    generator_a = TransactionGenerator(seed=42)
    generator_b = TransactionGenerator(seed=42)

    assert generator_a.generate(10) == generator_b.generate(10)
