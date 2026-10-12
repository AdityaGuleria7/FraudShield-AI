import pytest

from app.security.operation_registry import OperationRegistry


def test_registered_operation_can_be_resolved():
    registry = OperationRegistry()

    def read_transactions():
        return ["TX001"]

    registry.register(
        action="READ",
        resource="TRANSACTION_DATA",
        operation=read_transactions,
    )

    operation = registry.resolve(
        action="READ",
        resource="TRANSACTION_DATA",
    )

    assert operation is read_transactions


def test_unregistered_operation_is_not_resolvable():
    registry = OperationRegistry()

    with pytest.raises(KeyError):
        registry.resolve(
            action="DELETE",
            resource="TRANSACTION_DATA",
        )


def test_duplicate_operation_registration_is_rejected():
    registry = OperationRegistry()

    def operation_one():
        return "one"

    def operation_two():
        return "two"

    registry.register(
        action="READ",
        resource="TRANSACTION_DATA",
        operation=operation_one,
    )

    with pytest.raises(ValueError):
        registry.register(
            action="READ",
            resource="TRANSACTION_DATA",
            operation=operation_two,
        )


def test_blank_action_cannot_be_registered():
    registry = OperationRegistry()

    with pytest.raises(ValueError):
        registry.register(
            action="",
            resource="TRANSACTION_DATA",
            operation=lambda: None,
        )


def test_blank_resource_cannot_be_registered():
    registry = OperationRegistry()

    with pytest.raises(ValueError):
        registry.register(
            action="READ",
            resource="",
            operation=lambda: None,
        )
