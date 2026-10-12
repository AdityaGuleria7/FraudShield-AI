from typing import Callable, Any


class OperationRegistry:

    def __init__(self):
        self._operations: dict[tuple[str, str], Callable[..., Any]] = {}

    def register(
        self,
        action: str,
        resource: str,
        operation: Callable[..., Any],
    ) -> None:
        if not action or not action.strip():
            raise ValueError("Action cannot be blank")

        if not resource or not resource.strip():
            raise ValueError("Resource cannot be blank")

        key = (action, resource)

        if key in self._operations:
            raise ValueError(
                f"Operation already registered for "
                f"action '{action}' and resource '{resource}'"
            )

        self._operations[key] = operation

    def resolve(
        self,
        action: str,
        resource: str,
    ) -> Callable[..., Any]:
        key = (action, resource)

        if key not in self._operations:
            raise KeyError(
                f"No operation registered for "
                f"action '{action}' and resource '{resource}'"
            )

        return self._operations[key]
