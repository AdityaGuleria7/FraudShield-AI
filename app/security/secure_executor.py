from typing import Callable, Any

from app.audit.ledger import AuditLedger
from app.security.policy_engine import PolicyEngine, PolicyDecision
from app.security.operation_registry import OperationRegistry


class SecureExecutor:

    def __init__(
        self,
        policy_engine: PolicyEngine | None = None,
        operation_registry: OperationRegistry | None = None,
        audit_path: str = "audit_ledger.jsonl",
    ):
        self.policy_engine = policy_engine or PolicyEngine()
        self.operation_registry = operation_registry or OperationRegistry()
        self.audit_ledger = AuditLedger(audit_path)

    def execute(
        self,
        agent_id: str,
        action: str,
        operation: Callable[..., Any],
        *args,
        resource: str = "UNSPECIFIED",
        task_id: str = "UNKNOWN",
        **kwargs,
    ) -> tuple[PolicyDecision, Any]:

        decision = self.policy_engine.check(
            agent_id=agent_id,
            action=action,
            resource=resource,
        )

        self.audit_ledger.record(
            agent_id=agent_id,
            agent_role="SECURITY_EXECUTOR",
            task_id=task_id,
            action=f"POLICY_CHECK:{action}",
            status="ALLOW" if decision.allowed else "DENY",
            confidence=1.0,
            data={
                "requested_action": action,
                "resource": resource,
                "allowed": decision.allowed,
                "reason": decision.reason,
            },
        )

        if not decision.allowed:
            return decision, None

        registered_operation = self.operation_registry.resolve(
            action=action,
            resource=resource,
        )

        if registered_operation is not operation:
            raise ValueError(
                "Provided operation does not match the "
                "registered operation for this action and resource."
            )

        try:
            result = operation(*args, **kwargs)
        except Exception as exc:
            self.audit_ledger.record(
                agent_id=agent_id,
                agent_role="SECURITY_EXECUTOR",
                task_id=task_id,
                action=f"ACTION_EXECUTED:{action}",
                status="ERROR",
                confidence=1.0,
                data={
                    "resource": resource,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise

        self.audit_ledger.record(
            agent_id=agent_id,
            agent_role="SECURITY_EXECUTOR",
            task_id=task_id,
            action=f"ACTION_EXECUTED:{action}",
            status="PASS",
            confidence=1.0,
            data={
                "resource": resource,
                "result_type": type(result).__name__,
            },
        )

        return decision, result
