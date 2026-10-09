from typing import Callable, Any

from app.audit.ledger import AuditLedger
from app.security.policy_engine import PolicyEngine, PolicyDecision


class SecureExecutor:

    def __init__(
        self,
        policy_engine: PolicyEngine | None = None,
        audit_path: str = "audit_ledger.jsonl",
    ):
        self.policy_engine = policy_engine or PolicyEngine()
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

        result = operation(*args, **kwargs)

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
