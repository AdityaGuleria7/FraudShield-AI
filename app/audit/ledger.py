import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLedger:
    ALLOWED_STATUSES = frozenset({
        "ALLOW",
        "CONDITIONAL_PASS",
        "DENY",
        "ERROR",
        "PASS",
        "REJECT",
    })

    def __init__(self, path: str = "audit_ledger.jsonl"):
        self.path = Path(path)

    def record(
        self,
        agent_id: str,
        agent_role: str,
        task_id: str,
        action: str,
        status: str,
        confidence: float,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        if not agent_id or not agent_id.strip():
            raise ValueError("agent_id cannot be blank")

        if not agent_role or not agent_role.strip():
            raise ValueError("agent_role cannot be blank")

        if not task_id or not task_id.strip():
            raise ValueError("task_id cannot be blank")

        if not action or not action.strip():
            raise ValueError("action cannot be blank")

        if not isinstance(status, str) or status not in self.ALLOWED_STATUSES:
            raise ValueError(f"Unsupported audit status: {status!r}")

        if data is not None and not isinstance(data, dict):
            raise ValueError("data must be a dictionary or None")

        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError(
                "confidence must be a finite number between 0.0 and 1.0"
            )

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "agent_role": agent_role,
            "task_id": task_id,
            "action": action,
            "status": status,
            "confidence": confidence,
            "data": data or {},
        }

        try:
            serialized_entry = json.dumps(entry, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Audit event must contain valid JSON data"
            ) from exc

        with self.path.open("a", encoding="utf-8") as file:
            file.write(serialized_entry + "\n")

        return entry
