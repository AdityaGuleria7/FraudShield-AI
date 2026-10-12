import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLedger:

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

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

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

        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry) + "\n")

        return entry
