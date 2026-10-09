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
