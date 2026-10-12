import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GENESIS_HASH = "0" * 64


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

    @staticmethod
    def _serialize(event: dict[str, Any]) -> str:
        return json.dumps(
            event,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def _calculate_hash(cls, event: dict[str, Any]) -> str:
        payload = {
            key: value
            for key, value in event.items()
            if key != "event_hash"
        }
        serialized = cls._serialize(payload)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def verify_chain(self) -> bool:
        """Return False if any event is malformed or its chain is broken."""
        if not self.path.exists():
            return True

        previous_hash = GENESIS_HASH

        try:
            with self.path.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        return False

                    event = json.loads(line)

                    if not isinstance(event, dict):
                        return False

                    if "previous_hash" not in event or "event_hash" not in event:
                        return False

                    if event["previous_hash"] != previous_hash:
                        return False

                    recorded_hash = event["event_hash"]
                    if (
                        not isinstance(recorded_hash, str)
                        or len(recorded_hash) != 64
                        or recorded_hash != self._calculate_hash(event)
                    ):
                        return False

                    previous_hash = recorded_hash

        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            return False

        return True

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

        if self.path.exists() and self.path.stat().st_size > 0:
            if not self.verify_chain():
                raise ValueError(
                    "Cannot append to an invalid or legacy audit chain"
                )

            with self.path.open("r", encoding="utf-8") as file:
                last_line = None
                for line in file:
                    last_line = line

            if last_line is None or not last_line.strip():
                raise ValueError("Cannot append to a malformed audit chain")

            previous_hash = json.loads(last_line)["event_hash"]
        else:
            previous_hash = GENESIS_HASH

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "agent_role": agent_role,
            "task_id": task_id,
            "action": action,
            "status": status,
            "confidence": confidence,
            "data": data or {},
            "previous_hash": previous_hash,
        }

        try:
            serialized_payload = self._serialize(entry)
            entry["event_hash"] = hashlib.sha256(
                serialized_payload.encode("utf-8")
            ).hexdigest()
            serialized_entry = self._serialize(entry)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Audit event must contain valid JSON data"
            ) from exc

        with self.path.open("a", encoding="utf-8") as file:
            file.write(serialized_entry + "\n")

        return entry
