import hashlib
import hmac
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.audit.ledger import AuditLedger


class SignedCheckpointStore:
    """HMAC-authenticated checkpoints anchoring an audit-ledger prefix."""

    def __init__(self, checkpoint_path: str | Path, secret_key: bytes | str):
        self.checkpoint_path = Path(checkpoint_path)

        if isinstance(secret_key, str):
            secret_key = secret_key.encode("utf-8")

        if not isinstance(secret_key, bytes) or len(secret_key) < 32:
            raise ValueError("Checkpoint secret key must be at least 32 bytes")

        self._secret_key = secret_key

    @staticmethod
    def _canonical_json(payload: dict[str, Any]) -> bytes:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    def _signature(self, payload: dict[str, Any]) -> str:
        return hmac.new(
            self._secret_key,
            self._canonical_json(payload),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _read_events(ledger_path: str | Path) -> list[dict[str, Any]]:
        path = Path(ledger_path)
        if not AuditLedger(str(path)).verify_chain():
            raise ValueError("Cannot checkpoint an invalid audit chain")

        if not path.exists():
            raise ValueError("Cannot checkpoint a ledger that does not exist")

        events = []
        try:
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise ValueError("Audit event must be a JSON object")
                    events.append(event)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Cannot read audit ledger") from exc

        if not events:
            raise ValueError("Cannot checkpoint an empty audit ledger")

        return events

    def _load_existing(self) -> dict[str, Any] | None:
        if not self.checkpoint_path.exists():
            return None

        try:
            checkpoint = json.loads(
                self.checkpoint_path.read_text(encoding="utf-8")
            )
            if not isinstance(checkpoint, dict):
                return None

            signature = checkpoint.get("signature")
            payload = {
                key: value
                for key, value in checkpoint.items()
                if key != "signature"
            }

            if not isinstance(signature, str):
                return None

            if not hmac.compare_digest(signature, self._signature(payload)):
                return None

            sequence = payload.get("sequence")
            event_hash = payload.get("event_hash")
            if (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence < 1
                or not isinstance(event_hash, str)
                or len(event_hash) != 64
            ):
                return None

            return checkpoint
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            return None

    def create(self, ledger_path: str | Path) -> dict[str, Any]:
        events = self._read_events(ledger_path)
        sequence = len(events)
        event_hash = events[-1].get("event_hash")

        if not isinstance(event_hash, str) or len(event_hash) != 64:
            raise ValueError("Latest audit event has an invalid hash")

        existing = self._load_existing()
        if self.checkpoint_path.exists() and existing is None:
            raise ValueError("Existing checkpoint is invalid; refusing replacement")

        if existing is not None:
            old_sequence = existing["sequence"]
            old_hash = existing["event_hash"]

            if sequence < old_sequence:
                raise ValueError("Refusing checkpoint rollback to an older ledger")

            # Preserve the previously trusted history before advancing the anchor.
            anchored_old_hash = events[old_sequence - 1].get("event_hash")
            if (
                not isinstance(anchored_old_hash, str)
                or not hmac.compare_digest(anchored_old_hash, old_hash)
            ):
                raise ValueError(
                    "Existing checkpoint anchor does not match ledger history"
                )

            if sequence == old_sequence and event_hash != old_hash:
                raise ValueError("Refusing checkpoint replacement with a different chain")

        payload = {
            "version": 1,
            "sequence": sequence,
            "event_hash": event_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        checkpoint = {
            **payload,
            "signature": self._signature(payload),
        }

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.checkpoint_path.name}.",
            dir=str(self.checkpoint_path.parent),
        )

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                file.write(json.dumps(checkpoint, sort_keys=True) + "\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_name, self.checkpoint_path)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

        return checkpoint

    def verify(self, ledger_path: str | Path) -> bool:
        checkpoint = self._load_existing()
        if checkpoint is None:
            return False

        try:
            events = self._read_events(ledger_path)
        except (OSError, ValueError):
            return False

        sequence = checkpoint["sequence"]
        if sequence > len(events):
            return False

        anchored_event = events[sequence - 1]
        if not hmac.compare_digest(
            str(anchored_event.get("event_hash", "")),
            checkpoint["event_hash"],
        ):
            return False

        return True
