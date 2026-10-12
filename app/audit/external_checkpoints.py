import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from app.audit.checkpoints import SignedCheckpointStore
from app.audit.ledger import AuditLedger


class S3CheckpointAnchor:
    """Publish signed audit checkpoints to an independently protected S3 bucket."""

    def __init__(
        self,
        client: Any,
        bucket: str,
        secret_key: bytes | str,
        prefix: str = "fraudshield/audit-checkpoints",
        retention_days: int = 365,
    ):
        if not bucket.strip():
            raise ValueError("S3 bucket name is required")
        if not prefix.strip():
            raise ValueError("S3 prefix is required")
        if isinstance(secret_key, str):
            secret_key = secret_key.encode("utf-8")
        if not isinstance(secret_key, bytes) or len(secret_key) < 32:
            raise ValueError("Checkpoint secret key must be at least 32 bytes")
        if isinstance(retention_days, bool) or not isinstance(retention_days, int):
            raise ValueError("Retention days must be an integer")
        if retention_days < 1:
            raise ValueError("Retention must be at least one day")

        self.client = client
        self.bucket = bucket
        self.secret_key = secret_key
        self.prefix = prefix.strip("/")
        self.retention_days = retention_days

    @staticmethod
    def _canonical_json(value: dict[str, Any]) -> bytes:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    def _valid_checkpoint(self, checkpoint: Any) -> bool:
        if not isinstance(checkpoint, dict):
            return False

        signature = checkpoint.get("signature")
        payload = {
            key: value for key, value in checkpoint.items()
            if key != "signature"
        }

        if not isinstance(signature, str):
            return False

        try:
            expected = hmac.new(
                self.secret_key,
                self._canonical_json(payload),
                hashlib.sha256,
            ).hexdigest()
        except (TypeError, ValueError):
            return False

        sequence = payload.get("sequence")
        event_hash = payload.get("event_hash")

        return (
            hmac.compare_digest(signature, expected)
            and payload.get("version") == 1
            and isinstance(sequence, int)
            and not isinstance(sequence, bool)
            and sequence >= 1
            and isinstance(event_hash, str)
            and len(event_hash) == 64
            and all(char in "0123456789abcdef" for char in event_hash)
        )

    def publish(
        self, checkpoint: dict[str, Any], ledger_path: str | None = None
    ) -> str:
        """Validate and publish a signed checkpoint for the supplied ledger."""
        if not self._valid_checkpoint(checkpoint):
            raise ValueError("Cannot publish an invalid signed checkpoint")
        if ledger_path is None:
            raise ValueError("A ledger path is required to publish a valid checkpoint")

        ledger = AuditLedger(ledger_path)
        if not ledger.verify_chain():
            raise ValueError(
                "Cannot publish a checkpoint for an invalid ledger chain"
            )

        events = SignedCheckpointStore._read_events(ledger_path)
        sequence = checkpoint["sequence"]

        if sequence > len(events):
            raise ValueError("Checkpoint sequence exceeds ledger length")

        actual_hash = events[sequence - 1].get("event_hash")
        if (
            not isinstance(actual_hash, str)
            or not hmac.compare_digest(actual_hash, checkpoint["event_hash"])
        ):
            raise ValueError("Checkpoint does not match ledger history")

        previous = self._latest_checkpoint()
        if previous is not None:
            previous_sequence = previous["sequence"]
            if sequence < previous_sequence:
                raise ValueError("Refusing to publish an older checkpoint")

            previous_hash = events[previous_sequence - 1].get("event_hash") if previous_sequence <= len(events) else None
            if (
                not isinstance(previous_hash, str)
                or not hmac.compare_digest(previous_hash, previous["event_hash"])
            ):
                raise ValueError("Existing external anchor does not match ledger history")

        event_hash = checkpoint["event_hash"]
        key = (
            f"{self.prefix}/anchors/"
            f"{sequence:020d}-{event_hash}.json"
        )

        body = self._canonical_json(checkpoint)
        retain_until = (
            datetime.now(timezone.utc)
            + timedelta(days=self.retention_days)
        )

        # Requires an S3 bucket configured for Object Lock and versioning.
        # Conditional creation prevents silently overwriting an existing key.
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            IfNoneMatch="*",
            ObjectLockMode="COMPLIANCE",
            ObjectLockRetainUntilDate=retain_until,
        )
        return key

    def _latest_checkpoint(self) -> dict[str, Any] | None:
        prefix = f"{self.prefix}/anchors/"
        paginator = self.client.get_paginator("list_objects_v2")
        candidates: list[dict[str, Any]] = []

        for page in paginator.paginate(
            Bucket=self.bucket,
            Prefix=prefix,
        ):
            for item in page.get("Contents", []):
                key = item.get("Key")
                if not isinstance(key, str) or not key.endswith(".json"):
                    continue

                response = self.client.get_object(
                    Bucket=self.bucket,
                    Key=key,
                )
                raw = response["Body"].read()
                checkpoint = json.loads(raw.decode("utf-8"))

                if not self._valid_checkpoint(checkpoint):
                    raise ValueError("Invalid checkpoint found in external storage")

                if key != (
                    f"{prefix}{checkpoint['sequence']:020d}-"
                    f"{checkpoint['event_hash']}.json"
                ):
                    raise ValueError("External checkpoint key does not match its contents")

                candidates.append(checkpoint)

        if not candidates:
            return None

        highest_sequence = max(item["sequence"] for item in candidates)
        latest = [
            item for item in candidates
            if item["sequence"] == highest_sequence
        ]

        hashes = {item["event_hash"] for item in latest}
        if len(hashes) != 1:
            raise ValueError("Conflicting external checkpoints at the same sequence")

        # For the same sequence and event hash, select the newest checkpoint
        # timestamp deterministically. The anchored event hash is identical.
        return max(latest, key=lambda item: item.get("timestamp", ""))

    def verify(self, ledger_path: str) -> bool:
        """Verify the local ledger against the highest external checkpoint."""
        try:
            latest = self._latest_checkpoint()
            if latest is None:
                return False

            ledger = AuditLedger(ledger_path)
            if not ledger.verify_chain():
                return False

            events = SignedCheckpointStore._read_events(ledger_path)
            sequence = latest["sequence"]

            if sequence > len(events):
                return False

            actual_hash = events[sequence - 1].get("event_hash")
            return (
                isinstance(actual_hash, str)
                and hmac.compare_digest(actual_hash, latest["event_hash"])
            )
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            UnicodeError,
            json.JSONDecodeError,
        ):
            return False
