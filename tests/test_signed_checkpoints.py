import json

import pytest

from app.audit.ledger import AuditLedger
from app.audit.checkpoints import SignedCheckpointStore


SECRET = b"test-only-checkpoint-key-32-bytes-minimum!"


def build_ledger(path, count=3):
    ledger = AuditLedger(str(path))
    for index in range(count):
        ledger.record(
            agent_id="TEST_AGENT",
            agent_role="SECURITY_TEST",
            task_id="CHECKPOINT-TEST",
            action=f"TEST_ACTION_{index}",
            status="PASS",
            confidence=1.0,
            data={"index": index},
        )
    return ledger


def test_signed_checkpoint_verifies_valid_ledger(tmp_path):
    ledger_path = tmp_path / "audit.jsonl"
    checkpoint_path = tmp_path / "checkpoint.json"
    build_ledger(ledger_path)

    store = SignedCheckpointStore(checkpoint_path, SECRET)
    checkpoint = store.create(ledger_path)

    assert checkpoint["sequence"] == 3
    assert len(checkpoint["event_hash"]) == 64
    assert store.verify(ledger_path) is True


def test_signed_checkpoint_detects_ledger_truncation(tmp_path):
    ledger_path = tmp_path / "audit.jsonl"
    checkpoint_path = tmp_path / "checkpoint.json"
    build_ledger(ledger_path, count=3)

    store = SignedCheckpointStore(checkpoint_path, SECRET)
    store.create(ledger_path)

    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    ledger_path.write_text(lines[0] + "\n", encoding="utf-8")

    assert store.verify(ledger_path) is False


def test_signed_checkpoint_rejects_wrong_key(tmp_path):
    ledger_path = tmp_path / "audit.jsonl"
    checkpoint_path = tmp_path / "checkpoint.json"
    build_ledger(ledger_path)

    SignedCheckpointStore(checkpoint_path, SECRET).create(ledger_path)
    wrong_store = SignedCheckpointStore(
        checkpoint_path,
        b"different-key-that-is-also-32-bytes-long!",
    )

    assert wrong_store.verify(ledger_path) is False


def test_signed_checkpoint_detects_signature_tampering(tmp_path):
    ledger_path = tmp_path / "audit.jsonl"
    checkpoint_path = tmp_path / "checkpoint.json"
    build_ledger(ledger_path)

    store = SignedCheckpointStore(checkpoint_path, SECRET)
    store.create(ledger_path)

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["signature"] = "0" * 64
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    assert store.verify(ledger_path) is False


def test_signed_checkpoint_rejects_rollback(tmp_path):
    ledger_path = tmp_path / "audit.jsonl"
    checkpoint_path = tmp_path / "checkpoint.json"
    build_ledger(ledger_path, count=3)

    store = SignedCheckpointStore(checkpoint_path, SECRET)
    store.create(ledger_path)

    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    ledger_path.write_text(lines[0] + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="rollback|older|checkpoint"):
        store.create(ledger_path)


def test_signed_checkpoint_allows_valid_appends_after_checkpoint(tmp_path):
    ledger_path = tmp_path / "audit.jsonl"
    checkpoint_path = tmp_path / "checkpoint.json"
    ledger = build_ledger(ledger_path, count=2)

    store = SignedCheckpointStore(checkpoint_path, SECRET)
    store.create(ledger_path)

    ledger.record(
        agent_id="TEST_AGENT",
        agent_role="SECURITY_TEST",
        task_id="CHECKPOINT-TEST",
        action="AFTER_CHECKPOINT",
        status="PASS",
        confidence=1.0,
        data={},
    )

    assert store.verify(ledger_path) is True


def test_signed_checkpoint_requires_strong_key(tmp_path):
    with pytest.raises(ValueError, match="32|key|secret"):
        SignedCheckpointStore(tmp_path / "checkpoint.json", b"short")


def test_checkpoint_refuses_rewritten_history_when_advancing(tmp_path):
    import hashlib

    ledger_path = tmp_path / "audit.jsonl"
    checkpoint_path = tmp_path / "checkpoint.json"

    ledger = build_ledger(ledger_path, count=2)
    store = SignedCheckpointStore(checkpoint_path, SECRET)
    store.create(ledger_path)

    # Rewrite an earlier event and recompute a structurally valid hash chain.
    events = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    previous_hash = "0" * 64

    for index, event in enumerate(events):
        event.pop("event_hash", None)
        event["previous_hash"] = previous_hash

        if index == 0:
            event["data"]["index"] = 999

        canonical = json.dumps(
            event,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        event_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        event["event_hash"] = event_hash
        previous_hash = event_hash

    ledger_path.write_text(
        "".join(
            json.dumps(
                event,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n"
            for event in events
        ),
        encoding="utf-8",
    )

    # Extend the rewritten chain so it is longer than the original checkpoint.
    ledger.record(
        agent_id="TEST_AGENT",
        agent_role="SECURITY_TEST",
        task_id="CHECKPOINT-TEST",
        action="AFTER_REWRITE",
        status="PASS",
        confidence=1.0,
        data={},
    )

    with pytest.raises(ValueError, match="checkpoint|anchor|history|chain"):
        store.create(ledger_path)


class FakeS3Paginator:
    def __init__(self, client):
        self.client = client

    def paginate(self, Bucket, Prefix):
        contents = [
            {"Key": key}
            for bucket, key in self.client.objects
            if bucket == Bucket and key.startswith(Prefix)
        ]
        yield {"Contents": contents}


class FakeS3Client:
    def __init__(self):
        self.objects = {}

    def put_object(self, **kwargs):
        key = (kwargs["Bucket"], kwargs["Key"])
        if kwargs.get("IfNoneMatch") == "*" and key in self.objects:
            raise ValueError("Object already exists")
        self.objects[key] = kwargs["Body"]

    def get_object(self, Bucket, Key):
        return {"Body": FakeS3Body(self.objects[(Bucket, Key)])}

    def get_paginator(self, operation_name):
        assert operation_name == "list_objects_v2"
        return FakeS3Paginator(self)


class FakeS3Body:
    def __init__(self, body):
        self.body = body

    def read(self):
        return self.body


def test_external_anchor_publishes_and_verifies_checkpoint(tmp_path):
    from app.audit.external_checkpoints import S3CheckpointAnchor

    ledger_path = tmp_path / "audit.jsonl"
    checkpoint_path = tmp_path / "checkpoint.json"
    build_ledger(ledger_path, count=3)

    store = SignedCheckpointStore(checkpoint_path, SECRET)
    checkpoint = store.create(ledger_path)
    client = FakeS3Client()
    external = S3CheckpointAnchor(client, "test-bucket", SECRET)

    key = external.publish(checkpoint, str(ledger_path))

    assert key.endswith(
        f"{checkpoint['sequence']:020d}-{checkpoint['event_hash']}.json"
    )
    assert external.verify(str(ledger_path)) is True


def test_external_anchor_detects_ledger_rollback(tmp_path):
    from app.audit.external_checkpoints import S3CheckpointAnchor

    ledger_path = tmp_path / "audit.jsonl"
    checkpoint_path = tmp_path / "checkpoint.json"
    build_ledger(ledger_path, count=3)

    store = SignedCheckpointStore(checkpoint_path, SECRET)
    checkpoint = store.create(ledger_path)
    external = S3CheckpointAnchor(FakeS3Client(), "test-bucket", SECRET)
    external.publish(checkpoint, str(ledger_path))

    first_line = ledger_path.read_text(encoding="utf-8").splitlines()[0]
    ledger_path.write_text(first_line + "\n", encoding="utf-8")

    assert external.verify(str(ledger_path)) is False


def test_external_anchor_rejects_invalid_signature(tmp_path):
    from app.audit.external_checkpoints import S3CheckpointAnchor

    client = FakeS3Client()
    external = S3CheckpointAnchor(client, "test-bucket", SECRET)

    with pytest.raises(ValueError, match="invalid signed checkpoint"):
        external.publish({
            "version": 1,
            "sequence": 1,
            "event_hash": "a" * 64,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "signature": "invalid",
        })


def test_external_anchor_rejects_duplicate_object_key(tmp_path):
    from app.audit.external_checkpoints import S3CheckpointAnchor

    ledger_path = tmp_path / "audit.jsonl"
    checkpoint_path = tmp_path / "checkpoint.json"
    build_ledger(ledger_path, count=2)

    checkpoint = SignedCheckpointStore(
        checkpoint_path, SECRET
    ).create(ledger_path)

    external = S3CheckpointAnchor(FakeS3Client(), "test-bucket", SECRET)
    external.publish(checkpoint, str(ledger_path))

    with pytest.raises(ValueError, match="already exists"):
        external.publish(checkpoint, str(ledger_path))
