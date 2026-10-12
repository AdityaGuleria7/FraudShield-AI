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
