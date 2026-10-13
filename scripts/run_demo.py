import json
import tempfile
from pathlib import Path
from dataclasses import asdict, is_dataclass
from enum import Enum

from app.audit.ledger import AuditLedger
from app.core.orchestrator import FraudShieldOrchestrator
from app.data.transaction_generator import TransactionGenerator
from app.fraud.scenario_generator import FraudScenarioGenerator


def plain(value):
    if is_dataclass(value) and not isinstance(value, type):
        return plain(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if hasattr(value, "model_dump"):
        return plain(value.model_dump())
    if hasattr(value, "__dict__"):
        return plain(vars(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def field(value, name, default="Not available"):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def main():
    print("=" * 64)
    print("              FRAUDSHIELD AI - DEMO")
    print("=" * 64)
    print("Mode: Synthetic demonstration; no real transactions are used.")

    transactions = TransactionGenerator(seed=42).generate(count=10)
    transactions = FraudScenarioGenerator().inject_velocity_attack(
        transactions
    )

    print(f"\nTransactions supplied: {len(transactions)}")
    print("Injected scenario: synthetic velocity attack")

    with tempfile.TemporaryDirectory(prefix="fraudshield-demo-") as tmp:
        audit_path = Path(tmp) / "demo_audit_ledger.jsonl"

        orchestrator = FraudShieldOrchestrator(
            audit_path=str(audit_path)
        )

        result = orchestrator.run(
            task_id="FRAUDSHIELD-DEMO-001",
            data={"transactions": transactions},
        )

        print("\n--- DATA QUALITY ---")
        print(
            json.dumps(
                plain(result.get("data_quality")),
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\n--- FRAUD FINDINGS ---")
        findings = result.get("pattern_findings") or []
        print(f"Findings returned: {len(findings)}")

        for index, finding in enumerate(findings, start=1):
            item = plain(finding)
            print(f"\nFinding {index}:")
            print(json.dumps(item, indent=2, ensure_ascii=False))

        print("\n--- EVIDENCE ASSESSMENT ---")
        print(
            json.dumps(
                plain(result.get("evidence_assessment")),
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\n--- INVESTIGATION CASE ---")
        print(
            json.dumps(
                plain(result.get("investigation_case")),
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\n--- AGENT REVIEW STATUS ---")
        for key in ("review", "pattern_analysis", "pattern_review"):
            item = result.get(key)
            print(
                f"{key}: status={field(item, 'status')}, "
                f"approved={field(item, 'approved_for_next_stage')}"
            )

        print("\n--- AUDIT INTEGRITY ---")
        ledger = AuditLedger(str(audit_path))
        valid = ledger.verify_chain()

        with audit_path.open("r", encoding="utf-8") as file:
            event_count = sum(1 for line in file if line.strip())

        print(f"Audit events written: {event_count}")
        print(f"Hash chain valid: {valid}")

        if not valid:
            raise RuntimeError("Audit ledger integrity verification failed.")

        if not findings:
            raise RuntimeError(
                "No fraud findings returned. Investigate before presenting the demo."
            )

    print("\nDemo completed.")
    print("Temporary audit data was isolated and cleaned up.")
    print(
        "Note: synthetic scenario results do not establish "
        "real-world detection accuracy."
    )


if __name__ == "__main__":
    main()
