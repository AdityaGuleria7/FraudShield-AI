"""Build a small SQLite "warehouse" of fraud KPIs from the FraudShield pipeline.

Runs the full pipeline on synthetic transactions (with injected fraud
scenarios), then writes two tables to SQLite plus CSV exports that Power BI,
Tableau or Excel can open directly.

    PYTHONPATH=. python scripts/build_kpi_warehouse.py [--count 3000]
    sqlite3 warehouse/fraudshield_kpis.db < sql/kpis.sql

Synthetic data only: the numbers demonstrate the analytics layer, not
real-world detection performance.
"""
from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import tempfile
from pathlib import Path

from app.dashboard import analyze, plain
from app.data.transaction_generator import TransactionGenerator
from app.fraud.scenario_generator import FraudScenarioGenerator

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "warehouse" / "fraudshield_kpis.db"
EXPORT_DIR = ROOT / "warehouse" / "exports"

SCHEMA = """
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS findings;
CREATE TABLE transactions (
    transaction_id TEXT PRIMARY KEY, account_id TEXT, device_id TEXT,
    ip_address TEXT, country TEXT, merchant_category TEXT, amount REAL,
    ts_utc TEXT, hour_utc INTEGER, is_fraud INTEGER, fraud_scenario TEXT,
    flagged INTEGER
);
CREATE TABLE findings (
    finding_id INTEGER PRIMARY KEY AUTOINCREMENT, pattern TEXT, account_id TEXT,
    detector TEXT, evidence TEXT, transaction_id TEXT
);
"""


def _ids(finding: dict) -> list[str]:
    ids = finding.get("transaction_ids")
    if isinstance(ids, list):
        return [str(i) for i in ids]
    found = [
        str(finding[k])
        for k in ("transaction_id", "from_transaction_id", "to_transaction_id")
        if finding.get(k)
    ]
    return found


def build(count: int, seed: int) -> dict[str, int]:
    gen = TransactionGenerator(seed=seed)
    scen = FraudScenarioGenerator()
    txns = [plain(t) for t in gen.generate(count=count)]
    # Pin each ordinary account to one home country (the raw generator draws a
    # random country per transaction, which would trip travel rules on noise).
    homes = ["IN", "US", "GB", "SG"]
    for t in txns:
        t["country"] = homes[int(t["account_id"][-4:]) % len(homes)]
    # Inject several independent attacks so KPIs have something to measure.
    for n in range(8):
        txns = scen.inject_velocity_attack(txns, account_id=f"ACC9{n:03d}")
    for n in range(8):
        txns = scen.inject_new_device_high_value(txns, account_id=f"ACC8{n:03d}")
    # Impossible travel: same account, two distant countries 30 minutes apart.
    for n in range(6):
        acct = f"ACC7{n:03d}"
        for k, (country, minute) in enumerate((("US", "00"), ("SG", "30"))):
            txns.append(
                {
                    "transaction_id": f"FRAUD-TRAVEL-{n}-{k}",
                    "account_id": acct,
                    "device_id": f"DEV-TRAVEL-{n}",
                    "ip_address": "10.0.1.5",
                    "country": country,
                    "merchant_category": "TRAVEL",
                    "amount": 1800.0 + 100 * n,
                    "timestamp": f"2026-01-04T12:{minute}:00+00:00",
                    "is_fraud": True,
                    "fraud_scenario": "IMPOSSIBLE_TRAVEL",
                }
            )
    # injected scenario ids repeat per account; make them unique
    seen: dict[str, int] = {}
    rows = []
    for t in (plain(t) for t in txns):
        tid = t["transaction_id"]
        seen[tid] = seen.get(tid, 0) + 1
        if seen[tid] > 1:
            t["transaction_id"] = f"{tid}-{t['account_id']}"
        rows.append(t)

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["FRAUDSHIELD_AUDIT_DIR"] = tmp
        result = analyze(rows)

    findings = result.get("pattern_findings", [])
    flagged: set[str] = set()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    for f in findings:
        prov = f.get("provenance", {}) or {}
        for tid in _ids(f) or [None]:
            if tid:
                flagged.add(tid)
            con.execute(
                "INSERT INTO findings(pattern, account_id, detector, evidence, transaction_id)"
                " VALUES (?,?,?,?,?)",
                (f.get("pattern"), f.get("account_id"), prov.get("detector"), f.get("evidence"), tid),
            )
    for t in rows:
        hour = int(str(t["timestamp"])[11:13])
        con.execute(
            "INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                t["transaction_id"], t["account_id"], t["device_id"], t["ip_address"],
                t["country"], t["merchant_category"], float(t["amount"]), t["timestamp"],
                hour, int(bool(t.get("is_fraud"))), t.get("fraud_scenario"),
                int(t["transaction_id"] in flagged),
            ),
        )
    con.commit()

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    for table in ("transactions", "findings"):
        cur = con.execute(f"SELECT * FROM {table}")
        with open(EXPORT_DIR / f"{table}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([d[0] for d in cur.description])
            w.writerows(cur.fetchall())
    summary = {
        "transactions": con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0],
        "findings": con.execute("SELECT COUNT(*) FROM findings").fetchone()[0],
        "flagged": con.execute("SELECT COUNT(*) FROM transactions WHERE flagged=1").fetchone()[0],
    }
    con.close()
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    print(build(args.count, args.seed))
    print(f"Wrote {DB_PATH.relative_to(ROOT)} and CSV exports in {EXPORT_DIR.relative_to(ROOT)}/")
