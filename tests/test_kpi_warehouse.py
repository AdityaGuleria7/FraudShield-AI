import sqlite3
from pathlib import Path

from scripts import build_kpi_warehouse as wh

SQL_FILE = Path(__file__).resolve().parents[1] / "sql" / "kpis.sql"


def test_warehouse_builds_and_all_kpi_queries_run(tmp_path, monkeypatch):
    monkeypatch.setattr(wh, "DB_PATH", tmp_path / "kpi.db")
    monkeypatch.setattr(wh, "EXPORT_DIR", tmp_path / "exports")

    summary = wh.build(count=300, seed=1)
    assert summary["transactions"] > 300
    assert summary["flagged"] > 0
    assert (tmp_path / "exports" / "transactions.csv").exists()

    con = sqlite3.connect(tmp_path / "kpi.db")
    statements = [s.strip() for s in SQL_FILE.read_text().split(";") if "SELECT" in s.upper()]
    assert len(statements) >= 6
    for statement in statements:
        con.execute(statement).fetchall()  # raises if any query is invalid

    # Injected scenarios are detected and clean synthetic traffic is not flagged.
    missed = con.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud=1 AND flagged=0").fetchone()[0]
    false_pos = con.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud=0 AND flagged=1").fetchone()[0]
    assert missed == 0 and false_pos == 0
    con.close()
