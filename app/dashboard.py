
from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import re
import uuid
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
import secrets
import threading
import time
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse, JSONResponse

from app.audit.ledger import AuditLedger
from app.core.orchestrator import FraudShieldOrchestrator
from app.data.transaction_generator import TransactionGenerator
from app.fraud.scenario_generator import FraudScenarioGenerator
from app.ml.ieee_scorer import score_transactions


logger = logging.getLogger(__name__)


app = FastAPI(
    title="FraudShield AI",
    description=(
        "Evidence-driven, multi-agent fraud detection platform. "
        "Detects velocity attacks, impossible travel, account takeover, "
        "and new-device high-value transactions with full audit trails."
    ),
    version="2.0.0",
)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_TRANSACTIONS = 5000


security = HTTPBasic(auto_error=False)


def require_auth(
    credentials: HTTPBasicCredentials | None = Depends(security),
) -> str:
    """Require credentials configured through environment variables."""
    expected_username = os.environ.get("FRAUDSHIELD_AUTH_USERNAME", "")
    expected_password = os.environ.get("FRAUDSHIELD_AUTH_PASSWORD", "")

    if not expected_username or not expected_password:
        logger.error("Authentication is not configured; refusing access.")
        raise HTTPException(
            status_code=503,
            detail="Authentication is not configured on this server.",
        )

    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Basic"},
        )

    username_ok = secrets.compare_digest(
        credentials.username, expected_username
    )
    password_ok = secrets.compare_digest(
        credentials.password, expected_password
    )

    if not (username_ok and password_ok):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials.",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username

RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_CLIENTS = 10_000
_rate_limit_lock = threading.Lock()
_rate_limit_buckets: dict[str, tuple[int, float]] = {}


def require_rate_limit(
    request: Request,
    username: str = Depends(require_auth),
) -> str:
    """Limit authenticated requests per client IP using a fixed window."""
    client_ip = request.client.host if request.client else "unknown"
    key = f"{username}:{client_ip}"
    now = time.monotonic()

    with _rate_limit_lock:
        expired = [
            bucket_key
            for bucket_key, (_, started) in _rate_limit_buckets.items()
            if now - started >= RATE_LIMIT_WINDOW_SECONDS
        ]
        for bucket_key in expired:
            del _rate_limit_buckets[bucket_key]

        count, started = _rate_limit_buckets.get(key, (0, now))
        if count >= RATE_LIMIT_REQUESTS:
            retry_after = max(
                1, int(RATE_LIMIT_WINDOW_SECONDS - (now - started) + 0.999)
            )
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please retry later.",
                headers={"Retry-After": str(retry_after)},
            )

        if (
            key not in _rate_limit_buckets
            and len(_rate_limit_buckets) >= RATE_LIMIT_MAX_CLIENTS
        ):
            raise HTTPException(
                status_code=503,
                detail="Rate limiter is temporarily at capacity.",
            )

        _rate_limit_buckets[key] = (count + 1, started)

    return username


REQUIRED_FIELDS = {
    "transaction_id",
    "account_id",
    "device_id",
    "ip_address",
    "country",
    "merchant_category",
    "amount",
    "timestamp",
}

# is_fraud is optional: ground-truth labels are not available at inference time.
# When absent it defaults to False. When present it enables evaluation workflows.
OPTIONAL_LABEL_FIELD = "is_fraud"


def plain(value: Any) -> Any:
    """Convert pipeline results into JSON-compatible values."""
    if is_dataclass(value) and not isinstance(value, type):
        return plain(asdict(value))

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [plain(item) for item in value]

    if hasattr(value, "model_dump"):
        return plain(value.model_dump())

    if hasattr(value, "dict") and callable(value.dict):
        return plain(value.dict())

    if hasattr(value, "__dict__") and not isinstance(
        value, (str, int, float, bool)
    ):
        return plain(
            {
                k: v
                for k, v in vars(value).items()
                if not k.startswith("_")
            }
        )

    if isinstance(value, Path):
        return str(value)

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def parse_csv(contents: bytes) -> list[dict[str, Any]]:
    """Validate the uploaded CSV structure and convert transaction fields."""
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="The CSV file must be no larger than 5 MB.",
        )

    try:
        text = contents.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="The CSV must use UTF-8 encoding.",
        ) from exc

    try:
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if not reader.fieldnames:
            raise HTTPException(
                status_code=400,
                detail="The CSV file is empty or has no header row.",
            )

        normalized_headers = [
            header.strip() if header is not None else ""
            for header in reader.fieldnames
        ]
        if any(not header for header in normalized_headers):
            raise HTTPException(
                status_code=400,
                detail="CSV column headers must not be blank.",
            )
        if len(set(normalized_headers)) != len(normalized_headers):
            raise HTTPException(
                status_code=400,
                detail="CSV column headers must be unique.",
            )

        reader.fieldnames = normalized_headers
        missing = REQUIRED_FIELDS - set(normalized_headers)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Missing required CSV columns: "
                    f"{', '.join(sorted(missing))}"
                ),
            )

        transactions = []
        for row_number, raw_row in enumerate(reader, start=2):
            if raw_row is None or all(
                value is None or not str(value).strip()
                for value in raw_row.values()
            ):
                continue

            if None in raw_row or any(
                value is None for value in raw_row.values()
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Incorrect number of fields on CSV row {row_number}."
                    ),
                )

            if len(transactions) >= MAX_TRANSACTIONS:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "A maximum of 5,000 transactions is allowed per upload."
                    ),
                )

            row = {
                str(key).strip(): (
                    value.strip() if isinstance(value, str) else value
                )
                for key, value in raw_row.items()
                if key is not None
            }

            try:
                amount = float(row["amount"])
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid amount on CSV row {row_number}.",
                ) from exc

            if not math.isfinite(amount):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid amount on CSV row {row_number}.",
                )

            if amount < 0 or amount > 1_000_000:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Amount out of permitted range on CSV row {row_number}."
                    ),
                )

            fraud_raw = row.get(OPTIONAL_LABEL_FIELD, "false")
            fraud_value = str(fraud_raw).strip().lower()
            if fraud_value in {"true", "1", "yes"}:
                is_fraud = True
            elif fraud_value in {"false", "0", "no"}:
                is_fraud = False
            else:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid is_fraud value on CSV row {row_number}. "
                        "Use true/false, yes/no, or 1/0."
                    ),
                )

            row["amount"] = amount
            row["is_fraud"] = is_fraud
            transactions.append(row)

    except csv.Error as exc:
        raise HTTPException(
            status_code=400,
            detail="The CSV file contains malformed CSV data.",
        ) from exc

    if not transactions:
        raise HTTPException(
            status_code=400,
            detail="The CSV file contains no transactions.",
        )

    return transactions

def analyze(
    transactions: list[dict[str, Any]],
    demo_notice: bool = False,
) -> dict[str, Any]:
    """Run the FraudShield pipeline and retain a separate audit trail per run."""
    audit_dir = Path(
        os.environ.get("FRAUDSHIELD_AUDIT_DIR", "var/audit")
    )
    audit_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    task_id = f"FRAUDSHIELD-{uuid.uuid4().hex.upper()}"
    audit_path = audit_dir / f"{task_id}.jsonl"

    orchestrator = FraudShieldOrchestrator(audit_path=str(audit_path))
    result = orchestrator.run(
        task_id=task_id,
        data={"transactions": transactions},
    )

    ledger = AuditLedger(str(audit_path))

    try:
        chain_valid = ledger.verify_chain()
        if isinstance(chain_valid, dict):
            chain_valid = chain_valid.get("valid", False)
    except (AttributeError, TypeError, ValueError):
        chain_valid = None

    try:
        audit_events = sum(
            1
            for line in audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ) if audit_path.exists() else 0
    except OSError:
        audit_events = 0

    response = plain(result)

    if not isinstance(response, dict):
        response = {"pipeline_result": response}

    response["transaction_count"] = len(transactions)
    response["audit_integrity"] = {
        "valid": chain_valid,
        "events": audit_events,
    }
    response["demo_notice"] = demo_notice

    return response


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "application": "FraudShield AI",
        "version": app.version,
        "description": "Evidence-driven multi-agent fraud detection platform",
    }


@app.get("/", response_class=HTMLResponse)
def dashboard(username: str = Depends(require_auth)) -> str:
    return HTML


@app.get("/api/demo")
def demo(username: str = Depends(require_rate_limit)) -> JSONResponse:
    try:
        generator = TransactionGenerator(seed=42)
        transactions = generator.generate(count=10)

        scenario_generator = FraudScenarioGenerator()
        transactions = scenario_generator.inject_velocity_attack(
            transactions
        )

        result = analyze(
            [plain(transaction) for transaction in transactions],
            demo_notice=True,
        )

        return JSONResponse(content=result)

    except Exception as exc:
        logger.exception("Demo pipeline failed")
        raise HTTPException(
            status_code=500,
            detail="Demo pipeline failed. Check server logs for details.",
        ) from exc


@app.post("/api/analyze-csv")
async def analyze_csv(
    username: str = Depends(require_rate_limit),
    file: UploadFile = File(...),
) -> JSONResponse:
    filename = file.filename or ""

    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a .csv file.",
        )

    try:
        contents = await file.read(MAX_UPLOAD_BYTES + 1)

        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="The CSV file must be no larger than 5 MB.",
            )

        transactions = parse_csv(contents)
        result = analyze(transactions)

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("CSV analysis failed")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Check server logs for details.",
        ) from exc
    finally:
        await file.close()


@app.post("/api/ieee-score-csv")
async def ieee_score_csv(
    username: str = Depends(require_rate_limit),
    file: UploadFile = File(...),
) -> JSONResponse:
    """Score compatible IEEE-CIS feature rows from a CSV upload."""
    filename = file.filename or ""

    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a .csv file.",
        )

    try:
        contents = await file.read(MAX_UPLOAD_BYTES + 1)

        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="The CSV file must be no larger than 5 MB.",
            )

        if not contents:
            raise HTTPException(status_code=422, detail="The CSV file is empty.")

        try:
            decoded = contents.decode("utf-8-sig")
            csv_reader = csv.reader(io.StringIO(decoded), strict=True)
            header = next(csv_reader, [])
            normalized_header = [str(name).strip() for name in header]
            if not normalized_header or any(not name for name in normalized_header):
                raise HTTPException(
                    status_code=422,
                    detail="The CSV must have non-empty column names.",
                )
            if len(normalized_header) != len(set(normalized_header)):
                raise HTTPException(
                    status_code=422,
                    detail="The CSV contains duplicate column names.",
                )

            for row_number, row in enumerate(csv_reader, start=1):
                # Pandas ignores blank lines, so allow them here too.
                if not row:
                    continue
                if len(row) != len(normalized_header):
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"Row {row_number} has {len(row)} fields; "
                            f"expected {len(normalized_header)}."
                        ),
                    )

            import pandas as pd

            frame = pd.read_csv(io.StringIO(decoded))
            frame.columns = [str(name).strip() for name in frame.columns]
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail="Could not parse the uploaded CSV. Use a valid UTF-8 CSV file.",
            ) from exc

        if frame.empty:
            raise HTTPException(
                status_code=422,
                detail="The CSV must contain at least one transaction row.",
            )

        if len(frame) > MAX_TRANSACTIONS:
            raise HTTPException(
                status_code=413,
                detail=f"A maximum of {MAX_TRANSACTIONS} rows is allowed.",
            )

        if frame.columns.duplicated().any():
            raise HTTPException(
                status_code=422,
                detail="The CSV contains duplicate column names.",
            )

        # Validate required CSV columns together.
        required_columns = {"TransactionAmt", "identity_available"}
        missing_columns = sorted(required_columns - set(frame.columns))

        if missing_columns:
            raise HTTPException(
                status_code=422,
                detail=(
                    "CSV is missing required columns: "
                    + ", ".join(missing_columns)
                    + ". identity_available must be 0 or 1; it indicates whether identity data is available and is a model input, not proof of fraud."
                ),
            )

        # Cast to object before replacing nulls; float columns otherwise retain NaN.
        records = frame.astype(object).where(frame.notna(), None).to_dict(
            orient="records"
        )
        results = []
        batch_result = None

        for start in range(0, len(records), 100):
            batch = records[start:start + 100]
            try:
                batch_result = score_transactions(batch)
            except ValueError as exc:
                detail = re.sub(
                    r"\bRow (\d+)\b",
                    lambda match: f"Row {start + int(match.group(1))}",
                    str(exc),
                )
                raise HTTPException(status_code=422, detail=detail) from exc

            # Keep row numbers unique and aligned to the uploaded CSV across batches.
            for result in batch_result["results"]:
                result["row"] = start + int(result["row"])
            results.extend(batch_result["results"])

        if batch_result is None:
            raise HTTPException(
                status_code=422,
                detail="The CSV must contain at least one transaction row.",
            )

        return JSONResponse(content={
            "model_type": batch_result["model_type"],
            "status": batch_result["status"],
            "scored_count": len(results),
            "training_rows": batch_result.get("training_rows"),
            "training_fraud_rate": batch_result.get("training_fraud_rate"),
            "threshold": batch_result.get("threshold"),
            "score_semantics": batch_result["score_semantics"],
            "validation_note": batch_result.get("validation_note"),
            "decision_made": False,
            "results": results,
            "warning": batch_result["warning"],
        })

    except HTTPException:
        raise
    except FileNotFoundError as exc:
        logger.exception("IEEE-CIS model artifact unavailable")
        raise HTTPException(
            status_code=503,
            detail="IEEE-CIS model artifact unavailable.",
        ) from exc
    except Exception as exc:
        logger.exception("IEEE-CIS CSV scoring failed")
        raise HTTPException(
            status_code=500,
            detail="IEEE-CIS CSV scoring failed. Check server logs for details.",
        ) from exc
    finally:
        await file.close()


@app.post("/api/ieee-score")
def ieee_score(
    payload: dict[str, Any],
    username: str = Depends(require_rate_limit),
) -> JSONResponse:
    """Return experimental IEEE-CIS scores without making fraud decisions."""
    rows = payload.get("transactions") if isinstance(payload, dict) else None

    if not isinstance(rows, list):
        raise HTTPException(
            status_code=400,
            detail="Request must contain a 'transactions' array.",
        )

    explain = bool(payload.get("explain", True)) if isinstance(payload, dict) else True

    try:
        result = score_transactions(rows, explain=explain)
        return JSONResponse(content=result)
    except FileNotFoundError as exc:
        logger.exception("IEEE-CIS model artifact unavailable")
        raise HTTPException(
            status_code=503,
            detail="IEEE-CIS model artifact unavailable.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("IEEE-CIS scoring failed")
        raise HTTPException(
            status_code=500,
            detail="IEEE-CIS scoring failed. Check server logs for details.",
        ) from exc


HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FraudShield AI | Fraud Intelligence Platform</title>
<style>
:root {
  color-scheme: dark;
  --bg: #07090f;
  --surface: #0d1117;
  --panel: #111827;
  --panel2: #161e2e;
  --border: #1f2d42;
  --border2: #253147;
  --text: #e8edf8;
  --muted: #7a8fa8;
  --muted2: #566880;
  --accent: #6b8fff;
  --accent2: #4a6fe8;
  --green: #3dd68c;
  --green-bg: #0d2a1f;
  --red: #f87171;
  --red-bg: #2a0d0d;
  --yellow: #fbbf24;
  --yellow-bg: #2a1f0d;
  --orange: #fb923c;
  --orange-bg: #2a170d;
  --purple: #a78bfa;
  --radius: 14px;
  --radius-sm: 8px;
}
*, *::before, *::after { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  -webkit-font-smoothing: antialiased;
}
button, input, select { font: inherit; }
a { color: var(--accent); text-decoration: none; }

/* ── Layout ──────────────────────────────────────────────── */
.shell { max-width: 1280px; margin: 0 auto; padding: 24px 28px 48px; }

/* ── Header ──────────────────────────────────────────────── */
.top-bar {
  display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 12px; margin-bottom: 32px;
  padding-bottom: 18px; border-bottom: 1px solid var(--border);
}
.brand { display: flex; align-items: center; gap: 14px; }
.logo-wrap {
  display: grid; place-items: center; width: 44px; height: 44px;
  border-radius: 12px; background: linear-gradient(135deg,#1a2a5e,#2a1a5e);
  color: var(--accent); font-size: 20px; font-weight: 900; letter-spacing: -1px;
  box-shadow: 0 0 0 1px var(--border2);
}
.brand-name { font-size: 18px; font-weight: 700; letter-spacing: -.4px; }
.brand-tag { font-size: 11px; color: var(--muted); letter-spacing: .8px; text-transform: uppercase; margin-top: 2px; }
.top-bar-right { display: flex; align-items: center; gap: 16px; }
.status-dot {
  display: inline-flex; align-items: center; gap: 7px;
  font-size: 12px; color: var(--muted);
}
.dot {
  width: 7px; height: 7px; border-radius: 50%; background: var(--muted2);
  transition: background .4s;
}
.dot.online { background: var(--green); box-shadow: 0 0 6px var(--green); }
.ver-badge {
  font-size: 11px; padding: 3px 9px; border-radius: 20px;
  background: var(--panel2); border: 1px solid var(--border2);
  color: var(--muted); font-weight: 600;
}

/* ── Hero ────────────────────────────────────────────────── */
.hero {
  background: linear-gradient(135deg, #0f1a30 0%, #0d1117 60%, #0f0d1f 100%);
  border: 1px solid var(--border2); border-radius: var(--radius);
  padding: 32px 36px; margin-bottom: 24px; position: relative; overflow: hidden;
}
.hero::before {
  content: ""; position: absolute; top: -60px; right: -60px;
  width: 260px; height: 260px; border-radius: 50%;
  background: radial-gradient(circle, rgba(107,143,255,.08) 0%, transparent 70%);
  pointer-events: none;
}
.eyebrow {
  font-size: 10.5px; font-weight: 800; letter-spacing: 2px;
  text-transform: uppercase; color: var(--accent); margin-bottom: 10px;
}
.hero-title { font-size: 28px; font-weight: 800; margin: 0 0 10px; letter-spacing: -.6px; line-height: 1.25; }
.hero-sub { color: var(--muted); max-width: 680px; margin: 0 0 24px; line-height: 1.6; }
.btn-row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
button {
  cursor: pointer; border: 1px solid var(--border2);
  border-radius: var(--radius-sm); padding: 10px 18px;
  background: var(--panel2); color: var(--text);
  font-weight: 600; transition: all .15s;
}
button:hover:not(:disabled) { border-color: var(--accent); background: #1a2540; }
button.primary {
  background: var(--accent); color: #040812;
  border-color: transparent; font-weight: 700;
}
button.primary:hover:not(:disabled) { background: var(--accent2); }
button:disabled { opacity: .5; cursor: wait; }
.disclaimer { font-size: 11.5px; color: var(--muted2); margin-top: 14px; }

/* ── Section panel ───────────────────────────────────────── */
.section-panel {
  background: var(--panel); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 22px 24px; margin-bottom: 16px;
}
.section-panel h3 {
  font-size: 13px; font-weight: 700; margin: 0 0 18px;
  color: var(--text); letter-spacing: .1px;
}
.section-header {
  display: flex; justify-content: space-between; align-items: center;
  flex-wrap: wrap; gap: 10px; margin-bottom: 18px;
}
.section-header h3 { margin: 0; }

/* ── Metrics bar ─────────────────────────────────────────── */
.metrics-bar {
  display: grid; gap: 14px; margin-bottom: 16px;
  grid-template-columns: repeat(4, minmax(0,1fr));
}
.metric-card {
  background: var(--panel); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 18px 20px;
}
.metric-label { font-size: 11px; font-weight: 600; color: var(--muted); letter-spacing: .6px; text-transform: uppercase; margin-bottom: 6px; }
.metric-value { font-size: 28px; font-weight: 800; letter-spacing: -1px; line-height: 1; }
.metric-sub { font-size: 11px; color: var(--muted2); margin-top: 5px; }

/* ── Risk badge ──────────────────────────────────────────── */
.risk-badge {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 11px; font-weight: 700; letter-spacing: .8px;
  text-transform: uppercase; padding: 4px 10px; border-radius: 20px;
}
.risk-PATTERN { background: var(--panel2); color: var(--muted); border: 1px solid var(--border2); }

/* ── Executive summary card ──────────────────────────────── */
.exec-summary {
  background: linear-gradient(135deg,#0f1d32,#0d1117);
  border: 1px solid var(--border2); border-radius: var(--radius);
  padding: 22px 26px; margin-bottom: 16px;
}
.exec-summary .eyebrow { margin-bottom: 8px; }
.exec-summary p { margin: 0; line-height: 1.7; color: var(--text); font-size: 14.5px; }

/* ── Explanation cards ───────────────────────────────────── */
.findings-grid { display: grid; gap: 14px; }
.finding-card {
  background: var(--panel2); border: 1px solid var(--border);
  border-radius: var(--radius); overflow: hidden;
}
.finding-card-header {
  display: flex; justify-content: space-between; align-items: flex-start;
  gap: 12px; padding: 16px 18px 12px; border-bottom: 1px solid var(--border);
}
.finding-card-title { font-weight: 700; font-size: 14px; margin-bottom: 3px; }
.finding-card-summary { font-size: 13px; color: var(--muted); margin-bottom: 0; line-height: 1.5; }
.finding-card-body { padding: 14px 18px; }
.finding-context {
  font-size: 12.5px; color: var(--muted); background: var(--surface);
  border-radius: var(--radius-sm); padding: 10px 14px; margin-bottom: 12px;
  border-left: 3px solid var(--accent);
}
.key-facts { list-style: none; padding: 0; margin: 0 0 10px; display: grid; gap: 5px; }
.key-facts li {
  font-size: 12.5px; display: flex; align-items: flex-start; gap: 8px;
}
.key-facts li::before { content: "→"; color: var(--muted2); flex-shrink: 0; }
.risk-contribution {
  font-size: 11.5px; color: var(--muted2); border-top: 1px solid var(--border);
  padding-top: 10px; margin-top: 2px;
}

/* ── Two-column grid ─────────────────────────────────────── */
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 16px; }

/* ── Detail rows ─────────────────────────────────────────── */
.detail-row {
  display: flex; justify-content: space-between; align-items: flex-start;
  gap: 14px; padding: 8px 0; border-bottom: 1px solid var(--border);
  font-size: 13px;
}
.detail-row:last-child { border-bottom: 0; }
.detail-key { color: var(--muted); white-space: nowrap; }
.detail-val { font-weight: 600; text-align: right; overflow-wrap: anywhere; }

/* ── IEEE scoring table ──────────────────────────────────── */
.score-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.score-table th {
  text-align: left; padding: 8px 12px; color: var(--muted);
  font-size: 11px; font-weight: 700; letter-spacing: .5px; text-transform: uppercase;
  border-bottom: 1px solid var(--border);
}
.score-table td { padding: 9px 12px; border-bottom: 1px solid var(--border); }
.score-table tr:last-child td { border-bottom: 0; }
.score-table tr:hover td { background: var(--surface); }

/* ── Collapsible details ─────────────────────────────────── */
details.adv-panel {
  background: var(--panel); border: 1px solid var(--border);
  border-radius: var(--radius); margin-bottom: 16px;
}
details.adv-panel > summary {
  cursor: pointer; padding: 18px 22px; font-weight: 700; font-size: 13px;
  list-style: none; display: flex; justify-content: space-between; align-items: center;
  user-select: none;
}
details.adv-panel > summary::-webkit-details-marker { display: none; }
details.adv-panel > summary::after { content: "+"; color: var(--muted); font-size: 20px; font-weight: 400; }
details.adv-panel[open] > summary::after { content: "−"; }
details.adv-panel > summary:hover { color: var(--accent); }
.adv-body { padding: 0 22px 22px; }

/* ── Upload row ──────────────────────────────────────────── */
.upload-row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
input[type="file"] {
  background: var(--surface); border: 1px dashed var(--border2);
  border-radius: var(--radius-sm); padding: 8px 12px; color: var(--text);
}
.status-msg { font-size: 12.5px; color: var(--muted); margin-top: 8px; min-height: 18px; }

/* ── Error box ───────────────────────────────────────────── */
.error-box {
  display: none; padding: 12px 16px; border-radius: var(--radius-sm);
  background: var(--red-bg); border: 1px solid rgba(248,113,113,.3);
  color: var(--red); font-size: 13px; margin-bottom: 16px; overflow-wrap: anywhere;
}

/* ── Raw JSON ────────────────────────────────────────────── */
pre {
  white-space: pre-wrap; overflow-wrap: anywhere; max-height: 500px;
  overflow: auto; padding: 16px; border-radius: var(--radius-sm);
  background: #040810; color: #8ba7d4; font-size: 12px; font-family: "SF Mono", "Fira Code", monospace;
  border: 1px solid var(--border); margin: 0;
}

/* ── Tag ─────────────────────────────────────────────────── */
.tag {
  display: inline-block; border-radius: 5px; padding: 2px 8px;
  font-size: 11px; font-weight: 600; background: var(--panel2);
  border: 1px solid var(--border2); color: var(--muted);
}
.tag.ok { background: var(--green-bg); color: var(--green); border-color: transparent; }
.tag.warn { background: var(--yellow-bg); color: var(--yellow); border-color: transparent; }
.tag.bad { background: var(--red-bg); color: var(--red); border-color: transparent; }

/* ── Footer ──────────────────────────────────────────────── */
.site-footer {
  margin-top: 32px; padding-top: 18px; border-top: 1px solid var(--border);
  font-size: 11.5px; color: var(--muted2); display: flex;
  justify-content: space-between; flex-wrap: wrap; gap: 8px;
}

/* ── Hidden ──────────────────────────────────────────────── */
#results { display: none; }
[hidden] { display: none !important; }

/* ── Responsive ──────────────────────────────────────────── */
@media (max-width: 900px) {
  .metrics-bar { grid-template-columns: repeat(2, 1fr); }
  .two-col { grid-template-columns: 1fr; }
}
@media (max-width: 540px) {
  .shell { padding: 15px; }
  .hero { padding: 22px 18px; }
  .hero-title { font-size: 22px; }
  .metrics-bar { grid-template-columns: 1fr 1fr; gap: 10px; }
  .metric-value { font-size: 22px; }
}
</style>
</head>
<body>
<div class="shell">

  <!-- ── TOP BAR ────────────────────────────────────────── -->
  <div class="top-bar">
    <div class="brand">
      <div class="logo-wrap">FS</div>
      <div>
        <div class="brand-name">FraudShield AI</div>
        <div class="brand-tag">Multi-Agent Fraud Intelligence Platform</div>
      </div>
    </div>
    <div class="top-bar-right">
      <div class="status-dot">
        <div class="dot" id="statusDot"></div>
        <span id="statusText">Checking...</span>
      </div>
      <div class="ver-badge">v2.0.0</div>
    </div>
  </div>

  <!-- ── HERO ───────────────────────────────────────────── -->
  <section class="hero">
    <div class="eyebrow">Evidence-driven fraud detection</div>
    <h2 class="hero-title">Detect. Explain. Investigate.</h2>
    <p class="hero-sub">
      FraudShield AI combines deterministic fraud pattern detection, machine learning,
      and multi-agent adversarial review into a single auditable pipeline —
      explaining every flagged transaction in plain language.
    </p>
    <div class="btn-row">
      <button class="primary" id="demoBtn" onclick="runDemo()">▶ Run Synthetic Demo</button>
      <button id="scrollToBatch" onclick="document.getElementById('ieeeCsvScoringPanel').scrollIntoView({behavior:'smooth'})">📁 Score IEEE-CIS CSV</button>
    </div>
    <p class="disclaimer">
      Experimental model scores are not calibrated fraud probabilities and do not constitute
      approved fraud decisions. Use only data you are authorized to process.
    </p>
  </section>

  <!-- ── BATCH CSV SCORING ──────────────────────────────── -->
  <section class="section-panel" id="ieeeCsvScoringPanel">
    <div class="section-header">
      <div>
        <div class="eyebrow" style="margin-bottom:4px">IEEE-CIS Batch Scoring</div>
        <h3>Upload Transaction CSV</h3>
      </div>
      <button id="batchDownloadBtn" onclick="downloadIeeeCsv()" disabled>⬇ Download Results</button>
    </div>
    <p style="color:var(--muted);font-size:13px;margin:0 0 16px">
      Upload up to 5,000 rows of compatible IEEE-CIS feature data.
      Required columns: <code>TransactionAmt</code>, <code>identity_available</code> (0 or 1; indicates identity-data availability and is used as a model input, not proof of fraud).
      Maximum file size: 5 MB.
    </p>
    <div class="upload-row">
      <input id="ieeeCsvFile" type="file" accept=".csv,text/csv">
      <button id="batchBtn" onclick="runIeeeCsv()">Score CSV</button>
    </div>
    <div class="status-msg" id="batchStatus"></div>

    <div id="batchResults" style="display:none;margin-top:20px">
      <div class="metrics-bar" style="grid-template-columns:repeat(2,minmax(0,1fr));margin-bottom:16px">
        <div class="metric-card">
          <div class="metric-label">Rows scored</div>
          <div class="metric-value" id="batchCount">0</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Model status</div>
          <div style="margin-top:8px"><span class="tag warn" id="batchModelStatus">EXPERIMENTAL</span></div>
          <div class="metric-sub" id="batchSemantics"></div>
        </div>
      </div>
      <div style="overflow-x:auto">
        <table class="score-table">
          <thead>
            <tr>
              <th>Row</th>
              <th>Transaction ID</th>
              <th>Experimental score</th>
              <th>Input coverage</th>
              <th>Decision status</th>
            </tr>
          </thead>
          <tbody id="batchTable"></tbody>
        </table>
      </div>
      <p class="status-msg" id="batchWarning" style="margin-top:10px"></p>
      <p id="batchCoverageHelp" style="color:var(--muted);font-size:13px;margin-top:8px">
        Missing model inputs may affect score interpretation. More available inputs do not guarantee greater accuracy.
        Treat input coverage as context for investigation, not as a fraud verdict.
      </p>
    </div>
  </section>

  <!-- ── ADVANCED: SINGLE TRANSACTION ──────────────────── -->
  <details class="adv-panel" id="ieeeScoringPanel">
    <summary>Advanced: score a single transaction</summary>
    <div class="adv-body">
      <p style="color:var(--muted);font-size:13px;margin-bottom:16px">
        Manual scoring is a sparse-input demonstration. The enriched model expects
        additional transaction and identity/device features that this form does not collect.
        Missing optional inputs are handled by preprocessing and may affect the score.
        Choosing “Available” only sets the identity_available flag; it does not supply
        identity/device details. This experimental score is not a calibrated fraud
        probability or a fraud decision.
      </p>
      <form id="ieeeScoreForm" onsubmit="runIeeeScore(event)" style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px">
        <label>Transaction Amount *<br>
          <input id="ieeeAmount" type="number" min="0" step="any" required placeholder="125.50" style="width:100%;margin-top:4px;padding:8px 10px;background:var(--surface);border:1px solid var(--border2);border-radius:var(--radius-sm);color:var(--text)">
        </label>
        <label>Identity Data Available *<br>
          <select id="ieeeIdentityAvailable" required style="width:100%;margin-top:4px;padding:8px 10px;background:var(--surface);border:1px solid var(--border2);border-radius:var(--radius-sm);color:var(--text)">
            <option value="">Choose one</option>
            <option value="0">Unavailable (0)</option>
            <option value="1">Available (1)</option>
          </select>
        </label>
        <label>Transaction ID (optional)<br>
          <input id="ieeeTransactionId" type="text" placeholder="999001" style="width:100%;margin-top:4px;padding:8px 10px;background:var(--surface);border:1px solid var(--border2);border-radius:var(--radius-sm);color:var(--text)">
        </label>
        <label>card1 (optional)<br>
          <input id="ieeeCard1" type="text" placeholder="12345" style="width:100%;margin-top:4px;padding:8px 10px;background:var(--surface);border:1px solid var(--border2);border-radius:var(--radius-sm);color:var(--text)">
        </label>
        <label>card4 (optional)<br>
          <input id="ieeeCard4" type="text" placeholder="visa" style="width:100%;margin-top:4px;padding:8px 10px;background:var(--surface);border:1px solid var(--border2);border-radius:var(--radius-sm);color:var(--text)">
        </label>
        <label>ProductCD (optional)<br>
          <input id="ieeeProductCD" type="text" placeholder="W" style="width:100%;margin-top:4px;padding:8px 10px;background:var(--surface);border:1px solid var(--border2);border-radius:var(--radius-sm);color:var(--text)">
        </label>
        <div style="grid-column:1/-1">
          <button class="primary" id="ieeeScoreButton" type="submit">Score Transaction</button>
        </div>
      </form>
      <div id="ieeeError" class="error-box" role="alert"></div>
      <section id="ieeeScoreResult" style="display:none" aria-live="polite">
        <div class="detail-row"><span class="detail-key">Model status</span><strong class="detail-val" id="ieeeModelStatus">—</strong></div>
        <div class="detail-row"><span class="detail-key">Experimental score</span><strong class="detail-val" id="ieeeScoreValue">—</strong></div>
        <p class="status-msg">Treat this score as an investigation signal only. It does not confirm fraud and is not a calibrated fraud probability.</p>
        <div class="detail-row"><span class="detail-key">Input coverage</span><strong class="detail-val" id="ieeeInputCoverage">—</strong></div>
        <div class="detail-row"><span class="detail-key">Transaction ID</span><strong class="detail-val" id="ieeeScoreTransactionId">—</strong></div>
        <div id="ieeeShapDrivers" style="margin-top:14px;display:none">
          <div style="font-weight:700;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;margin-bottom:8px">Top model-score contributions</div>
          <div id="ieeeShapDriversList" style="display:grid;gap:6px"></div>
          <p class="status-msg" style="margin-top:8px">Positive values push the model score higher; negative values push it lower. A zero value means this feature contributes no direction to the model score. These contributions are not fraud probabilities and do not prove fraud.</p>
        </div>
        <p id="ieeeExplanationStatus" class="status-msg" style="margin-top:8px;display:none"></p>
        <p id="ieeeScoreSemantics" class="status-msg" style="margin-top:8px"></p>
        <p id="ieeeScoreWarning" class="status-msg"></p>
      </section>
    </div>
  </details>

  <!-- ── ADVANCED: CASE CSV ─────────────────────────────── -->
  <details class="adv-panel" id="caseCsvPanel">
    <summary>Advanced — Analyze a FraudShield Case CSV</summary>
    <div class="adv-body">
      <p style="color:var(--muted);font-size:13px;margin-bottom:14px">
        Upload transactions in FraudShield's schema. Required fields:
        <code>transaction_id, account_id, device_id, ip_address, country, merchant_category, amount, timestamp</code>.
        The <code>is_fraud</code> label is optional (for evaluation mode).
        See <code>examples/</code> for ready-to-use sample files.
      </p>
      <div class="upload-row">
        <input id="csvFile" type="file" accept=".csv,text/csv">
        <button id="uploadButton" onclick="runCsv()">Analyze Case CSV</button>
      </div>
    </div>
  </details>

  <!-- ── ERROR ──────────────────────────────────────────── -->
  <div class="error-box" id="error"></div>

  <!-- ── RESULTS ────────────────────────────────────────── -->
  <section id="results">

    <!-- Metrics bar -->
    <div class="metrics-bar">
      <div class="metric-card">
        <div class="metric-label">Transactions</div>
        <div class="metric-value" id="metricTxn">—</div>
        <div class="metric-sub">Records analyzed</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Fraud Findings</div>
        <div class="metric-value" id="metricFindings">—</div>
        <div class="metric-sub">Patterns detected</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Risk Score</div>
        <div class="metric-value" id="metricRisk">—</div>
        <div class="metric-sub" id="metricRiskLevel">—</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Audit Integrity</div>
        <div class="metric-value" id="metricAudit" style="font-size:20px">—</div>
        <div class="metric-sub" id="metricAuditEvents">—</div>
      </div>
    </div>

    <!-- Executive summary -->
    <div class="exec-summary" id="execSummaryCard" style="display:none">
      <div class="eyebrow">Investigation Summary</div>
      <p id="execSummaryText"></p>
    </div>

    <!-- Fraud findings (explanation cards) -->
    <div class="section-panel">
      <h3>Fraud Findings &amp; Explanations</h3>
      <div id="findingsGrid" class="findings-grid">
        <div style="color:var(--muted);font-size:13px">Run a demo or upload a CSV to see findings.</div>
      </div>
    </div>

    <!-- Two-column: Quality + Case -->
    <div class="two-col">
      <div class="section-panel">
        <h3>Data Quality</h3>
        <div id="qualityDetails"></div>
      </div>
      <div class="section-panel">
        <h3>Investigation Case</h3>
        <div id="caseDetails"></div>
      </div>
    </div>

    <!-- Two-column: Evidence + Agent Reviews -->
    <div class="two-col">
      <div class="section-panel">
        <h3>Evidence Assessment</h3>
        <div id="evidenceDetails"></div>
      </div>
      <div class="section-panel">
        <h3>Agent Reviews</h3>
        <div id="reviewDetails"></div>
      </div>
    </div>

    <!-- Raw JSON -->
    <details class="adv-panel">
      <summary>Raw Pipeline Response (JSON)</summary>
      <div class="adv-body" style="padding-top:0">
        <pre id="rawJson"></pre>
      </div>
    </details>

  </section>

  <!-- ── FOOTER ─────────────────────────────────────────── -->
  <footer class="site-footer">
    <span>FraudShield AI &nbsp;·&nbsp; Evidence-driven fraud detection &nbsp;·&nbsp; v2.0.0</span>
    <span>Experimental scores do not constitute approved fraud decisions.</span>
  </footer>

</div><!-- /.shell -->

<script>
// ── Utilities ────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const esc = v => { const n = document.createElement('span'); n.textContent = String(v ?? '—'); return n.innerHTML; };
const pretty = v => v == null ? '—' : typeof v === 'object' ? JSON.stringify(v) : String(v);

function pick(data, keys) {
  for (const k of keys) if (data?.[k] != null) return data[k];
  return null;
}

// ── Render detail rows ───────────────────────────────────────
function renderDetails(obj) {
  if (!obj || typeof obj !== 'object') return `<div style="color:var(--muted);font-size:13px">${esc(pretty(obj))}</div>`;
  const entries = Array.isArray(obj)
    ? obj.map((v, i) => [String(i + 1), v])
    : Object.entries(obj);
  if (!entries.length) return '<div style="color:var(--muted);font-size:13px">No data returned.</div>';
  return entries.map(([k, v]) => `
    <div class="detail-row">
      <span class="detail-key">${esc(String(k).replaceAll('_', ' '))}</span>
      <strong class="detail-val">${esc(pretty(v))}</strong>
    </div>`).join('');
}

// ── Render explanation finding card ─────────────────────────
function riskColor(level) {
  return {CRITICAL:'var(--red)',HIGH:'var(--orange)',MEDIUM:'var(--yellow)',LOW:'var(--green)'}[level] || 'var(--muted)';
}

function renderFindingCard(finding) {
  const pattern = finding.pattern || 'UNKNOWN';
  const expl = finding.explanation || {};
  const headline = expl.headline || pattern.replaceAll('_', ' ');
  const summary = expl.plain_summary || finding.evidence || '';
  const context = expl.what_this_means || '';
  const facts = expl.key_facts || [];
  const contribution = expl.risk_contribution || '';

  // A detected pattern is an investigation signal, not a severity verdict.
  const badgeLabel = 'PATTERN DETECTED';

  const factsHtml = facts.length
    ? `<ul class="key-facts">${facts.map(f => `<li>${esc(f)}</li>`).join('')}</ul>`
    : '';

  const contextHtml = context
    ? `<div class="finding-context">${esc(context)}</div>`
    : '';

  const contribHtml = contribution
    ? `<div class="risk-contribution">⚡ ${esc(contribution)}</div>`
    : '';

  return `
    <div class="finding-card">
      <div class="finding-card-header">
        <div>
          <div class="finding-card-title">${esc(headline)}</div>
          <div class="finding-card-summary">${esc(summary)}</div>
        </div>
        <span class="risk-badge risk-PATTERN">${badgeLabel}</span>
      </div>
      <div class="finding-card-body">
        ${contextHtml}
        ${factsHtml}
        ${contribHtml}
      </div>
    </div>`;
}

// ── Status bar ───────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    const ok = res.ok && data.status === 'ok';
    $('statusDot').className = 'dot ' + (ok ? 'online' : '');
    $('statusText').textContent = ok ? 'API Online' : 'Service unavailable';
  } catch {
    $('statusDot').className = 'dot';
    $('statusText').textContent = 'Service unavailable';
  }
}

// ── Error display ────────────────────────────────────────────
function showError(msg) { $('error').textContent = msg; $('error').style.display = 'block'; }
function clearError()   { $('error').textContent = ''; $('error').style.display = 'none'; }

// ── Main render ──────────────────────────────────────────────
function render(data) {
  clearError();
  $('results').style.display = 'block';

  const findings = pick(data, ['pattern_findings','findings','fraud_findings']) || [];
  const evidence = pick(data, ['evidence_assessment','evidence']) || {};
  const quality  = pick(data, ['data_quality','data_quality_result']);
  const invCase  = pick(data, ['investigation_case','case']);
  const audit    = data.audit_integrity || {};
  const summary  = data.executive_summary || '';

  // Metrics
  $('metricTxn').textContent = pretty(data.transaction_count ?? '—');
  $('metricFindings').textContent = Array.isArray(findings)
    ? findings.length
    : typeof findings === 'object' ? Object.keys(findings).length : '—';

  const riskScore = evidence.risk_score ?? evidence.score ?? null;
  const riskLevel = evidence.risk_level ?? 'NONE';
  $('metricRisk').textContent    = riskScore != null ? riskScore + '/100' : '—';
  $('metricRisk').style.color    = riskColor(riskLevel);
  $('metricRiskLevel').textContent = riskLevel;

  const auditValid = audit.valid;
  $('metricAudit').textContent = auditValid === true ? '✓ Verified' : auditValid === false ? '✗ Invalid' : '—';
  $('metricAudit').style.color  = auditValid === true ? 'var(--green)' : auditValid === false ? 'var(--red)' : 'var(--muted)';
  $('metricAuditEvents').textContent = `${audit.events ?? 0} audit events`;

  // Executive summary
  if (summary) {
    $('execSummaryText').textContent = summary;
    $('execSummaryCard').style.display = 'block';
  } else {
    $('execSummaryCard').style.display = 'none';
  }

  // Findings cards
  const findingsArr = Array.isArray(findings)
    ? findings
    : typeof findings === 'object' ? Object.values(findings) : [];

  if (findingsArr.length) {
    $('findingsGrid').innerHTML = findingsArr.map(renderFindingCard).join('');
  } else {
    $('findingsGrid').innerHTML = '<div style="color:var(--green);font-size:13px">✓ No suspicious patterns detected in this batch.</div>';
  }

  // Detail panels
  $('qualityDetails').innerHTML  = renderDetails(quality);
  $('caseDetails').innerHTML     = renderDetails(invCase);
  $('evidenceDetails').innerHTML = renderDetails(evidence);

  const reviews = {
    data_review: data.review,
    pattern_review: data.pattern_review,
    analysis: data.analysis,
    pattern_analysis: data.pattern_analysis,
  };
  $('reviewDetails').innerHTML = renderDetails(
    Object.fromEntries(Object.entries(reviews).filter(([,v]) => v != null))
  );

  // Raw JSON
  $('rawJson').textContent = JSON.stringify(data, null, 2);

  $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Demo ─────────────────────────────────────────────────────
async function runDemo() {
  clearError();
  const btn = $('demoBtn');
  btn.disabled = true; btn.textContent = 'Running pipeline…';
  try {
    const res = await fetch('/api/demo');
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Demo failed.');
    render(data);
  } catch (e) {
    showError(e.message || 'Demo failed.');
  } finally {
    btn.disabled = false; btn.textContent = '▶ Run Synthetic Demo';
  }
}

// ── Case CSV ─────────────────────────────────────────────────
async function runCsv() {
  clearError();
  const file = $('csvFile').files?.[0];
  if (!file) { showError('Choose a CSV file first.'); return; }
  if (!file.name.toLowerCase().endsWith('.csv')) { showError('Please select a .csv file.'); return; }
  const btn = $('uploadButton');
  btn.disabled = true; btn.textContent = 'Analyzing…';
  try {
    const form = new FormData(); form.append('file', file);
    const res = await fetch('/api/analyze-csv', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Analysis failed.');
    render(data);
  } catch (e) {
    showError(e.message || 'Analysis failed.');
  } finally {
    btn.disabled = false; btn.textContent = 'Analyze Case CSV';
  }
}

function hasValidShapContributions(drivers) {
  return Array.isArray(drivers) && drivers.some(driver =>
    Number.isFinite(Number(driver?.shap_value))
  );
}

function formatShapContribution(value) {
  const contribution = Number(value);
  if (!Number.isFinite(contribution)) return null;
  return `${contribution > 0 ? '+' : ''}${contribution.toFixed(4)}`;
}

function getShapExplanationStatus(status, drivers) {
  const hasValidContributions = hasValidShapContributions(drivers);
  if (status === 'not_requested') return 'not_requested';
  if (status === 'unavailable') return 'unavailable';
  if (status === 'available') {
    return hasValidContributions ? 'available' : 'no_valid_contributions';
  }
  if (!status) return hasValidContributions ? 'available' : 'unavailable';
  return 'unavailable';
}

// ── IEEE single score ─────────────────────────────────────────
async function runIeeeScore(event) {
  event.preventDefault();
  const errBox = $('ieeeError'); const resultBox = $('ieeeScoreResult'); const btn = $('ieeeScoreButton');
  errBox.style.display = 'none'; resultBox.style.display = 'none';
  const amount = Number($('ieeeAmount').value);
  const identityValue = $('ieeeIdentityAvailable').value;
  if (!Number.isFinite(amount) || amount < 0) { errBox.textContent = 'Enter a valid non-negative amount.'; errBox.style.display = 'block'; return; }
  if (identityValue !== '0' && identityValue !== '1') { errBox.textContent = 'Choose identity data availability.'; errBox.style.display = 'block'; return; }
  const tx = { TransactionAmt: amount, identity_available: Number(identityValue) };
  const opts = { TransactionID: $('ieeeTransactionId').value.trim(), card1: $('ieeeCard1').value.trim(), card4: $('ieeeCard4').value.trim(), ProductCD: $('ieeeProductCD').value.trim() };
  for (const [k, v] of Object.entries(opts)) if (v) tx[k] = v;
  btn.disabled = true; btn.textContent = 'Scoring…';
  try {
    const res = await fetch('/api/ieee-score', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({transactions:[tx]}) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Scoring failed.');
    const r = data.results?.[0];
    if (!r) throw new Error('No score result returned.');
    $('ieeeModelStatus').textContent = data.status || 'EXPERIMENTAL_NOT_APPROVED';
    $('ieeeScoreValue').textContent = String(r.experimental_model_score ?? 'Unavailable');
    const coverage = r.input_coverage;
    $('ieeeInputCoverage').textContent = coverage
      ? `${coverage.available_features} of ${coverage.expected_features} model inputs available; ${coverage.missing_features} missing`
      : 'Coverage unavailable';
    $('ieeeScoreTransactionId').textContent = String(r.TransactionID ?? 'Not provided');
    $('ieeeScoreSemantics').textContent = data.score_semantics || '';
    $('ieeeScoreWarning').textContent  = data.warning || '';

    const drivers = r.shap_explanation?.top_n_drivers || [];
    const hasValidContributions = hasValidShapContributions(drivers);
    if (hasValidContributions) {
      $('ieeeShapDriversList').innerHTML = drivers.map(d => {
        const formattedContribution = formatShapContribution(d.shap_value);
        if (formattedContribution === null) return '';
        const color = 'var(--muted)';
        return `<div style="display:flex;justify-content:space-between;align-items:center;padding:7px 10px;background:var(--surface);border-radius:var(--radius-sm);font-size:12px">
          <span>${esc(d.plain_description)}</span>
          <span style="font-weight:700;color:${color}">${formattedContribution}</span>
        </div>`;
      }).join('');
      $('ieeeShapDrivers').style.display = 'block';
    } else {
      $('ieeeShapDriversList').textContent = '';
      $('ieeeShapDrivers').style.display = 'none';
    }

    const explanationStatus = getShapExplanationStatus(
      r.explanation_status,
      drivers
    );
    const explanationStatusEl = $('ieeeExplanationStatus');
    if (explanationStatus === 'unavailable') {
      explanationStatusEl.textContent = hasValidContributions
        ? 'The explanation status is unavailable, but contribution values were returned. Treat this experimental score as an investigation signal only.'
        : 'The score was generated, but no valid explanation contributions are available. This experimental score is not proof of fraud.';
      explanationStatusEl.style.display = 'block';
    } else if (explanationStatus === 'not_requested') {
      explanationStatusEl.textContent = 'An explanation was not requested for this result.';
      explanationStatusEl.style.display = 'block';
    } else if (explanationStatus === 'available' && !hasValidContributions) {
      explanationStatusEl.textContent =
        'An explanation was generated, but no valid contribution values are available to display.';
      explanationStatusEl.style.display = 'block';
    } else {
      explanationStatusEl.textContent = '';
      explanationStatusEl.style.display = 'none';
    }

    resultBox.style.display = 'block';
  } catch (e) {
    errBox.textContent = e.message || 'Scoring failed.'; errBox.style.display = 'block';
  } finally {
    btn.disabled = false; btn.textContent = 'Score Transaction';
  }
}

// ── IEEE batch CSV ────────────────────────────────────────────
let ieeeCsvResultsData = null;

async function runIeeeCsv() {
  const file  = $('ieeeCsvFile').files?.[0];
  const status = $('batchStatus');
  const btn   = $('batchBtn');
  status.textContent = '';
  $('batchResults').style.display = 'none';
  $('batchDownloadBtn').disabled = true;
  ieeeCsvResultsData = null;
  if (!file) { status.textContent = 'Choose a CSV file first.'; return; }
  if (!file.name.toLowerCase().endsWith('.csv')) { status.textContent = 'Please select a .csv file.'; return; }
  if (file.size > 5 * 1024 * 1024) { status.textContent = 'File must be no larger than 5 MB.'; return; }
  const form = new FormData(); form.append('file', file);
  btn.disabled = true; btn.textContent = 'Scoring…';
  try {
    const res  = await fetch('/api/ieee-score-csv', { method:'POST', body:form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Scoring failed.');
    if (!Array.isArray(data.results)) throw new Error('Invalid results.');
    ieeeCsvResultsData = data;
    $('batchCount').textContent      = String(data.scored_count ?? data.results.length);
    $('batchModelStatus').textContent = data.status || 'EXPERIMENTAL_NOT_APPROVED';
    $('batchSemantics').textContent  = data.score_semantics || '';
    $('batchWarning').textContent    = data.warning || '';
    const E = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c]);
    $('batchTable').innerHTML = data.results.map(r => {
      const coverage = r.input_coverage;
      const coverageText = coverage
        ? `${coverage.available_features} of ${coverage.expected_features} model inputs available; ${coverage.missing_features} missing`
        : 'Coverage unavailable';
      return `<tr>
        <td>${E(r.row)}</td>
        <td>${E(r.TransactionID ?? '—')}</td>
        <td style="font-weight:700">${E(r.experimental_model_score)}</td>
        <td>${E(coverageText)}</td>
        <td>No decision made</td>
      </tr>`;
    }).join('');
    $('batchResults').style.display = 'block';
    $('batchDownloadBtn').disabled  = data.results.length === 0;
    status.textContent = `✓ Scored ${data.results.length} transaction${data.results.length !== 1 ? 's' : ''} successfully.`;
  } catch (e) {
    status.textContent = e.message || 'Scoring failed.';
  } finally {
    btn.disabled = false; btn.textContent = 'Score CSV';
  }
}

function downloadIeeeCsv() {
  if (!ieeeCsvResultsData?.results) { $('batchStatus').textContent = 'Score a CSV before downloading.'; return; }
  const cols = ['row','TransactionID','experimental_model_score','score_interpretation'];
  const q = v => {
    let value = String(v ?? '');
    if (typeof v === 'string' && /^[\s\u200B\uFEFF]*[=+\-@]/u.test(value)) {
      value = "'" + value;
    }
    return '"' + value.replace(/"/g, '""') + '"';
  };
  const lines = [cols.map(q).join(','), ...ieeeCsvResultsData.results.map(r => cols.map(c => q(r[c])).join(','))];
  const blob = new Blob(['\uFEFF' + lines.join('\r\n')], {type:'text/csv;charset=utf-8;'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = 'fraudshield-scores.csv';
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ── Init ──────────────────────────────────────────────────────
checkHealth();
</script>
</body>
"""
