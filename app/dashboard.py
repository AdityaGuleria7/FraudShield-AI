
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
    description="Evidence-driven, multi-agent fraud investigation dashboard",
    version="1.0.0",
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
    "is_fraud",
}


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

            fraud_value = str(row["is_fraud"]).strip().lower()
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
                    + ". identity_available must be 0 or 1 and is metadata only."
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

    try:
        result = score_transactions(rows)
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
<title>FraudShield AI | Intelligence Console</title>
<style>
:root {
  color-scheme: dark;
  --bg: #090d16;
  --panel: #111827;
  --panel2: #151e30;
  --border: #253147;
  --text: #edf2fc;
  --muted: #91a0b8;
  --accent: #7c9cff;
  --green: #5ee0b0;
  --red: #ff7f91;
  --yellow: #ffd47a;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
button, input { font: inherit; }
button {
  cursor: pointer;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px 15px;
  background: var(--panel2);
  color: var(--text);
}
button.primary {
  background: var(--accent);
  color: #091020;
  border-color: transparent;
  font-weight: 700;
}
button:disabled { opacity: .55; cursor: wait; }
.shell { max-width: 1400px; margin: auto; padding: 28px; }
header {
  display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 16px; margin-bottom: 28px;
}
.brand { display: flex; align-items: center; gap: 13px; }
.logo {
  display: grid; place-items: center; width: 46px; height: 46px;
  border-radius: 14px; background: #202d4b; color: var(--accent);
  font-size: 23px; font-weight: 800;
}
h1 { font-size: 23px; margin: 0; letter-spacing: -.5px; }
.subtitle, .muted { color: var(--muted); }
.subtitle { margin-top: 2px; font-size: 12px; }
.status { color: var(--green); font-size: 12px; }
.status::before {
  content: ""; display: inline-block; width: 7px; height: 7px;
  border-radius: 50%; background: var(--green); margin-right: 8px;
}
.hero {
  padding: 26px; border: 1px solid var(--border); border-radius: 18px;
  background: linear-gradient(120deg, #17223a, #101726 65%);
  margin-bottom: 20px;
}
.eyebrow {
  color: var(--accent); font-size: 11px; font-weight: 800;
  text-transform: uppercase; letter-spacing: 1.8px;
}
.hero h2 { font-size: 27px; margin: 7px 0; }
.hero p { color: var(--muted); max-width: 720px; margin: 0 0 18px; }
.actions { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.upload {
  border: 1px dashed #405171; padding: 9px; border-radius: 10px;
  background: #0c1422; max-width: 100%;
}
.grid { display: grid; gap: 16px; margin-bottom: 16px; }
.metrics { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.panel {
  min-width: 0; padding: 20px; background: var(--panel);
  border: 1px solid var(--border); border-radius: 16px;
}
.metric-label { color: var(--muted); font-size: 12px; }
.metric-value { font-size: 26px; font-weight: 750; margin: 5px 0; }
.panel h3 { font-size: 14px; margin: 0 0 15px; }
.panel p { overflow-wrap: anywhere; }
.tag {
  display: inline-block; border-radius: 7px; padding: 3px 8px;
  font-size: 11px; background: #273248; color: var(--text);
  overflow-wrap: anywhere;
}
.good { color: var(--green); background: #12372f; }
.bad { color: var(--red); background: #3a202b; }
.warn { color: var(--yellow); background: #3a3020; }
.stack { display: grid; gap: 11px; }
.advanced-panel { margin-top: 16px; }
.advanced-panel > summary {
  cursor: pointer; font-weight: 650; list-style: none;
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
}
.advanced-panel > summary::-webkit-details-marker { display: none; }
.advanced-panel > summary::after { content: "+"; color: var(--muted); font-size: 20px; }
.advanced-panel[open] > summary::after { content: "−"; }
.advanced-content { margin-top: 18px; }
.primary-workflow { margin-top: 18px; }

.item {
  border-bottom: 1px solid var(--border); padding: 0 0 12px;
  overflow-wrap: anywhere;
}
.item:last-child { border-bottom: 0; padding-bottom: 0; }
.item-title { font-weight: 650; margin-bottom: 4px; }
.item-detail { font-size: 12px; color: var(--muted); }
.detail {
  display: flex; justify-content: space-between; gap: 14px;
  border-bottom: 1px solid var(--border); padding: 9px 0;
}
.detail:last-child { border-bottom: 0; }
.detail span { color: var(--muted); }
.detail strong { text-align: right; overflow-wrap: anywhere; }
pre {
  white-space: pre-wrap; overflow-wrap: anywhere; max-height: 450px;
  overflow: auto; padding: 15px; border-radius: 10px;
  background: #090e18; color: #c7d7f7; font-size: 12px;
}
#error {
  display: none; padding: 12px; border: 1px solid #63303b;
  background: #2a1720; color: var(--red); border-radius: 10px;
  margin-bottom: 16px; overflow-wrap: anywhere;
}
#results { display: none; }
footer { color: var(--muted); font-size: 11px; padding: 18px 0; }
@media (max-width: 850px) {
  .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .two { grid-template-columns: 1fr; }
}
@media (max-width: 520px) {
  .shell { padding: 15px; }
  .metrics { grid-template-columns: 1fr 1fr; gap: 9px; }
  .panel { padding: 14px; }
  .metric-value { font-size: 21px; }
  .hero h2 { font-size: 23px; }
}
</style>
</head>
<body>
<div class="shell">
  <header>
    <div class="brand">
      <div class="logo">F</div>
      <div>
        <h1>FraudShield AI</h1>
        <div class="subtitle">MULTI-AGENT FRAUD INTELLIGENCE CONSOLE</div>
      </div>
    </div>
    <div id="health" class="status">Checking local service...</div>
  </header>

  <section class="hero">
    <div class="eyebrow">Investigations workspace</div>
    <h2>Detect patterns. Inspect evidence. Verify decisions.</h2>
    <p>
      Start with a guided demo or score an IEEE-CIS feature CSV. FraudShield
      keeps model scores separate from evidence, agent reviews, and policy decisions.
    </p>
    <div class="actions">
      <button class="primary" id="demoButton" onclick="runDemo()">
        Run synthetic demo
      </button>
    </div>
    <p class="subtitle" style="margin-top:13px">
      Experimental model scores are not calibrated fraud probabilities or production decisions.
      Use only data you are authorized to process.
    </p>
  </section>


  <section class="panel primary-workflow" id="ieeeCsvScoringPanel" style="margin-top:18px">
    <div class="eyebrow">Batch scoring</div>
    <h3>IEEE-CIS CSV Scoring</h3>
    <p class="muted">
      Upload up to 5,000 rows of compatible transaction features.
      CSV files must include TransactionAmt and identity_available (0 or 1).
      Identity availability is metadata only; it does not directly affect scores.
      Maximum file size: 5 MB.
    </p>
    <div class="actions">
      <input id="ieeeCsvFile" type="file" accept=".csv,text/csv">
      <button id="ieeeCsvButton" onclick="runIeeeCsv()">Score CSV</button>
      <button id="ieeeCsvDownload" onclick="downloadIeeeCsv()" disabled>
        Download results CSV
      </button>
    </div>
    <p id="ieeeCsvStatus" class="muted" role="status" aria-live="polite"></p>
    <div id="ieeeCsvResults" style="display:none; margin-top:16px">
      <div class="grid metrics">
        <div>
          <div class="metric-label">Rows scored</div>
          <div class="metric-value" id="ieeeCsvCount">0</div>
        </div>
        <div>
          <div class="metric-label">Model status</div>
          <div class="tag warn" id="ieeeCsvModelStatus">EXPERIMENTAL_NOT_APPROVED</div>
        </div>
      </div>
      <p class="muted" id="ieeeCsvSemantics"></p>
      <div style="overflow-x:auto">
        <table style="width:100%; border-collapse:collapse">
          <thead>
            <tr>
              <th style="text-align:left; padding:9px">Row</th>
              <th style="text-align:left; padding:9px">Transaction ID</th>
              <th style="text-align:left; padding:9px">Experimental score</th>
            </tr>
          </thead>
          <tbody id="ieeeCsvTable"></tbody>
        </table>
      </div>
      <p class="muted" id="ieeeCsvWarning"></p>
    </div>
  </section>


  <details class="panel advanced-panel" id="ieeeScoringPanel">
    <summary>Advanced: score a single transaction</summary>
    <div class="advanced-content">
    <div class="eyebrow">Experimental machine learning</div>
    <h3>IEEE-CIS Transaction Scoring</h3>
    <p class="muted">
      Score a compatible transaction using the experimental IEEE-CIS model.
      Leave unavailable optional features blank.
    </p>

    <form id="ieeeScoreForm" onsubmit="runIeeeScore(event)">
      <div class="grid two">
        <label>
          Transaction amount *
          <input id="ieeeAmount" type="number" min="0" step="any"
                 required placeholder="125.50">
        </label>
        <label>
          Identity data source status *
          <select id="ieeeIdentityAvailable" required>
            <option value="">Choose one</option>
            <option value="0">Unavailable (0)</option>
            <option value="1">Available (1)</option>
          </select>
        </label>
        <label>
          Transaction ID (optional)
          <input id="ieeeTransactionId" type="text" placeholder="999001">
        </label>
        <label>
          card1 (optional)
          <input id="ieeeCard1" type="text" placeholder="12345">
        </label>
        <label>
          card2 (optional)
          <input id="ieeeCard2" type="text" placeholder="111">
        </label>
        <label>
          ProductCD (optional)
          <input id="ieeeProductCD" type="text" placeholder="W">
        </label>
        <label>
          card4 (optional)
          <input id="ieeeCard4" type="text" placeholder="visa">
        </label>
        <label>
          card6 (optional)
          <input id="ieeeCard6" type="text" placeholder="debit">
        </label>
      </div>

      <p class="muted">
        This form supplies the fields shown above. Other model features are
        left missing and handled by the model's configured preprocessing.
        This is request metadata only; it does not directly affect the model score.
        Identity features must be supplied separately to influence the prediction.
      </p>
      <button class="primary" id="ieeeScoreButton" type="submit">
        Score transaction
      </button>
    </form>

    <div id="ieeeError" role="alert" style="display:none; margin-top:12px"></div>

    <section id="ieeeScoreResult" style="display:none; margin-top:18px"
             aria-live="polite">
      <h3>Experimental score result</h3>
      <div class="detail">
        <span>Model status</span>
        <strong id="ieeeModelStatus">—</strong>
      </div>
      <div class="detail">
        <span>Experimental model score</span>
        <strong id="ieeeScoreValue">—</strong>
      </div>
      <div class="detail">
        <span>Transaction ID</span>
        <strong id="ieeeScoreTransactionId">—</strong>
      </div>
      <p id="ieeeScoreSemantics" class="muted"></p>
      <p id="ieeeScoreWarning" class="muted"></p>
    </section>

    <p class="muted" style="margin-top:14px">
      Safety notice: this score is uncalibrated, not a fraud probability,
      and not approved evidence. It must not independently trigger account
      restrictions, payment declines, or other adverse actions.
    </p>
  </div>
  </details>


  <details class="panel advanced-panel" id="caseCsvPanel">
    <summary>Advanced: analyze a FraudShield case CSV</summary>
    <div class="advanced-content">
      <p class="muted">
        This workflow uses FraudShield's internal transaction schema, not the
        IEEE-CIS competition schema. Required fields include transaction_id,
        account_id, device_id, ip_address, country, merchant_category, amount,
        timestamp, and is_fraud.
      </p>
      <div class="actions">
        <input id="csvFile" type="file" accept=".csv,text/csv">
        <button id="uploadButton" onclick="runCsv()">Analyze case CSV</button>
      </div>
    </div>
  </details>

<div id="error"></div>

  <section id="results">
    <div class="grid metrics">
      <div class="panel">
        <div class="metric-label">Transactions analyzed</div>
        <div class="metric-value" id="transactionCount">—</div>
        <div class="muted">Records processed</div>
      </div>
      <div class="panel">
        <div class="metric-label">Fraud findings</div>
        <div class="metric-value" id="findingCount">—</div>
        <div class="muted">Pipeline findings returned</div>
      </div>
      <div class="panel">
        <div class="metric-label">Evidence score</div>
        <div class="metric-value" id="evidenceScore">—</div>
        <div class="muted">From evidence assessment</div>
      </div>
      <div class="panel">
        <div class="metric-label">Audit integrity</div>
        <div class="metric-value" id="auditStatus">—</div>
        <div class="muted" id="auditEvents">Awaiting result</div>
      </div>
    </div>

    <div class="grid two">
      <section class="panel">
        <h3>Data quality</h3>
        <div id="quality"></div>
      </section>
      <section class="panel">
        <h3>Investigation case</h3>
        <div id="caseDetails"></div>
      </section>
    </div>

    <div class="grid two">
      <section class="panel">
        <h3>Fraud findings</h3>
        <div id="findings" class="stack"></div>
      </section>
      <section class="panel">
        <h3>Evidence assessment</h3>
        <div id="evidence"></div>
      </section>
    </div>

    <div class="grid two">
      <section class="panel">
        <h3>Agent reviews</h3>
        <div id="reviews" class="stack"></div>
      </section>
      <section class="panel">
        <h3>Run information</h3>
        <div id="runInfo"></div>
      </section>
    </div>

    <section class="panel">
      <div class="actions" style="justify-content:space-between">
        <h3 style="margin:0">Raw pipeline response</h3>
        <button onclick="toggleRaw()">Show / hide JSON</button>
      </div>
      <pre id="raw" hidden></pre>
    </section>
  </section>

  <footer>
    FraudShield AI · Local development dashboard ·
    Synthetic demo results do not establish real-world fraud detection accuracy.
  </footer>
</div>

<script>
const $ = id => document.getElementById(id);

function esc(value) {
  const node = document.createElement("span");
  node.textContent = String(value ?? "—");
  return node.innerHTML;
}

function pretty(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function entries(value) {
  if (value && typeof value === "object") {
    if (Array.isArray(value)) {
      return value.map((item, i) => [String(i + 1), item]);
    }
    return Object.entries(value);
  }
  return value === undefined ? [] : [["Result", value]];
}

function renderDetails(value) {
  const rows = entries(value);
  if (!rows.length) {
    return '<div class="muted">No information returned.</div>';
  }

  return rows.map(([key, val]) => `
    <div class="detail">
      <span>${esc(String(key).replaceAll("_", " "))}</span>
      <strong>${esc(pretty(val))}</strong>
    </div>
  `).join("");
}

function renderItems(value) {
  const rows = entries(value);
  if (!rows.length) {
    return '<div class="muted">No items returned.</div>';
  }

  return rows.map(([key, val]) => {
    const title = typeof val === "object" && val !== null
      ? (val.name ?? val.pattern ?? val.finding_type ?? val.type ?? key)
      : key;

    const detail = typeof val === "object" && val !== null
      ? JSON.stringify(val, null, 2)
      : String(val);

    return `
      <div class="item">
        <div class="item-title">${esc(title)}</div>
        <div class="item-detail">${esc(detail)}</div>
      </div>
    `;
  }).join("");
}

function firstPresent(data, keys) {
  for (const key of keys) {
    if (data && data[key] !== undefined && data[key] !== null) {
      return data[key];
    }
  }
  return null;
}

function showError(message) {
  $("error").textContent = message;
  $("error").style.display = "block";
}

function clearError() {
  $("error").textContent = "";
  $("error").style.display = "none";
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error("Service unavailable");
    const data = await response.json();
    $("health").textContent = data.status === "ok"
      ? "Local service online"
      : "Service status unknown";
  } catch (_) {
    $("health").textContent = "Service unavailable";
  }
}

async function runDemo() {
  await request("/api/demo", null);
}

async function runCsv() {
  const file = $("csvFile").files[0];

  if (!file) {
    showError("Choose a CSV file first.");
    return;
  }

  if (!file.name.toLowerCase().endsWith(".csv")) {
    showError("Please select a .csv file.");
    return;
  }

  const form = new FormData();
  form.append("file", file);

  await request("/api/analyze-csv", form);
}

async function request(url, body) {
  clearError();

  $("demoButton").disabled = true;
  $("uploadButton").disabled = true;
  $("demoButton").textContent = "Processing...";

  try {
    const options = body
      ? { method: "POST", body }
      : { method: "GET" };

    const response = await fetch(url, options);
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Request failed.");
    }

    render(data);
  } catch (error) {
    showError(error.message || "An unexpected error occurred.");
  } finally {
    $("demoButton").disabled = false;
    $("uploadButton").disabled = false;
    $("demoButton").textContent = "Run synthetic demo";
  }
}

function render(data) {
  $("results").style.display = "block";

  const findings = firstPresent(data, [
    "pattern_findings", "findings", "fraud_findings"
  ]);

  const evidence = firstPresent(data, [
    "evidence_assessment", "evidence"
  ]);

  const quality = firstPresent(data, [
    "data_quality", "data_quality_result"
  ]);

  const investigationCase = firstPresent(data, [
    "investigation_case", "case"
  ]);

  const audit = data.audit_integrity || {};

  $("transactionCount").textContent =
    pretty(data.transaction_count ?? "—");

  $("findingCount").textContent =
    Array.isArray(findings) ? findings.length
    : findings && typeof findings === "object"
      ? Object.keys(findings).length
      : "—";

  $("evidenceScore").textContent = pretty(
    firstPresent(evidence, ["score", "evidence_score", "risk_score"])
  );

  $("auditStatus").textContent =
    audit.valid === true ? "Verified"
    : audit.valid === false ? "Invalid"
    : "Unknown";

  $("auditStatus").style.color =
    audit.valid === true ? "var(--green)"
    : audit.valid === false ? "var(--red)"
    : "var(--yellow)";

  $("auditEvents").textContent =
    `${audit.events ?? 0} audit events`;

  $("quality").innerHTML = renderDetails(quality);
  $("caseDetails").innerHTML = renderDetails(investigationCase);
  $("findings").innerHTML = renderItems(findings);
  $("evidence").innerHTML = renderDetails(evidence);

  const reviews = {
    data_review: data.review,
    pattern_review: data.pattern_review,
    analysis: data.analysis,
    pattern_analysis: data.pattern_analysis
  };

  $("reviews").innerHTML = renderItems(reviews);

  $("runInfo").innerHTML = renderDetails({
    task_id: data.task_id ?? "FRAUDSHIELD-DASHBOARD",
    demo_data: data.demo_notice === true,
    audit_integrity: audit.valid,
    audit_events: audit.events,
    returned_fields: Object.keys(data)
  });

  $("raw").textContent = JSON.stringify(data, null, 2);
  $("raw").hidden = false;

  $("results").scrollIntoView({
    behavior: "smooth",
    block: "start"
  });
}


async function runIeeeScore(event) {
  event.preventDefault();

  const errorBox = $("ieeeError");
  const resultBox = $("ieeeScoreResult");
  const button = $("ieeeScoreButton");

  errorBox.textContent = "";
  errorBox.style.display = "none";
  resultBox.style.display = "none";

  const amount = Number($("ieeeAmount").value);
  const identityValue = $("ieeeIdentityAvailable").value;

  if (!Number.isFinite(amount) || amount < 0) {
    errorBox.textContent = "Enter a valid, non-negative transaction amount.";
    errorBox.style.display = "block";
    return;
  }

  if (identityValue !== "0" && identityValue !== "1") {
    errorBox.textContent = "Choose whether identity data is available.";
    errorBox.style.display = "block";
    return;
  }

  const transaction = {
    TransactionAmt: amount,
    identity_available: Number(identityValue)
  };

  const optionalFields = {
    TransactionID: $("ieeeTransactionId").value.trim(),
    card1: $("ieeeCard1").value.trim(),
    card2: $("ieeeCard2").value.trim(),
    ProductCD: $("ieeeProductCD").value.trim(),
    card4: $("ieeeCard4").value.trim(),
    card6: $("ieeeCard6").value.trim()
  };

  for (const [key, value] of Object.entries(optionalFields)) {
    if (value !== "") transaction[key] = value;
  }

  button.disabled = true;
  button.textContent = "Scoring...";

  try {
    const response = await fetch("/api/ieee-score", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({transactions: [transaction]})
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Scoring request failed.");
    }

    const result = data.results?.[0];
    if (!result) throw new Error("The API returned no score result.");

    $("ieeeModelStatus").textContent =
      data.status || "EXPERIMENTAL_NOT_APPROVED";
    $("ieeeScoreValue").textContent =
      String(result.experimental_model_score ?? "Unavailable");
    $("ieeeScoreTransactionId").textContent =
      String(result.TransactionID ?? "Not provided");
    $("ieeeScoreSemantics").textContent =
      data.score_semantics || result.score_interpretation ||
      "Uncalibrated model score; not a calibrated fraud probability.";
    $("ieeeScoreWarning").textContent =
      data.warning || "This result does not constitute an approved fraud decision.";

    resultBox.style.display = "block";
  } catch (error) {
    errorBox.textContent = error.message || "Scoring request failed.";
    errorBox.style.display = "block";
  } finally {
    button.disabled = false;
    button.textContent = "Score transaction";
  }
}


let ieeeCsvResultsData = null;

async function runIeeeCsv() {
  const fileInput = $("ieeeCsvFile");
  const status = $("ieeeCsvStatus");
  const button = $("ieeeCsvButton");
  const file = fileInput.files?.[0];

  status.textContent = "";
  $("ieeeCsvResults").style.display = "none";
  $("ieeeCsvDownload").disabled = true;
  ieeeCsvResultsData = null;

  if (!file) {
    status.textContent = "Choose a CSV file first.";
    return;
  }

  if (!file.name.toLowerCase().endsWith(".csv")) {
    status.textContent = "Please select a .csv file.";
    return;
  }

  if (file.size > 5 * 1024 * 1024) {
    status.textContent = "The CSV file must be no larger than 5 MB.";
    return;
  }

  const form = new FormData();
  form.append("file", file);
  button.disabled = true;
  button.textContent = "Scoring CSV...";

  try {
    const response = await fetch("/api/ieee-score-csv", {
      method: "POST",
      body: form
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "CSV scoring failed.");
    }

    if (!Array.isArray(data.results)) {
      throw new Error("The API returned an invalid results list.");
    }

    ieeeCsvResultsData = data;
    $("ieeeCsvCount").textContent = String(data.scored_count ?? data.results.length);
    $("ieeeCsvModelStatus").textContent =
      data.status || "EXPERIMENTAL_NOT_APPROVED";
    $("ieeeCsvSemantics").textContent =
      data.score_semantics || "Uncalibrated model score; not a calibrated fraud probability.";
    $("ieeeCsvWarning").textContent =
      data.warning || "Experimental scores are not approved fraud decisions.";

    const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, char => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;"
    })[char]);

    $("ieeeCsvTable").innerHTML = data.results.map(result => `
      <tr>
        <td style="padding:9px; border-top:1px solid var(--border)">
          ${escapeHtml(result.row)}
        </td>
        <td style="padding:9px; border-top:1px solid var(--border)">
          ${escapeHtml(result.TransactionID ?? "Not provided")}
        </td>
        <td style="padding:9px; border-top:1px solid var(--border)">
          ${escapeHtml(result.experimental_model_score)}
        </td>
      </tr>
    `).join("");

    $("ieeeCsvResults").style.display = "block";
    $("ieeeCsvDownload").disabled = data.results.length === 0;
    status.textContent = "CSV scoring completed successfully.";
  } catch (error) {
    status.textContent = error.message || "CSV scoring failed.";
  } finally {
    button.disabled = false;
    button.textContent = "Score CSV";
  }
}

function downloadIeeeCsv() {
  if (!ieeeCsvResultsData || !Array.isArray(ieeeCsvResultsData.results)) {
    $("ieeeCsvStatus").textContent = "Score a CSV before downloading results.";
    return;
  }

  const columns = [
    "row",
    "TransactionID",
    "experimental_model_score",
    "score_interpretation"
  ];

  const quote = value => {
    const text = String(value ?? "");
    return '"' + text.replace(/"/g, '""') + '"';
  };

  const lines = [
    columns.map(quote).join(","),
    ...ieeeCsvResultsData.results.map(result =>
      columns.map(column => quote(result[column])).join(",")
    )
  ];

  const blob = new Blob(["\uFEFF" + lines.join("\r\n")], {
    type: "text/csv;charset=utf-8;"
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "fraudshield-ieee-experimental-scores.csv";
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function toggleRaw() {
  $("raw").hidden = !$("raw").hidden;
}

checkHealth();
</script>
</body>
</html>
"""
