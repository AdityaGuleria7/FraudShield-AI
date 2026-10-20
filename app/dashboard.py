
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

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


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


HTML = (TEMPLATE_DIR / "dashboard.html").read_text(encoding="utf-8")
