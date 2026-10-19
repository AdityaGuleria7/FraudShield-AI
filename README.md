# FraudShield AI

<div align="center">

**Evidence-Driven · Analyst/Reviewer · Explainable · Auditable**

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.143-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn)](https://scikit-learn.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.4-1C3C3C)](https://langchain.com)
[![Tests](https://img.shields.io/badge/Tests-364%20passing-brightgreen)](tests/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

</div>

---

FraudShield AI is a **experimental fraud detection research and portfolio platform** built around one principle:

> **No trust without verification. Evidence over consensus.**

Instead of a single model score triggering a decision, FraudShield combines deterministic rule-based pattern detection, a machine-learning ranking model, analyst/reviewer validation stages, and a tamper-evident audit ledger — making every flagged transaction **explainable, traceable, and reviewable**.

> **Project status:** Portfolio and research project under active development. Not a production deployment or guarantee of fraud prevention.

---

## Performance Highlights

The current identity/device-enriched Extra Trees model was evaluated experimentally on the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/overview) using a chronological split.

**Important:** the evaluation period was examined during earlier experiments. These figures are exploratory, not an independent final benchmark, and do not establish real-world or production performance.

| Metric | Exploratory result |
|--------|-------------------:|
| ROC-AUC | **0.7906** |
| Average Precision | **0.1891** |
| Precision @ Top 1% Review | **35.6%** |
| Precision @ Top 2% Review | **34.2%** |
| Precision @ Top 5% Review | **24.4%** |
| Training rows | 413,378 |
| Test rows | 88,581 |

In this reported split, the top 1% review queue contains 885 transactions and includes 315 fraud cases. This describes the dataset experiment only; model scores are uncalibrated, and no production decision threshold is approved.

See [Model Card](docs/MODEL_CARD.md), [Evaluation Protocol](docs/EVALUATION.md), and [IEEE-CIS Research Notes](docs/IEEE_CIS_RESEARCH.md) for methodology and limitations.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FraudShield AI Pipeline                          │
│                                                                         │
│  Input                                                                  │
│  Transactions ──►  Stage 1: Data Analyst Agent                         │
│                         │   (pattern recognition, feature extraction)   │
│                         ▼                                               │
│                    Stage 2: Data Guardian Review                        │
│                         │   (deterministic reviewer validation)            │
│                         ▼                                               │
│                    Stage 3: Data Quality Gate                           │
│                         │   (reject malformed or low-quality data)      │
│                         ▼                                               │
│                    Stage 4: Fraud Pattern Engine                        │
│                         │   ├─ Velocity Attack Detection                │
│                         │   ├─ Impossible Travel Detection              │
│                         │   ├─ Account Takeover Detection               │
│                         │   └─ New Device High-Value Detection          │
│                         ▼                                               │
│                    Stage 5: Evidence Correlation Engine                 │
│                         │   (provenance validation, score weighting,    │
│                         │    ML corroboration, overlap-aware scoring)   │
│                         ▼                                               │
│                    Stage 6: Investigation Case Builder                  │
│                         │   (structured case with risk level + actions) │
│                         ▼                                               │
│                    Stage 7: Fraud Pattern Analyst Agent                 │
│                         │   (secure execution via policy + registry)    │
│                         ▼                                               │
│                    Stage 8: Fraud Pattern Guardian Review               │
│                         │                                               │
│                         ▼                                               │
│                    Audit Ledger (hash-linked, signed checkpoints)       │
│                         │                                               │
│                         ▼                                               │
│                    Dashboard / API Response                             │
└─────────────────────────────────────────────────────────────────────────┘
```

**Why analyst/reviewer validation?**

Analyst outputs are checked by separate reviewer components before the pipeline advances. These are deterministic Python components that validate agent identity and evidence requirements; they are not autonomous LLM agents. This reflects a review principle used in fraud investigation: an analyst proposes findings, a reviewer checks them, and policy controls authorize protected operations.

---

## Key Features

### 🔍 Fraud Pattern Detection
- **Velocity Attack** — multiple transactions from one account in a short window (card-testing detection)
- **Impossible Travel** — physical location jumps that exceed plausible travel speed
- **Account Takeover** — coordinated signals: new device + new IP + high-value + unusual time
- **New Device High-Value** — first-time device used for a large transaction

### 🤖 Analyst/Reviewer Validation
- Data Analyst + Data Guardian (independent review of analysis)
- Fraud Pattern Analyst + Fraud Pattern Guardian
- Structured agent contracts (Pydantic) — outputs must satisfy defined schemas before downstream use
- Least-privilege policy engine — agents only access what they're authorized for

### 📊 Machine Learning (IEEE-CIS)
- ExtraTrees classifier trained on 413,378 transaction rows in the chronological training partition (approximately 3.52% fraud prevalence)
- Chronological three-period train/validate/test split (no future leakage)
- Cost-sensitive threshold optimization (false-negative cost weighted 5× false-positive)
- Batch CSV scoring via dashboard; per-row experimental scores

### 🛡️ Security & Auditability
- Hash-linked JSONL audit ledger — every pipeline decision recorded
- Signed checkpoints with rollback detection
- S3 external checkpoint anchoring (configurable)
- Operation registry with least-privilege authorization
- Secure executor — operations run only after policy approval

### 💡 Evidence Corroboration
- Multiple patterns on the same transaction are **not** treated as independent — overlap-aware scoring prevents inflated risk scores
- ML scores only corroborate deterministic findings, never replace them
- Evidence quality graded: STRONG / PARTIAL / NONE

---

## Quick Start

### Option A — Docker (Recommended)

```bash
git clone https://github.com/AdityaGuleria7/FraudShield-AI.git
cd FraudShield-AI

# Configure credentials
cp .env.example .env
# Edit .env and set FRAUDSHIELD_AUTH_USERNAME and FRAUDSHIELD_AUTH_PASSWORD

docker-compose up
```

Open [http://localhost:8000](http://localhost:8000) and sign in.

### Option B — Local Python

```bash
git clone https://github.com/AdityaGuleria7/FraudShield-AI.git
cd FraudShield-AI

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt

export FRAUDSHIELD_AUTH_USERNAME="admin"
export FRAUDSHIELD_AUTH_PASSWORD="your-strong-password"

python -m uvicorn app.dashboard:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## Dashboard

The dashboard provides three workflows:

| Workflow | Description |
|----------|-------------|
| **Run Demo** | Generates synthetic transactions with an injected velocity attack, runs the full pipeline, and displays findings with audit verification |
| **IEEE-CIS CSV Scoring** | Upload up to 5,000 rows of compatible transaction features and receive experimental fraud risk scores |
| **FraudShield Case CSV** | Upload transactions in FraudShield's schema (see `examples/`) and run the full multi-agent investigation pipeline |

### Example CSVs

Three ready-to-use example files are in `examples/`:

| File | Description |
|------|-------------|
| `sample_transactions.csv` | Clean inference CSV — velocity attack injected on ACC-001 |
| `sample_transactions_labelled.csv` | With ground-truth `is_fraud` labels for evaluation |
| `sample_impossible_travel.csv` | Same account transacting in US and Japan within 90 minutes |

---

## Running Tests

```bash
# Full suite (329 tests at the time of this documentation update)
python -m pytest -q

# By category
python -m pytest tests/test_signed_checkpoints.py tests/test_audit_ledger.py -q
python -m pytest tests/test_policy_engine.py tests/test_secure_executor.py -q
python -m pytest tests/test_evidence_engine.py tests/test_pattern_engine.py -q
python -m pytest tests/test_ml_pipeline.py tests/test_ieee_scorer.py -q
```

---

## Running the Synthetic Demo (CLI)

```bash
PYTHONPATH=. .venv/bin/python scripts/run_demo.py
```

Generates synthetic transactions, injects a velocity-attack scenario, runs the full investigation pipeline, and verifies the audit hash chain. Expected output ends with `Hash chain valid: True`.

---

## Reproducing IEEE-CIS Results

1. Download the competition data from [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data) (requires account and competition acceptance).
2. Place `train_transaction.csv` and `train_identity.csv` in `~/Downloads/ieee-fraud-detection/`.
3. Run training scripts in `scripts/`:
   ```bash
   PYTHONPATH=. python scripts/train_ieee_tree.py
   PYTHONPATH=. python scripts/validate_ieee_three_period.py
   ```
4. Metrics are saved locally to `reports/`, and the trained artifact goes to `artifacts/`. Both directories are excluded from Git; summarized experimental results and their limitations are documented in `docs/EVALUATION.md` and `docs/IEEE_CIS_RESEARCH.md`.

See [IEEE-CIS Research Notes](docs/IEEE_CIS_RESEARCH.md) for full details, split strategy, and limitations.

---

## Project Structure

```
FraudShield-AI/
├── app/
│   ├── agents/              # Data analyst + fraud pattern analyst agents
│   │   └── guardians/       # Deterministic reviewer components
│   ├── audit/               # Hash-linked ledger, signed checkpoints, S3 anchoring
│   ├── core/                # 8-stage orchestration pipeline
│   ├── data/                # Synthetic transaction generation
│   ├── data_quality/        # Data validation gate
│   ├── features/            # Feature engineering (behavioral, temporal, velocity…)
│   ├── fraud/               # Pattern detection (velocity, travel, ATO, new-device)
│   ├── investigation/       # Investigation case construction
│   ├── ml/                  # Model pipeline, evaluation, threshold optimization
│   ├── models/              # Agent contract schemas (Pydantic)
│   └── security/            # Policy engine, operation registry, secure executor
├── examples/                # Ready-to-use sample CSV files
├── scripts/                 # Training, evaluation, and demo scripts
├── tests/                   # 364 automated tests
├── docs/                    # Research notes and methodology
├── .env.example             # Environment variable template
├── Dockerfile               # Production container image
├── docker-compose.yml       # One-command local deployment
├── requirements.txt         # Runtime dependencies
├── requirements-dev.txt     # Runtime + test dependencies
└── requirements-research.txt # Optional challenger-model dependencies
```

---

## Security Design

| Principle | Implementation |
|-----------|----------------|
| **Evidence over consensus** | Multiple agents agreeing is not sufficient — findings must satisfy provenance and quality checks |
| **Independent review** | Guardian agents review analyst outputs before pipeline advances |
| **Least privilege** | Protected operations require explicit policy authorization |
| **Structured contracts** | Agent outputs validated against Pydantic schemas before downstream use |
| **Auditability** | Every pipeline decision recorded in a hash-linked ledger |
| **Fail safely** | Invalid evidence, unauthorized operations, and integrity failures surface as explicit errors |

---

## Audit Integrity

The audit subsystem records every pipeline stage decision in a hash-linked JSONL ledger. Each entry's hash includes the previous entry's hash — tampering with any record invalidates all subsequent ones.

Signed checkpoints can be anchored to AWS S3 with Object Lock for external integrity verification. See [`app/audit/external_checkpoints.py`](app/audit/external_checkpoints.py) and the [operational requirements](#audit-integrity-and-external-checkpoints) below.

### Important Operational Requirements

For meaningful tamper-evidence:
- Configure S3 Object Lock and bucket versioning
- Use restrictive bucket permissions and secure key management
- Monitor and alert on verification failures
- A hash chain alone cannot stop an attacker with full access — the S3 anchor and signing key are the trust roots

---

## Limitations

- Synthetic or test data does not establish effectiveness on real transaction traffic
- IEEE-CIS results are exploratory on a Kaggle competition dataset — not a production benchmark
- No live payment integrations — the pipeline runs on provided data only
- A detected anomaly is a signal for investigation, not proof of fraud
- Human review, privacy controls, model monitoring, and operational governance are required before high-impact real-world use

---

## Roadmap

- [x] Analyst/reviewer pipeline (8 stages)
- [x] Hash-linked audit ledger with signed checkpoints
- [x] IEEE-CIS ML model with chronological validation
- [x] FastAPI dashboard with rate limiting and authentication
- [x] Docker one-command deployment
- [ ] SHAP explainability — plain-language "why flagged?" cards per finding
- [ ] XGBoost / LightGBM challenger models
- [ ] Graph-based fraud ring detection (NetworkX)
- [ ] D3.js interactive entity relationship visualization
- [ ] OpenTelemetry pipeline latency tracing
- [ ] API key authentication layer

---

## Author

**Aditya Guleria**

GitHub: [AdityaGuleria7](https://github.com/AdityaGuleria7)

---

## Disclaimer

FraudShield AI is a software research and portfolio project. It is not a guarantee of fraud prevention, a substitute for professional fraud investigation, or a claim of regulatory compliance. Validate the system and its controls before using it with real financial data or consequential decisions.
