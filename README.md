# FraudShield AI

<div align="center">

**Evidence-Driven · Explainable · Auditable · Analyst-Friendly**

[![CI](https://github.com/AdityaGuleria7/FraudShield-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/AdityaGuleria7/FraudShield-AI/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.143-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn)](https://scikit-learn.org)
[![Tests](https://img.shields.io/badge/tests-449%20passing-brightgreen)](tests/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

</div>

FraudShield AI is an **experimental fraud-detection research and portfolio platform** built around one principle:

> **No trust without verification. Evidence over consensus.**

Instead of a single model score triggering a decision, FraudShield combines deterministic pattern detection, an ML ranking model, analyst/reviewer validation stages and a tamper-evident audit ledger, so every flagged transaction is **explainable, traceable and reviewable**.

> **Status:** portfolio and research project. Not a production deployment or a guarantee of fraud prevention.

---

## Results at a glance

Evaluated on the public [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/overview) dataset (590,540 transactions) with a strict chronological 70/15/15 split.

| Review queue (test period) | Fraud caught | Precision | Lift vs random review |
|---|---:|---:|---:|
| Top 1% (885 transactions) | 315 | 35.6% | **10.2x** |
| Top 2% (1,771) | 606 | 34.2% | 9.8x |
| Top 5% (4,429) | 1,080 | 24.4% | 7.0x |

ROC-AUC 0.7906 · Average precision 0.1891 · base fraud rate 3.48%.

**Honest caveats:** the test period was examined during earlier experiments, so these are exploratory, not an independent benchmark. Scores are uncalibrated and no production threshold is approved. See [Business Impact](docs/BUSINESS_IMPACT.md), [Model Card](docs/MODEL_CARD.md), [Evaluation Protocol](docs/EVALUATION.md) and the raw metrics in [`docs/results/`](docs/results/).

---

## Explainable by design: "why was this flagged?"

Every scored transaction comes with ranked, plain-language drivers computed as **Shapley values** through the real scikit-learn pipeline, so contributions add up exactly to *(model score − baseline score)*:

```
Score 0.221  (baseline 0.293, i.e. the score when no features are provided)
  Card Group Code (card2) = 360      strongly pushes the model score lower     -0.128
  Card Fingerprint (card1) = 13926   moderately pushes the model score higher  +0.044
  Transaction amount $25.00          slightly pushes the model score higher    +0.012
```

Exact enumeration is used for up to 8 supplied features, and deterministic permutation sampling beyond that. Scores are identical whether a row is scored alone or in a batch (regression-tested).

---

## Architecture

```
Transactions ─► Stage 1  Data Analyst Agent
                Stage 2  Data Guardian Review          (deterministic reviewer)
                Stage 3  Data Quality Gate             (reject malformed data)
                Stage 4  Fraud Pattern Engine          velocity · impossible travel ·
                                                       account takeover · new device
                Stage 5  Evidence Correlation Engine   overlap-aware scoring, ML corroboration
                Stage 6  Investigation Case Builder    risk level + required actions
                Stage 7  Fraud Pattern Analyst Agent   secure execution via policy + registry
                Stage 8  Fraud Pattern Guardian Review
                         Audit Ledger                  hash-linked, signed checkpoints
                         Dashboard / API
```

Reviewer components are **deterministic Python**, not LLM agents. They validate agent identity, schemas and evidence requirements before the pipeline advances.

## Key features

- **Pattern detection:** velocity attacks, impossible travel (24 supported countries), account takeover, new-device high-value, plus graph-based relationship analysis (NetworkX).
- **Evidence corroboration:** overlapping patterns are not treated as independent; ML scores only corroborate rule findings and never replace them. Evidence is graded STRONG / PARTIAL / NONE.
- **ML ranking (IEEE-CIS):** Extra Trees on 28 features, chronological validation, cost-sensitive threshold tooling, challenger comparison (logistic regression, XGBoost, LightGBM).
- **Security and audit:** least-privilege policy engine, operation registry, secure executor, SHA-256 artifact verification, hash-linked JSONL ledger with signed checkpoints and optional S3 Object Lock anchoring, Basic auth with rate limiting.
- **Analytics layer:** SQLite KPI warehouse, CSV exports for Power BI/Tableau and six SQL KPI queries ([`sql/kpis.sql`](sql/kpis.sql)).

---

## Quick start

### Option A: Docker

```bash
git clone https://github.com/AdityaGuleria7/FraudShield-AI.git
cd FraudShield-AI
cp .env.example .env            # then set FRAUDSHIELD_AUTH_PASSWORD
mkdir -p var/audit artifacts    # on Linux: sudo chown 1000:1000 var/audit
docker compose up --build
```

Open <http://localhost:8000>. Without a model in `artifacts/` the rule-based pipeline works fully and the IEEE scoring endpoints return a clear `503`.

### Option B: local Python

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
export FRAUDSHIELD_AUTH_USERNAME=admin FRAUDSHIELD_AUTH_PASSWORD="a-long-random-password"
python -m uvicorn app.dashboard:app --reload
```

### Get the model (optional)

The trained artifact and competition data are not in Git (Kaggle terms, file size). To create it:

1. Download the data from [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data) and place `train_transaction.csv` and `train_identity.csv` in `~/Downloads/ieee-fraud-detection/`.
2. `PYTHONPATH=. python scripts/build_ieee_enriched_artifact.py` writes `artifacts/ieee_extratrees_identity_enriched.joblib`.
3. Optionally pin it: `export FRAUDSHIELD_MODEL_SHA256=<sha256 of the file>`.

---

## Dashboard workflows

| Workflow | What it does |
|---|---|
| **Run Demo** | Synthetic transactions with an injected velocity attack, full pipeline, audit verification |
| **IEEE-CIS CSV Scoring** | Up to 5,000 rows scored by the experimental model, with plain-language drivers |
| **FraudShield Case CSV** | Your transactions in FraudShield's schema through the full investigation pipeline |

Schema rules for case CSVs: timestamps must be timezone-aware ISO-8601 (e.g. `2024-01-15T09:00:00Z`) and `country` must be one of the supported ISO codes. Three ready-to-run files are in [`examples/`](examples/) and are covered by automated tests.

```bash
python scripts/build_kpi_warehouse.py          # synthetic KPI warehouse + CSV exports
python scripts/review_queue_economics.py       # lift / cost-per-fraud table
```

## Running tests

```bash
PYTHONPATH=. python -m pytest -q      # 449 tests (446 in the base install; 3 research tests need xgboost)
                                      # no model file or Kaggle data required
```

Tests that need optional pieces (the trained artifact, `xgboost`) skip cleanly when they are absent. CI runs the suite on Python 3.12 and 3.13.

## Project structure

```
app/
  agents/ + guardians/   analyst and deterministic reviewer components
  audit/                 hash-linked ledger, signed checkpoints, S3 anchoring
  core/                  8-stage orchestrator
  data_quality/          validation gate
  explainability/        Shapley attribution + plain-language drivers
  features/ fraud/       feature engineering and pattern detectors
  ml/                    scorer, evaluation, threshold tooling, artifact integrity
  models/ security/      contracts, policy engine, secure executor
  templates/             dashboard HTML
docs/                    model card, evaluation, business impact, results/ (metrics JSON)
examples/ sql/ scripts/ tests/
```

## Security design

| Principle | Implementation |
|---|---|
| Evidence over consensus | Findings must satisfy provenance and quality checks, not just agree |
| Independent review | Guardian components review analyst output before the pipeline advances |
| Least privilege | Protected operations need explicit policy authorization |
| Structured contracts | Agent outputs validated against Pydantic schemas |
| Auditability | Every decision recorded in a hash-linked ledger |
| Artifact integrity | Model bytes verified (SHA-256) before deserialization |
| Fail safely | Invalid evidence, unauthorized operations and integrity failures surface as explicit errors |

A hash chain alone cannot stop an attacker with full access: for real tamper-evidence use S3 Object Lock with bucket versioning, restrictive permissions and key management ([`app/audit/external_checkpoints.py`](app/audit/external_checkpoints.py)).

## Limitations

- Synthetic data does not establish effectiveness on real traffic; injected scenarios are built for the rules to find.
- IEEE-CIS results are exploratory and the model uses a limited feature set (28 features).
- No live payment integrations. A flagged transaction is a signal for investigation, not proof of fraud.
- Human review, privacy controls, monitoring and governance are required before any high-impact use.

## Roadmap

- [x] Analyst/reviewer pipeline (8 stages), hash-linked audit ledger, signed checkpoints
- [x] IEEE-CIS model with chronological validation and challenger comparison (XGBoost, LightGBM)
- [x] Plain-language Shapley "why flagged?" explanations
- [x] Graph-based relationship analysis (NetworkX)
- [x] Docker deployment, CI, SQL/BI analytics layer
- [ ] Richer feature engineering and an untouched final holdout
- [ ] Probability calibration and a documented decision-threshold policy
- [ ] Interactive entity-relationship visualization
- [ ] OpenTelemetry latency tracing and API-key authentication

## Author

**Aditya Guleria** · GitHub: [AdityaGuleria7](https://github.com/AdityaGuleria7)

## Disclaimer

FraudShield AI is a research and portfolio project. It is not a guarantee of fraud prevention, a substitute for professional fraud investigation, or a claim of regulatory compliance. Validate the system and its controls before using it with real financial data or consequential decisions.
