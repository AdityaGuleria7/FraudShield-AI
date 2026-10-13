# FraudShield AI

### Evidence-Driven, Adversarial Multi-Agent Fraud Detection

FraudShield AI is a Python-based fraud detection and investigation project designed around one core principle:

**No trust without verification. Evidence over consensus.**

The project combines transaction feature engineering, fraud-pattern analysis, machine-learning evaluation, investigation case building, agent review, policy enforcement, and tamper-evident audit records.

Rather than allowing one agent to make consequential decisions independently, FraudShield uses structured agent contracts, independent reviewers, policy checks, and auditable decision paths.

> **Project status:** Research and portfolio project under active development. The repository contains implemented modules and automated tests. Production deployment, live payment integrations, and real-world fraud-prevention effectiveness are not claimed.

## Table of Contents

- [Key Features](#key-features)
- [Architecture](#architecture)
- [Security Principles](#security-principles)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Audit Integrity and External Checkpoints](#audit-integrity-and-external-checkpoints)
- [Limitations](#limitations)
- [IEEE-CIS Research Notes](docs/IEEE_CIS_RESEARCH.md)\n- [Roadmap](#roadmap)

## Key Features

### 1. Transaction Data and Quality
- Synthetic transaction generation for controlled testing.
- Transaction data validation and quality checks.
- Feature construction for behavioral, temporal, category, velocity, and relationship signals.

### 2. Fraud Detection and Pattern Analysis
- Fraud-pattern analysis and evidence evaluation.
- Account-takeover scenario detection.
- Impossible-travel signal analysis.
- Scenario generation for testing detection logic.
- Evidence aggregation to support investigation workflows.

### 3. Machine Learning
- Baseline model and dataset pipelines.
- Model evaluation and validation.
- Holdout-validation tests.
- Evidence correlation and corroboration metadata.
- Threshold optimization and threshold policies.

### 4. Multi-Agent Review
- Data analyst and fraud-pattern analyst agents.
- Guardian reviewers for data quality and fraud-pattern findings.
- Structured agent contracts and orchestration.
- Tests for agent identity, authorization, and orchestration behavior.

### 5. Security and Policy Enforcement
- Policy engine for authorizing or denying operations.
- Operation registry and integrity checks.
- Secure execution controls.
- Least-privilege-oriented authorization tests.

### 6. Auditable Decisions
- Hash-linked audit ledger.
- Signed checkpoint generation and validation.
- Protection against certain checkpoint rollback and rewritten-history scenarios.
- S3 external checkpoint anchoring implementation with signature verification and retention settings.

## Architecture

The high-level workflow is:

1. **Generate or receive transaction data.**
2. **Validate data quality** before downstream processing.
3. **Build features** describing transaction behavior, timing, categories, velocity, and relationships.
4. **Analyze fraud signals** using pattern detection and machine-learning components.
5. **Corroborate evidence** rather than relying on a single signal or agent.
6. **Review findings** through guardian agents and orchestration controls.
7. **Apply policy checks** before protected operations are permitted.
8. **Record relevant events** in the audit ledger and use signed checkpoints to help detect historical changes.

This describes the intended processing flow across the available modules; it does not imply that every component is connected to a production service or external payment platform.

## Security Principles

FraudShield AI is designed around the following principles:

- **Evidence over consensus:** multiple agents agreeing is not sufficient proof that a finding is correct.
- **Independent review:** guardian components can review analyst findings.
- **Least privilege:** protected operations should be limited by explicit authorization rules.
- **Structured contracts:** agent outputs should satisfy defined contracts before downstream use.
- **Auditability:** important processing steps should be traceable.
- **Fail safely:** invalid evidence, unauthorized operations, and integrity failures should not silently be accepted.

Security tests provide evidence for the specific scenarios they exercise. They do not establish that the entire application is secure against every attack.

## Technology Stack

The repository's pinned dependencies include:

- **Python** — application logic and orchestration.
- **Pandas and NumPy** — data processing and numerical operations.
- **scikit-learn and SciPy** — machine-learning and scientific-computing components.
- **Pydantic** — data validation and structured contracts.
- **LangChain and LangGraph** — available agent and workflow tooling.
- **SQLAlchemy and psycopg** — database-access tooling.
- **FastAPI and Uvicorn** — API framework and server dependencies.
- **OpenTelemetry API** — observability instrumentation interface.
- **pytest** — automated testing.

A dependency being installed does not necessarily mean that its corresponding production integration is complete or configured.

## Getting Started

### Prerequisites

- Python compatible with the pinned dependencies.
- Git.
- A terminal.
- macOS, Linux, or another compatible Python environment.

The project has been tested locally using Python 3.13.2 in a virtual environment.

### 1. Clone the repository

```bash
git clone https://github.com/AdityaGuleria7/FraudShield-AI.git
cd FraudShield-AI
```

### 2. Create a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Upgrade pip and install the pinned requirements:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If a pinned package does not support your Python version or platform, resolve that compatibility issue in a dedicated environment before changing dependency pins.

### 4. Run the test suite

```bash
python -m pytest -q
```

The source review baseline is **276 passing tests**. Re-run the suite in your own environment after any changes.

### 5. Explore the modules

Start with these files:

- `app/core/orchestrator.py` — orchestration.
- `app/fraud/evidence_engine.py` — fraud evidence evaluation.
- `app/features/builder.py` — feature construction.
- `app/ml/pipeline.py` — machine-learning pipeline.
- `app/security/policy_engine.py` — policy enforcement.
- `app/audit/ledger.py` — audit ledger.
- `app/audit/checkpoints.py` — signed checkpoints.
- `app/audit/external_checkpoints.py` — external checkpoint anchoring.
- `tests/` — automated behavior and security tests.

### 6. Launch the dashboard

```bash
python -m uvicorn app.dashboard:app --reload
```

Open `http://127.0.0.1:8000`. The synthetic demo and internal-schema case CSV workflow do not require the IEEE-CIS model artifact. IEEE-CIS scoring requires the separately trained artifact at `artifacts/ieee_extratrees_identity_enriched.joblib`; without it, the scoring API returns a clear unavailable response.

The primary dashboard workflow is batch-scoring a compatible IEEE-CIS feature CSV. Single-transaction scoring and FraudShield's separate case-CSV schema are available under collapsed **Advanced** sections.

## Running Tests

Run the full suite:

```bash
python -m pytest -q
```

Run checkpoint and audit-related tests:

```bash
python -m pytest tests/test_signed_checkpoints.py tests/test_audit_ledger.py -q
```

Run policy and secure-execution tests:

```bash
python -m pytest tests/test_policy_engine.py tests/test_secure_executor.py tests/test_secure_executor_integrity.py -q
```

Run feature and fraud-detection tests:

```bash
python -m pytest tests/test_feature_builder.py tests/test_evidence_engine.py tests/test_pattern_engine.py -q
```

Tests are organized around data quality, features, fraud scenarios, model validation, agent contracts, authorization, orchestration, and audit integrity.

## Project Structure

```text
FraudShield-AI/
├── app/
│   ├── agents/
│   │   └── guardians/       # Independent review components
│   ├── audit/               # Ledger, signatures, external checkpoints
│   ├── core/                # Orchestration
│   ├── data/                # Transaction data generation
│   ├── data_quality/        # Data validation
│   ├── features/             # Feature engineering
│   ├── fraud/                # Fraud patterns and evidence
│   ├── investigation/        # Investigation case construction
│   ├── ml/                   # Models, evaluation, thresholds
│   ├── models/               # Agent contracts
│   └── security/             # Policies and secure execution
├── tests/                    # Automated tests
├── requirements.txt          # Pinned Python dependencies
├── .gitignore
└── README.md
```

## Audit Integrity and External Checkpoints

The audit subsystem includes a hash-linked ledger and signed checkpoints.

The local checkpoint store checks that a previously trusted anchor remains consistent with the ledger before advancing it. The external anchoring module can publish signed checkpoints to S3 and compare a local ledger against the highest checkpoint it discovers.

### Important operational requirements

External anchoring is not automatically active just because the module exists. Actual deployment requires appropriate AWS configuration and a compatible S3 client.

For meaningful protection, configure and verify:

- S3 Object Lock and bucket versioning.
- Appropriate retention settings and restrictive bucket permissions.
- Secure storage and rotation procedures for signing keys.
- Independent protection of the checkpoint discovery mechanism.
- Monitoring and alerting for verification failures.
- Recovery procedures and documented handling of invalid audit history.

A hash chain alone cannot prevent an attacker with sufficient access from rewriting the entire chain. A signed checkpoint is only as trustworthy as its key management and the independence of its anchor. The implementation must also be reviewed against the target AWS environment before production use.

## Limitations

- Synthetic or test data does not establish effectiveness on real transaction traffic.
- No fraud-detection accuracy, recall, precision, or financial-loss reduction is claimed without measured evaluation results.
- A detected anomaly is a signal for investigation, not proof of fraud.
- The presence of agent, API, database, and observability dependencies does not imply that production integrations are deployed.
- S3 checkpoint anchoring requires explicit configuration and operational verification.
- The current test suite covers specific tested behaviors; it is not a complete security certification.
- Human review, privacy controls, model monitoring, incident response, and operational governance are necessary before high-impact real-world use.

## Roadmap

Potential next milestones:

- [ ] Document and demonstrate a complete end-to-end transaction workflow.
- [ ] Add reproducible evaluation on an appropriately sourced and documented dataset.
- [ ] Publish measured precision, recall, false-positive rates, and threshold trade-offs.
- [ ] Add a documented API entry point and example requests.
- [ ] Add structured logs, tracing, and operational health checks.
- [ ] Add integration tests for an isolated S3 test environment.
- [ ] Document threat modeling, key rotation, and recovery procedures.
- [ ] Build a safe demonstration interface with synthetic transactions and explainable evidence.

## Disclaimer

FraudShield AI is a software research and portfolio project. It is not a guarantee of fraud prevention, a substitute for professional fraud investigation, or a claim of regulatory compliance. Validate the system, its data handling, and its controls before using it with real financial data or consequential decisions.

## Author

**Aditya Guleria**

GitHub: https://github.com/AdityaGuleria7
