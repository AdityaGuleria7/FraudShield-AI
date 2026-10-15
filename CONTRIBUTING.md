# Contributing to FraudShield AI

Thank you for your interest! This document covers setup, conventions, and the test suite.

## Setup

```bash
git clone https://github.com/AdityaGuleria7/FraudShield-AI.git
cd FraudShield-AI

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
# Edit .env — set FRAUDSHIELD_AUTH_USERNAME and FRAUDSHIELD_AUTH_PASSWORD
```

## Running Tests

```bash
python -m pytest -q
```

The suite must pass (305+ tests) before submitting a pull request.

Run a specific module:

```bash
python -m pytest tests/test_evidence_engine.py -v
python -m pytest tests/test_policy_engine.py -v
```

## Code Conventions

- Python 3.13 compatible
- Type annotations on all public functions
- Docstrings on classes and non-trivial methods
- No hardcoded secrets — use environment variables
- New fraud detection patterns must include:
  - Detection logic in `app/fraud/`
  - `provenance` dict with `detector`, `method`, and `version` fields
  - Tests in `tests/`

## Adding a New Fraud Pattern

1. Implement in `app/fraud/` (see `pattern_engine.py` for examples)
2. Register the pattern weight in `EvidenceEngine.PATTERN_WEIGHTS`
3. Add `DETECTOR_BY_PATTERN` mapping
4. Define required evidence fields in `EvidenceEngine.assess()`
5. Write tests in `tests/test_pattern_engine.py` and `tests/test_evidence_engine.py`

## Pull Request Checklist

- [ ] `python -m pytest -q` passes
- [ ] New code has type annotations
- [ ] No secrets, API keys, or real data committed
- [ ] `docs/` updated if architecture changes
