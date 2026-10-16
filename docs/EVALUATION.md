# FraudShield AI — Model Evaluation

## Status

**Experimental research results — not production validation.**

The IEEE-CIS results below come from local experiments on the public competition dataset. The chronological evaluation period was examined in earlier experiments and is not an untouched final test set. Do not describe these results as an independent benchmark or expected real-world performance.

## Reported exploratory comparison

Source reports:
- `reports/ieee_identity_ablation_metrics.json`
- `reports/ieee_feature_group_ablation.json`
- `reports/ieee_identity_enriched_metrics.json`

| Model configuration | Average precision | ROC-AUC | Fraud found in top 1% review queue | Top 2% | Top 5% |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.1439 | 0.7769 | 223 | 421 | 911 |
| Baseline + identity availability | 0.1525 | 0.7790 | 244 | 462 | 917 |
| Identity/device enriched | 0.1891 | 0.7906 | 315 | 606 | 1,080 |

The review queue sizes in the source reports are 885, 1,771, and 4,429 transactions, respectively. These figures describe the reported dataset split only.

## Interpretation limits

- Average precision and ROC-AUC measure different aspects of ranking performance; neither alone establishes business value.
- The reported test period was reused during development. Comparisons are exploratory and may be optimistic.
- The identity availability rate in the reports is approximately 24.42%. Missing identity data must not be treated as proof of fraud.
- The serialized ExtraTrees artifact is experimental. Its scores are uncalibrated and it has no approved production decision threshold.
- These reports do not establish causal relationships, production effectiveness, fairness, or performance on other datasets.
- The source reports do not provide confidence intervals for these comparisons. Do not imply statistical certainty.

## Evaluation protocol required before a final claim

1. Freeze the feature definitions, code revision, dependencies, and temporal boundaries before evaluation.
2. Train using the earliest period only.
3. Use a separate later validation period for model selection and calibration.
4. Select any operational threshold using documented costs and review capacity, without consulting the final test labels.
5. Evaluate once on a later, untouched test period.
6. Report average precision, ROC-AUC, precision and recall at review capacity, calibration, uncertainty intervals, and relevant subgroup/time-period checks.
7. Record the exact artifact hash and the complete evaluation configuration.
8. If the final test results influence further development, treat that test period as validation data and reserve a new later period for a future final evaluation.

No untouched final evaluation is claimed by this document.

## Reproducible challenger experiment workflow

The script `scripts/validate_ieee_challengers.py` compares logistic regression, ExtraTrees, XGBoost, and LightGBM using six selected transaction features. It creates chronological training, calibration, and test partitions. Thresholds are selected from calibration scores, not test labels.

### Preconditions

Before running an experiment:

1. Confirm the dataset exists at `~/Downloads/ieee-fraud-detection/train_transaction.csv`.
2. Commit the intended source changes and ensure the Git working tree is clean.
3. Choose a report path that does not already exist.
4. Treat the existing chronological test period as previously examined research data, not an independent final holdout.

The script checks for an existing output report and a clean Git working tree before loading data or training.

### Example command

Replace the date placeholder with a unique experiment identifier, and confirm that the report path does not already exist:

    FRAUDSHIELD_CHALLENGER_OUTPUT=reports/challenger_experiment_YYYYMMDD.json .venv/bin/python scripts/validate_ieee_challengers.py

Run from the repository root with the research dependencies installed.

### Metadata in new reports

New challenger reports record:

- An explicit exploratory status and `test_period_is_untouched: false`.
- Dataset filename, SHA-256, and row count after required-field filtering.
- Python and selected dependency versions.
- Current Git commit revision.
- Selected features, partition rules and sizes, temporal boundaries, threshold rules, and model metrics.

The Git revision identifies `HEAD`; it does not independently fingerprint source contents. The clean-working-tree check reduces this risk by requiring changes to be committed before a run. Preserve each report with its corresponding commit and dependency configuration.

### Interpretation limits

- Existing benchmark results are exploratory because the chronological evaluation period has already been examined during development.
- The 98th-percentile calibration threshold is a ranking rule, not a cost-optimized business decision threshold.
- Dataset fingerprints and environment metadata improve traceability but do not establish independent validation or real-world effectiveness.

## Requirements for a genuinely independent evaluation

A future independent evaluation requires a later dataset period or another holdout that has not influenced feature engineering, model selection, threshold selection, or other development decisions.

Before evaluation, freeze the dataset identity and temporal coverage, feature definitions and availability times, source revision, dependencies, model configuration, artifact hashes, training and calibration periods, and operational decision rule.

Evaluate the untouched period only after those decisions are frozen. Report average precision, ROC-AUC, precision and recall at realistic review capacities, calibration, uncertainty intervals, and relevant time-period and subgroup checks. If the holdout results influence further development, treat that period as validation data rather than an untouched final test.
