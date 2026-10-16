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
