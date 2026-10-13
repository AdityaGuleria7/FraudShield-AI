# IEEE-CIS research and evaluation plan

This document separates public competition references from results measured in this repository. It is a reproducibility guide, not a claim of production effectiveness.

## Public references

- [Official IEEE-CIS Fraud Detection competition](https://www.kaggle.com/competitions/ieee-fraud-detection/overview)
- [Competition data and rules](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
- [Competition leaderboard](https://www.kaggle.com/competitions/ieee-fraud-detection/leaderboard)
- [FraudSquad first-place solution write-up](https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/fraudsquad-1st-place-solution-part-2)
- [KovalevEvgeny solution repository](https://github.com/KovalevEvgeny/kaggle-fraud-detection)
- [VadimTTT time-aware validation repository](https://github.com/VadimTTT/ieee-cis-fraud-detection)

The competition predicts the `isFraud` target and uses ROC-AUC for leaderboard evaluation. FraudShield also reports average precision because the positive class is rare and investigators often need to prioritize a limited review queue. These metrics answer different questions and should be reported together.

## Lessons to evaluate, not blindly copy

Winning public solutions use ensembles and extensive feature engineering. FraudShield should test those ideas under its own chronological holdout rather than selecting a model from leaderboard rank alone.

1. **Compare diverse learners.** Keep the current ExtraTrees model as a baseline. Evaluate LightGBM and CatBoost as optional challenger models, then test a blend only if it improves untouched validation results and stability.
2. **Engineer transaction context carefully.** Card/client-level counts, amount deviations, time-window velocity, and identity/device signals can be useful, but every statistic must be computed using only information available at prediction time.
3. **Prevent leakage.** Do not use `isFraud` or post-event fields as features. Keep related rows and temporal boundaries in mind when constructing aggregates. Fit imputers, encoders, and other preprocessing only on the training period.
4. **Respect missing identity data.** An unmatched identity record is common in this dataset. Missing identity must not itself be treated as proof of fraud. The dashboard's `identity_available` flag is metadata and is not a model feature in the current artifact.
5. **Use chronological evaluation.** Preserve a final later time period that is not used to choose features, models, thresholds, or ensembles. Repeatedly tuning against the same holdout turns it into validation data.
6. **Report operational metrics.** In addition to ROC-AUC and average precision, report precision/recall at a stated review capacity (for example, top 1%, 2%, and 5%), confusion matrices at a predeclared threshold, score calibration, subgroup performance, and stability across time.
7. **Keep decisions separate from scores.** A leaderboard score is not a calibrated probability, an approved threshold, or authorization to decline a payment or restrict an account.

## Current repository results

Prior exploratory runs recorded these results for the same chronological test period:

| Model | Average precision | ROC-AUC | Fraud in top 2% | Fraud in top 5% |
|---|---:|---:|---:|---:|
| Baseline | 0.1439 | 0.7769 | 421 | 911 |
| Identity-enriched ExtraTrees | 0.1891 | 0.7906 | 606 | 1,080 |

These are exploratory internal results, not an independent final benchmark. The test period was reused during experimentation, so it must not be described as a pristine untouched holdout. Re-run a frozen evaluation on a later reserved period before making comparative performance claims in a portfolio or approving any model for operational use.

## Reproduce locally

1. Download the competition data through Kaggle while signed in and following the competition rules. Do not commit the raw data to Git.
2. Place `train_transaction.csv` and `train_identity.csv` in `~/Downloads/ieee-fraud-detection/`.
3. Run the existing baseline, identity-enriched training, temporal validation, and feature-ablation scripts in `scripts/`. Review each script's split strategy and output before comparing numbers.
4. Save reports under `reports/` and model artifacts under `artifacts/`; both are excluded from the source archive and should remain out of Git.
5. Freeze the model, feature list, split boundaries, metrics, and environment before the final evaluation. Keep model artifact metadata explicit about training period, features, score semantics, and approval status.

## Current limitations

- The source archive intentionally excludes the trained model artifact and raw competition data. IEEE-CIS scoring will return an unavailable response until the artifact is generated locally.
- The current scoring artifact is an experimental ExtraTrees model with uncalibrated scores and no approved production threshold.
- LightGBM/CatBoost challenger training and a new untouched holdout have not been executed as part of this source review.
- Kaggle solutions may have different licenses, dependencies, split choices, and post-processing assumptions. Read the original license and reproduce ideas independently before reusing code.
