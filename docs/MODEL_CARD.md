# FraudShield AI — Model Card

## Model identity

- **Model family:** Extra Trees classifier in a scikit-learn pipeline.
- **Artifact:** `artifacts/ieee_extratrees_identity_enriched.joblib`
- **Intended project use:** Research prototype for ranking transaction records for human fraud review.
- **Status:** Experimental; not approved for production decisions.
- **Artifact availability:** The trained artifact and competition data are local files and are not included in the source repository.

## Training and inputs

The current artifact metadata records 413,378 training rows and a training fraud rate of approximately 3.52%. The enriched pipeline uses 28 features, including `identity_available` and identity/device-derived features.

`identity_available` indicates whether a matching identity record was available. It must not be interpreted as evidence that a transaction is fraudulent. Identity coverage and feature definitions should be checked for every dataset and deployment context.

Only features available at prediction time should be used. Before retraining or deployment, audit all features and joins for target leakage, post-event information, and temporal leakage.

## Output semantics

The model produces an uncalibrated score for ranking transactions. The score must not be presented as a calibrated probability of fraud.

There is no approved production threshold. A score alone must not authorize a payment decline, account restriction, or other consequential action. Human review and a separately documented decision policy are required.

## Evaluation evidence

Exploratory IEEE-CIS reports show average precision of 0.1439 for the baseline and 0.1891 for the identity-enriched configuration, with ROC-AUC of 0.7769 and 0.7906, respectively.

These results use a chronological period that was examined during earlier experimentation. They are not an independent final benchmark and do not establish production performance. See [EVALUATION.md](EVALUATION.md) and [IEEE_CIS_RESEARCH.md](IEEE_CIS_RESEARCH.md) for context and limitations.

## Limitations and risks

- The training data comes from a public competition dataset and may not represent current transactions, real customer populations, or a target deployment.
- Fraud prevalence and behavior can change over time.
- Missing identity data can reflect data availability and must not be treated as proof of fraud.
- No claim of calibrated risk, fairness, causal effect, or production effectiveness is made.
- Subgroup performance, drift, operational costs, and the consequences of false positives and false negatives require further evaluation.

## Artifact integrity

The application verifies the model file's SHA-256 digest before deserializing it. The built-in digest is for the currently reviewed local artifact. A deliberately rebuilt or replaced artifact requires a separately verified, trusted digest to be configured before loading.

Hash verification helps detect unexpected artifact changes; it does not make an untrusted model safe, protect a compromised machine, or replace access controls for the artifact and its trusted hash configuration.

## Explainability

Per-transaction explanations are Shapley values computed over the *original input features* through the full scikit-learn pipeline (not the ~1,600 one-hot columns). The baseline is the model score when no supplied feature is provided, and contributions sum exactly to `model score - baseline score`. Features a caller did not supply receive no driver. Explanations describe this model's uncalibrated score; they do not confirm fraud, establish causality, or prove fairness.

Raw metrics behind every figure in this card are committed in `docs/results/`.
