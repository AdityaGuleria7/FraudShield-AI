# FraudShield AI — Business Impact of the Review Queue

> **Read this first.** These numbers come from the exploratory IEEE-CIS experiment
> (chronological split, test period reused during development). They show how a
> ranked review queue behaves; they are **not** a forecast of production results.
> The cost inputs below are **assumptions you should replace with your own**.

## The question an operations team actually asks

*"We can only manually review a small slice of transactions. Does the model put fraud
at the top of the queue, and is reviewing that slice worth the analyst time?"*

Random review is the honest baseline. In the test period **3.48% of transactions were
fraud** (3,083 of 88,581), so reviewing 885 transactions at random would find about 31 frauds.

## Results (test period, 88,581 transactions)

Reproduce with `python scripts/review_queue_economics.py` (reads `docs/results/`).

| Queue | Reviewed | Fraud caught | Precision | Lift vs random | Recall | Random would catch |
|---|---:|---:|---:|---:|---:|---:|
| Top 1% | 885 | 315 | 35.6% | **10.2x** | 10.2% | 31 |
| Top 2% | 1,771 | 606 | 34.2% | 9.8x | 19.7% | 62 |
| Top 5% | 4,429 | 1,080 | 24.4% | 7.0x | 35.0% | 154 |

**Reading it:** the queue is about ten times richer in fraud than random review at the
top, and the lift decays as the queue grows, the usual precision/recall trade-off. Even
the top 5% queue only recovers about a third of fraud, so this model is a *prioritisation
aid for human reviewers*, not a standalone control.

## Illustrative economics

Assumptions (editable): **5.00 per manual review**; **149.24 loss prevented per fraud
caught** (the mean fraud transaction amount in the IEEE-CIS training data, in the
dataset's own currency units). Treating that mean as the loss for every caught fraud is
a simplification.

| Queue | Cost / fraud caught | Net value | Net value if reviewed at random |
|---|---:|---:|---:|
| Top 1% | 14.05 | 42,586 | 172 |
| Top 2% | 14.61 | 81,584 | 344 |
| Top 5% | 20.50 | 139,034 | 860 |

Net value = fraud caught × loss prevented − reviews × cost per review. A larger queue
earns more in absolute terms but each additional review is less productive, so the right
queue size depends on analyst capacity and your real cost-per-review. Re-run the script
with `--review-cost` and `--avg-fraud-loss` to test your own numbers.

## Why this is more honest than a single accuracy number

- Fraud is rare, so accuracy is meaningless (predicting "never fraud" scores ~96.5%).
- ROC-AUC (0.7906) and average precision (0.1891) are reported in the README and
  `docs/EVALUATION.md`, but *precision at a fixed review capacity* maps directly to an
  operational decision, so it is the headline metric here.
- The model uses only 28 features (6 core card/amount fields plus identity/device
  features available for ~24% of rows). Stronger feature engineering is the obvious
  next lever and is listed in the roadmap.

## Analytics layer (SQL / BI)

`scripts/build_kpi_warehouse.py` runs the pipeline on synthetic data and writes a SQLite
warehouse plus CSV exports (`warehouse/exports/`) that load straight into Power BI,
Tableau or Excel. `sql/kpis.sql` contains six ready-made KPI queries: detection KPIs,
findings by pattern, value at risk, flag rate by hour, riskiest accounts, and flagged value by
country and merchant (with a window function). The synthetic scenarios are injected for
the rule-based detectors to find, so 100% recall there demonstrates the plumbing, **not**
real-world detection performance.
