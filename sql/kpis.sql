-- FraudShield KPI queries (SQLite). Build the DB first:
--   PYTHONPATH=. python scripts/build_kpi_warehouse.py
-- Synthetic data: these demonstrate the analytics layer, not real-world results.

-- 1. Headline detection KPIs
SELECT COUNT(*)                                             AS transactions,
       SUM(is_fraud)                                        AS injected_fraud,
       SUM(flagged)                                         AS flagged,
       ROUND(100.0 * SUM(flagged) / COUNT(*), 2)            AS flag_rate_pct,
       ROUND(100.0 * SUM(CASE WHEN is_fraud=1 AND flagged=1 THEN 1 END)
             / NULLIF(SUM(is_fraud), 0), 1)                 AS recall_pct,
       ROUND(100.0 * SUM(CASE WHEN is_fraud=1 AND flagged=1 THEN 1 END)
             / NULLIF(SUM(flagged), 0), 1)                  AS precision_pct
FROM transactions;

-- 2. Findings by pattern: volume and accounts affected
SELECT pattern,
       COUNT(*)                    AS findings,
       COUNT(DISTINCT account_id)  AS accounts_affected
FROM findings
GROUP BY pattern
ORDER BY findings DESC;

-- 3. Value at risk: amount on flagged transactions, by scenario
SELECT COALESCE(fraud_scenario, 'none')  AS scenario,
       COUNT(*)                          AS flagged_txns,
       ROUND(SUM(amount), 2)             AS flagged_amount,
       ROUND(AVG(amount), 2)             AS avg_amount
FROM transactions
WHERE flagged = 1
GROUP BY 1
ORDER BY flagged_amount DESC;

-- 4. Flag rate by hour of day (UTC)
SELECT hour_utc,
       COUNT(*)                                  AS transactions,
       SUM(flagged)                              AS flagged,
       ROUND(100.0 * SUM(flagged) / COUNT(*), 2) AS flag_rate_pct
FROM transactions
GROUP BY hour_utc
ORDER BY hour_utc;

-- 5. Riskiest accounts: ranked by flagged value
SELECT account_id,
       COUNT(*)                         AS flagged_txns,
       ROUND(SUM(amount), 2)            AS flagged_amount,
       GROUP_CONCAT(DISTINCT fraud_scenario) AS scenarios
FROM transactions
WHERE flagged = 1
GROUP BY account_id
ORDER BY flagged_amount DESC
LIMIT 10;

-- 6. Flag rate by country and merchant category (window function: share of flagged value)
SELECT country,
       merchant_category,
       SUM(flagged)                                       AS flagged,
       ROUND(SUM(CASE WHEN flagged=1 THEN amount END), 2) AS flagged_amount,
       ROUND(100.0 * SUM(CASE WHEN flagged=1 THEN amount END)
             / SUM(SUM(CASE WHEN flagged=1 THEN amount END)) OVER (), 1) AS pct_of_flagged_value
FROM transactions
GROUP BY country, merchant_category
HAVING SUM(flagged) > 0
ORDER BY flagged_amount DESC;
