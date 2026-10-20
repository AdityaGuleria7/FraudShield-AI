"""Translate review-queue metrics into analyst-friendly business numbers.

Reads the committed result files in docs/results/ and prints a markdown table:
lift over random review, recall, cost per fraud caught and an illustrative net
value. The two cost inputs are ASSUMPTIONS you control, not measured facts.

    python scripts/review_queue_economics.py --review-cost 5 --avg-fraud-loss 149.24
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "docs" / "results"


def compute(review_cost: float, avg_fraud_loss: float) -> list[dict]:
    model = json.loads((RESULTS / "ieee_identity_enriched_metrics.json").read_text())
    parts = json.loads((RESULTS / "ieee_three_period_validation.json").read_text())["partitions"]
    base_rate = parts["test_fraud_cases"] / parts["test_rows"]
    queues = model["models"]["extra_trees_identity_enriched"]["review_capacity"]
    rows = []
    for name, q in queues.items():
        reviewed, caught = q["transactions_reviewed"], q["frauds_found"]
        random_caught = reviewed * base_rate
        rows.append(
            {
                "queue": name.replace("top_", "Top ").replace("_percent", "%"),
                "reviewed": reviewed,
                "caught": caught,
                "precision": caught / reviewed,
                "lift": (caught / reviewed) / base_rate,
                "recall": q["recall"],
                "random_caught": random_caught,
                "cost_per_fraud": reviewed * review_cost / caught,
                "net_value": caught * avg_fraud_loss - reviewed * review_cost,
                "random_net_value": random_caught * avg_fraud_loss - reviewed * review_cost,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-cost", type=float, default=5.0, help="assumed cost per manual review")
    ap.add_argument("--avg-fraud-loss", type=float, default=149.24, help="assumed loss prevented per fraud caught")
    a = ap.parse_args()
    print(f"Assumptions: review cost {a.review_cost:.2f}/case, loss prevented {a.avg_fraud_loss:.2f}/fraud caught\n")
    print("| Queue | Reviewed | Fraud caught | Precision | Lift vs random | Recall | Random would catch | Cost / fraud caught | Net value | Net value (random) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in compute(a.review_cost, a.avg_fraud_loss):
        print(
            f"| {r['queue']} | {r['reviewed']:,} | {r['caught']:,} | {r['precision']:.1%} | {r['lift']:.1f}x | "
            f"{r['recall']:.1%} | {r['random_caught']:.0f} | {r['cost_per_fraud']:.2f} | "
            f"{r['net_value']:,.0f} | {r['random_net_value']:,.0f} |"
        )


if __name__ == "__main__":
    main()
