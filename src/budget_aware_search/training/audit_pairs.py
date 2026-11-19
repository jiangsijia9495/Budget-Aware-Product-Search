from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from budget_aware_search.labels import ESCI_GAINS, validate_esci_labels
from budget_aware_search.utils import write_json


def audit_pairs(frame: pd.DataFrame) -> dict[str, object]:
    required = {
        "query_id",
        "anchor",
        "positive",
        "negative",
        "positive_product_id",
        "negative_product_id",
        "positive_label",
        "negative_label",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Pair file is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Pair file is empty")

    validate_esci_labels(set(frame["positive_label"].astype(str).unique()))
    validate_esci_labels(set(frame["negative_label"].astype(str).unique()))

    same_product = frame["positive_product_id"].astype(str).eq(
        frame["negative_product_id"].astype(str)
    )
    same_text = frame["positive"].astype(str).eq(frame["negative"].astype(str))
    non_decreasing_gain = frame.apply(
        lambda row: ESCI_GAINS[str(row["negative_label"])]
        >= ESCI_GAINS[str(row["positive_label"])],
        axis=1,
    )
    empty_anchor = frame["anchor"].fillna("").astype(str).str.strip().eq("")
    empty_positive = frame["positive"].fillna("").astype(str).str.strip().eq("")
    empty_negative = frame["negative"].fillna("").astype(str).str.strip().eq("")

    exact_duplicate_rows = frame.duplicated(
        ["query_id", "positive_product_id", "negative_product_id"]
    )
    query_counts = frame.groupby("query_id").size()

    summary: dict[str, object] = {
        "rows": int(len(frame)),
        "queries": int(frame["query_id"].nunique()),
        "rows_per_query": {
            "min": int(query_counts.min()),
            "mean": float(query_counts.mean()),
            "max": int(query_counts.max()),
        },
        "positive_labels": frame["positive_label"].value_counts().to_dict(),
        "negative_labels": frame["negative_label"].value_counts().to_dict(),
        "same_product_rows": int(same_product.sum()),
        "same_text_rows": int(same_text.sum()),
        "negative_gain_ge_positive_gain_rows": int(non_decreasing_gain.sum()),
        "empty_anchor_rows": int(empty_anchor.sum()),
        "empty_positive_rows": int(empty_positive.sum()),
        "empty_negative_rows": int(empty_negative.sum()),
        "duplicate_triplet_rows": int(exact_duplicate_rows.sum()),
        "anchor_char_length": {
            "mean": float(frame["anchor"].astype(str).str.len().mean()),
            "p95": float(frame["anchor"].astype(str).str.len().quantile(0.95)),
            "max": int(frame["anchor"].astype(str).str.len().max()),
        },
        "positive_char_length": {
            "mean": float(frame["positive"].astype(str).str.len().mean()),
            "p95": float(frame["positive"].astype(str).str.len().quantile(0.95)),
            "max": int(frame["positive"].astype(str).str.len().max()),
        },
        "negative_char_length": {
            "mean": float(frame["negative"].astype(str).str.len().mean()),
            "p95": float(frame["negative"].astype(str).str.len().quantile(0.95)),
            "max": int(frame["negative"].astype(str).str.len().max()),
        },
    }
    critical = (
        int(same_product.sum())
        + int(same_text.sum())
        + int(non_decreasing_gain.sum())
        + int(empty_anchor.sum())
        + int(empty_positive.sum())
        + int(empty_negative.sum())
        + int(exact_duplicate_rows.sum())
    )
    summary["passed"] = critical == 0
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit bi-encoder triplets before training.")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    path = Path(args.pairs)
    frame = pd.read_parquet(path)
    summary = audit_pairs(frame)
    output = Path(args.output) if args.output else path.with_suffix(".audit.json")
    write_json(summary, output)

    print(f"Rows: {summary['rows']:,}")
    print(f"Queries: {summary['queries']:,}")
    print(f"Positive labels: {summary['positive_labels']}")
    print(f"Negative labels: {summary['negative_labels']}")
    print(f"Audit passed: {summary['passed']}")
    print(f"Saved audit to {output}")
    if not summary["passed"]:
        raise SystemExit("Pair audit failed; inspect the JSON before training.")


if __name__ == "__main__":
    main()
