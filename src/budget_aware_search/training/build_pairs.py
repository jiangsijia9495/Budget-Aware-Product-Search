from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from budget_aware_search.labels import validate_esci_labels
from budget_aware_search.utils import write_json


def _sample_by_label_priority(
    group: pd.DataFrame,
    *,
    positive_product_id: str,
    positive_text: str,
    positive_label: str,
    labels: Sequence[str],
    count: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample negatives from the first available relevance levels in priority order.

    For example, with labels=("I", "C"), irrelevant products are sampled first. If
    there are not enough, complements fill the remaining slots. A product with the
    same relevance label as the positive is never used as an explicit negative.
    """
    selected: list[pd.DataFrame] = []
    remaining = count
    for label in labels:
        if remaining <= 0:
            break
        if label == positive_label:
            continue
        candidate_text = group["product_text"].fillna("").astype(str).str.strip()
        normalized_positive_text = str(positive_text).strip()
        pool = group.loc[
            (group["esci_label"] == label)
            & (group["product_id"].astype(str) != str(positive_product_id))
            & candidate_text.ne(normalized_positive_text)
        ].drop_duplicates("product_id")
        if pool.empty:
            continue
        take = min(remaining, len(pool))
        sampled = pool.sample(
            n=take,
            random_state=int(rng.integers(0, 2**31 - 1)),
            replace=False,
        )
        selected.append(sampled)
        remaining -= take
    if not selected:
        return group.iloc[0:0].copy()
    return pd.concat(selected, ignore_index=True)


def build_biencoder_pairs(
    frame: pd.DataFrame,
    seed: int = 42,
    max_positives_per_query: int = 1,
    negatives_per_positive: int = 1,
    allow_substitute_positive_when_no_exact: bool = True,
    negative_label_priority: Sequence[str] = ("I", "C"),
) -> pd.DataFrame:
    required = {"query_id", "query", "product_id", "product_text", "esci_label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    if max_positives_per_query <= 0:
        raise ValueError("max_positives_per_query must be positive")
    if negatives_per_positive <= 0:
        raise ValueError("negatives_per_positive must be positive")

    labels = tuple(str(label).strip().upper() for label in negative_label_priority)
    validate_esci_labels(set(labels))
    if not labels:
        raise ValueError("negative_label_priority must contain at least one label")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for query_id, group in frame.groupby("query_id", sort=False):
        exact = group.loc[group["esci_label"] == "E"]
        positives = exact
        positive_label = "E"
        if positives.empty and allow_substitute_positive_when_no_exact:
            positives = group.loc[group["esci_label"] == "S"]
            positive_label = "S"
        if positives.empty:
            continue

        positives = positives.sample(
            n=min(max_positives_per_query, len(positives)),
            random_state=int(rng.integers(0, 2**31 - 1)),
            replace=False,
        )
        for _, positive in positives.iterrows():
            sampled = _sample_by_label_priority(
                group,
                positive_product_id=str(positive["product_id"]),
                positive_text=str(positive["product_text"]),
                positive_label=positive_label,
                labels=labels,
                count=negatives_per_positive,
                rng=rng,
            )
            for _, negative in sampled.iterrows():
                rows.append(
                    {
                        "query_id": query_id,
                        "anchor": str(positive["query"]),
                        "positive": str(positive["product_text"]),
                        "negative": str(negative["product_text"]),
                        "positive_product_id": str(positive["product_id"]),
                        "negative_product_id": str(negative["product_id"]),
                        "positive_label": positive_label,
                        "negative_label": str(negative["esci_label"]),
                    }
                )
    return pd.DataFrame(rows)


def _parse_label_priority(raw: str) -> tuple[str, ...]:
    labels = tuple(part.strip().upper() for part in raw.split(",") if part.strip())
    if not labels:
        raise argparse.ArgumentTypeError("At least one label is required, e.g. I,C")
    try:
        validate_esci_labels(set(labels))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Build query/positive/negative training triples.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-positives-per-query", type=int, default=1)
    parser.add_argument("--negatives-per-positive", type=int, default=1)
    parser.add_argument("--exact-only", action="store_true")
    parser.add_argument(
        "--negative-label-priority",
        type=_parse_label_priority,
        default=("I", "C"),
        help="Comma-separated priority order for explicit negatives. Default: I,C. "
        "Substitutes are intentionally excluded from the v1 random-negative run.",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    pairs = build_biencoder_pairs(
        frame,
        seed=args.seed,
        max_positives_per_query=args.max_positives_per_query,
        negatives_per_positive=args.negatives_per_positive,
        allow_substitute_positive_when_no_exact=not args.exact_only,
        negative_label_priority=args.negative_label_priority,
    )
    if pairs.empty:
        raise RuntimeError("No training pairs were generated")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(output, index=False)
    write_json(
        {
            "rows": int(len(pairs)),
            "queries": int(pairs["query_id"].nunique()),
            "positive_labels": pairs["positive_label"].value_counts().to_dict(),
            "negative_labels": pairs["negative_label"].value_counts().to_dict(),
            "negative_label_priority": list(args.negative_label_priority),
            "max_positives_per_query": args.max_positives_per_query,
            "negatives_per_positive": args.negatives_per_positive,
            "seed": args.seed,
        },
        output.with_suffix(".summary.json"),
    )
    print(f"Saved {len(pairs):,} training triples to {output}")


if __name__ == "__main__":
    main()
