from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from budget_aware_search.utils import write_json


def mine_from_candidate_scores(
    scored: pd.DataFrame,
    negatives_per_positive: int = 1,
    max_positives_per_query: int = 1,
    allow_complements: bool = True,
    allow_substitutes: bool = False,
) -> pd.DataFrame:
    required = {"query_id", "query", "product_id", "product_text", "esci_label", "score"}
    missing = required.difference(scored.columns)
    if missing:
        raise ValueError(f"Scored data is missing columns: {sorted(missing)}")

    negative_labels = ["I"]
    if allow_complements:
        negative_labels.append("C")
    if allow_substitutes:
        negative_labels.append("S")

    rows: list[dict[str, object]] = []
    for query_id, group in scored.groupby("query_id", sort=False):
        positives = group.loc[group["esci_label"] == "E"].sort_values("score", ascending=False)
        negatives = group.loc[group["esci_label"].isin(negative_labels)].sort_values(
            "score", ascending=False
        )
        if positives.empty or negatives.empty:
            continue
        positives = positives.head(max_positives_per_query)
        for _, positive in positives.iterrows():
            selected_negatives = negatives.loc[
                negatives["product_id"] != positive["product_id"]
            ].head(negatives_per_positive)
            for negative_rank, (_, negative) in enumerate(selected_negatives.iterrows(), start=1):
                rows.append(
                    {
                        "query_id": query_id,
                        "anchor": str(positive["query"]),
                        "positive": str(positive["product_text"]),
                        "negative": str(negative["product_text"]),
                        "positive_product_id": str(positive["product_id"]),
                        "negative_product_id": str(negative["product_id"]),
                        "negative_label": str(negative["esci_label"]),
                        "negative_model_score": float(negative["score"]),
                        "negative_rank": negative_rank,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine highest-scoring false candidates as hard negatives.")
    parser.add_argument("--scored-train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--negatives-per-positive", type=int, default=1)
    parser.add_argument("--max-positives-per-query", type=int, default=1)
    parser.add_argument("--no-complements", action="store_true")
    parser.add_argument("--allow-substitutes", action="store_true")
    args = parser.parse_args()

    scored = pd.read_parquet(args.scored_train)
    pairs = mine_from_candidate_scores(
        scored,
        negatives_per_positive=args.negatives_per_positive,
        max_positives_per_query=args.max_positives_per_query,
        allow_complements=not args.no_complements,
        allow_substitutes=args.allow_substitutes,
    )
    if pairs.empty:
        raise RuntimeError("No hard-negative triples were generated")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(output, index=False)
    write_json(
        {
            "rows": int(len(pairs)),
            "queries": int(pairs["query_id"].nunique()),
            "negative_labels": pairs["negative_label"].value_counts().to_dict(),
        },
        output.with_suffix(".summary.json"),
    )
    print(f"Saved {len(pairs):,} hard-negative triples to {output}")


if __name__ == "__main__":
    main()
