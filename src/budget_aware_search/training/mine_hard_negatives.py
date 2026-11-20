from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from budget_aware_search.training.text_normalization import normalize_model_text
from budget_aware_search.utils import write_json


OUTPUT_COLUMNS = [
    "query_id",
    "anchor",
    "positive",
    "negative",
    "positive_product_id",
    "negative_product_id",
    "positive_label",
    "negative_label",
    "positive_model_score",
    "negative_model_score",
    "score_margin",
    "negative_rank",
]


def mine_from_candidate_scores(
    scored: pd.DataFrame,
    negatives_per_positive: int = 1,
    max_positives_per_query: int = 1,
    allow_complements: bool = True,
    allow_substitutes: bool = False,
) -> pd.DataFrame:
    """Mine hard-negative triples from *training-only* scored ESCI candidates.

    Choose the highest-model-score valid I negative first; fall back to C,
    then (only when explicitly enabled) S. Exclude duplicate product IDs,
    normalized-identical positive/negative text, and repeated negative text.
    The normalizer must be identical to the one used in audit_pairs.py.
    """
    required = {
        "query_id", "query", "product_id", "product_text", "esci_label", "score"
    }
    missing = required.difference(scored.columns)
    if missing:
        raise ValueError(f"Scored data is missing columns: {sorted(missing)}")
    if negatives_per_positive < 1 or max_positives_per_query < 1:
        raise ValueError("Both per-positive and per-query limits must be >= 1")

    # Label priority is intentional: maintain the same I -> C negative-label
    # preference as v1, changing primarily the score-based selection difficulty.
    negative_labels = ["I"]
    if allow_complements:
        negative_labels.append("C")
    if allow_substitutes:
        negative_labels.append("S")

    frame = scored.copy()
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame = frame.dropna(subset=["score"])
    frame["_product_id_key"] = frame["product_id"].astype(str)
    frame["_text_key"] = frame["product_text"].fillna("").map(normalize_model_text)
    frame = frame.loc[frame["_text_key"] != ""]

    rows: list[dict[str, object]] = []
    for query_id, group in frame.groupby("query_id", sort=False):
        positives = (
            group.loc[group["esci_label"] == "E"]
            .sort_values(["score", "_product_id_key"], ascending=[False, True], kind="mergesort")
            .head(max_positives_per_query)
        )
        if positives.empty:
            continue

        for _, positive in positives.iterrows():
            positive_id = positive["_product_id_key"]
            positive_text_key = positive["_text_key"]
            selected_count = 0
            seen_negative_texts: set[str] = set()

            for label in negative_labels:
                # Order by model score within each relevance tier; this is what
                # makes the selected examples hard instead of random.
                candidates = group.loc[group["esci_label"] == label].sort_values(
                    ["score", "_product_id_key"],
                    ascending=[False, True],
                    kind="mergesort",
                )
                for _, negative in candidates.iterrows():
                    negative_id = negative["_product_id_key"]
                    negative_text_key = negative["_text_key"]

                    if negative_id == positive_id:
                        continue
                    if negative_text_key == positive_text_key:
                        continue
                    if negative_text_key in seen_negative_texts:
                        continue

                    seen_negative_texts.add(negative_text_key)
                    selected_count += 1
                    rows.append(
                        {
                            "query_id": query_id,
                            "anchor": str(positive["query"]),
                            "positive": str(positive["product_text"]),
                            "negative": str(negative["product_text"]),
                            "positive_product_id": positive_id,
                            "negative_product_id": negative_id,
                            "positive_label": str(positive["esci_label"]),
                            "negative_label": str(negative["esci_label"]),
                            "positive_model_score": float(positive["score"]),
                            "negative_model_score": float(negative["score"]),
                            "score_margin": float(positive["score"] - negative["score"]),
                            "negative_rank": selected_count,
                        }
                    )
                    if selected_count >= negatives_per_positive:
                        break
                if selected_count >= negatives_per_positive:
                    break

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine hardest valid I/C candidates while excluding text-equivalent pairs."
    )
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

    # Recheck before saving, using exactly the same normalization as mining.
    positive_keys = pairs["positive"].map(normalize_model_text)
    negative_keys = pairs["negative"].map(normalize_model_text)
    same_text_count = int((positive_keys == negative_keys).sum())
    if same_text_count:
        raise RuntimeError(
            f"Internal mining error: {same_text_count} normalized-identical triples remain"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(output, index=False)
    write_json(
        {
            "rows": int(len(pairs)),
            "queries": int(pairs["query_id"].nunique()),
            "positive_labels": pairs["positive_label"].value_counts().to_dict(),
            "negative_labels": pairs["negative_label"].value_counts().to_dict(),
            "same_normalized_text_rows": same_text_count,
            "negative_score": {
                "mean": float(pairs["negative_model_score"].mean()),
                "p95": float(pairs["negative_model_score"].quantile(0.95)),
            },
            "score_margin": {
                "mean": float(pairs["score_margin"].mean()),
                "negative_fraction": float((pairs["score_margin"] < 0).mean()),
            },
        },
        output.with_suffix(".summary.json"),
    )
    print(f"Saved {len(pairs):,} hard-negative triples to {output}")
    print(f"Normalized-identical pairs: {same_text_count}")


if __name__ == "__main__":
    main()
