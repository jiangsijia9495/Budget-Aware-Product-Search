from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from budget_aware_search.labels import ESCI_GAINS, validate_esci_labels
from budget_aware_search.utils import write_json


def dcg_at_k(gains: np.ndarray, k: int) -> float:
    values = np.asarray(gains, dtype=np.float64)[:k]
    if values.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, values.size + 2, dtype=np.float64))
    return float(np.sum(values / discounts))


def ndcg_at_k(gains_in_rank_order: np.ndarray, all_gains: np.ndarray, k: int) -> float:
    denominator = dcg_at_k(np.sort(np.asarray(all_gains))[::-1], k)
    if denominator <= 0.0:
        return 0.0
    return dcg_at_k(gains_in_rank_order, k) / denominator


def _pairwise_accuracy(gains: np.ndarray, scores: np.ndarray) -> float:
    gain_diff = gains[:, None] - gains[None, :]
    score_diff = scores[:, None] - scores[None, :]
    upper = np.triu(np.ones_like(gain_diff, dtype=bool), k=1)
    comparable = upper & (gain_diff != 0)
    if not np.any(comparable):
        return 0.0
    correct = np.sign(gain_diff[comparable]) == np.sign(score_diff[comparable])
    ties = score_diff[comparable] == 0
    return float(np.mean(np.where(ties, 0.5, correct.astype(float))))


def evaluate_query_group(
    group: pd.DataFrame,
    score_col: str,
    k_values: Iterable[int],
) -> dict[str, float | int]:
    ranked = group.sort_values(
        [score_col, "product_id"], ascending=[False, True], kind="mergesort"
    )
    labels = ranked["esci_label"].to_numpy()
    scores = ranked[score_col].to_numpy(dtype=np.float64)
    gains = np.asarray([ESCI_GAINS[label] for label in labels], dtype=np.float64)
    ideal_gains = np.asarray(
        [ESCI_GAINS[label] for label in group["esci_label"].to_numpy()], dtype=np.float64
    )

    result: dict[str, float | int] = {
        "query_id": group["query_id"].iloc[0],
        "candidate_count": int(len(group)),
        "pairwise_accuracy": _pairwise_accuracy(gains, scores),
    }
    exact_positions = np.flatnonzero(labels == "E")
    result["mrr_exact"] = float(1.0 / (exact_positions[0] + 1)) if exact_positions.size else 0.0

    total_exact = int(np.sum(labels == "E"))
    for k in k_values:
        result[f"ndcg@{k}"] = ndcg_at_k(gains, ideal_gains, k)
        top_labels = labels[:k]
        exact_at_k = int(np.sum(top_labels == "E"))
        result[f"exact_recall@{k}"] = (
            float(exact_at_k / total_exact) if total_exact > 0 else 0.0
        )
        result[f"exact_success@{k}"] = float(exact_at_k > 0)
    return result


def bootstrap_mean_ci(
    values: np.ndarray,
    samples: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if samples <= 0 or len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    n = len(values)
    chunk_size = 200
    for start in range(0, samples, chunk_size):
        end = min(samples, start + chunk_size)
        indices = rng.integers(0, n, size=(end - start, n))
        means[start:end] = values[indices].mean(axis=1)
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def evaluate_predictions(
    predictions: pd.DataFrame,
    score_col: str = "score",
    k_values: tuple[int, ...] = (5, 10, 20),
    bootstrap_samples: int = 0,
    seed: int = 42,
) -> tuple[dict[str, object], pd.DataFrame]:
    required = {"query_id", "product_id", "esci_label", score_col}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    if predictions[score_col].isna().any():
        raise ValueError(f"Score column {score_col!r} contains NaN values")
    if predictions.duplicated(["query_id", "product_id"]).any():
        raise ValueError("Predictions contain duplicate query_id/product_id pairs")
    validate_esci_labels(set(predictions["esci_label"].unique()))

    rows = [
        evaluate_query_group(group, score_col, k_values)
        for _, group in predictions.groupby("query_id", sort=False)
    ]
    per_query = pd.DataFrame(rows)
    metric_columns = [column for column in per_query.columns if column not in {"query_id"}]

    summary_metrics: dict[str, object] = {}
    for index, column in enumerate(metric_columns):
        values = per_query[column].to_numpy(dtype=np.float64)
        mean = float(values.mean())
        record: dict[str, float] = {"mean": mean}
        if bootstrap_samples > 0 and column != "candidate_count":
            low, high = bootstrap_mean_ci(values, bootstrap_samples, seed + index)
            record.update({"ci95_low": low, "ci95_high": high})
        summary_metrics[column] = record

    summary: dict[str, object] = {
        "queries": int(per_query["query_id"].nunique()),
        "rows": int(len(predictions)),
        "score_column": score_col,
        "gain_mapping": ESCI_GAINS,
        "metrics": summary_metrics,
    }
    return summary, per_query


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate candidate-pool ranking predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--score-col", default="score")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--per-query-output", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    path = Path(args.predictions)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    summary, per_query = evaluate_predictions(
        frame,
        score_col=args.score_col,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output_json = args.output_json or str(path.with_suffix(".metrics.json"))
    per_query_output = args.per_query_output or str(path.with_suffix(".per_query.parquet"))
    write_json(summary, output_json)
    per_query.to_parquet(per_query_output, index=False)

    print(f"Evaluated {summary['queries']:,} queries")
    for name, record in summary["metrics"].items():
        if name.startswith("ndcg") or name in {"mrr_exact", "pairwise_accuracy"}:
            print(f"  {name:>20}: {record['mean']:.6f}")
    print(f"Saved summary to {output_json}")
    print(f"Saved per-query metrics to {per_query_output}")


if __name__ == "__main__":
    main()
