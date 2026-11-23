from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd

LABEL_ORDER: tuple[str, ...] = ("E", "S", "C", "I")
LABEL_TO_ID: dict[str, int] = {label: index for index, label in enumerate(LABEL_ORDER)}
ESCI_GAINS: dict[str, float] = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
GAIN_VECTOR = np.asarray([ESCI_GAINS[label] for label in LABEL_ORDER], dtype=np.float64)


def expected_gain_scores(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(LABEL_ORDER):
        raise ValueError(
            f"Expected probabilities with shape (n, {len(LABEL_ORDER)}), "
            f"got {probabilities.shape}"
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities contain NaN or infinite values")
    return probabilities @ GAIN_VECTOR


def _dcg(gains: np.ndarray, k: int | None = None) -> float:
    if k is not None:
        gains = gains[:k]
    if gains.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, gains.size + 2, dtype=np.float64))
    return float(np.sum(gains / discounts))


def _ndcg(gains: np.ndarray, k: int) -> float:
    ideal = np.sort(gains)[::-1]
    denominator = _dcg(ideal, k=k)
    if denominator <= 0.0:
        return 0.0
    return _dcg(gains, k=k) / denominator


def _mrr_exact(labels: np.ndarray) -> float:
    exact_positions = np.flatnonzero(labels == "E")
    if exact_positions.size == 0:
        return 0.0
    return 1.0 / float(exact_positions[0] + 1)


def _exact_recall(labels: np.ndarray, k: int) -> float:
    total_exact = int(np.sum(labels == "E"))
    if total_exact == 0:
        return 0.0
    return float(np.sum(labels[:k] == "E") / total_exact)


def _pairwise_accuracy(scores: np.ndarray, gains: np.ndarray) -> float:
    correct = 0.0
    total = 0
    for left, right in combinations(range(len(scores)), 2):
        true_difference = gains[left] - gains[right]
        if true_difference == 0.0:
            continue
        predicted_difference = scores[left] - scores[right]
        product = true_difference * predicted_difference
        if product > 0.0:
            correct += 1.0
        elif product == 0.0:
            correct += 0.5
        total += 1
    return float(correct / total) if total else 0.0


def evaluate_scored_candidates(
    frame: pd.DataFrame,
    *,
    score_column: str,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    required = {"query_id", "product_id", "esci_label", score_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Scored frame is missing columns: {sorted(missing)}")

    unknown_labels = set(frame["esci_label"].dropna().astype(str)).difference(ESCI_GAINS)
    if unknown_labels:
        raise ValueError(f"Unknown ESCI labels: {sorted(unknown_labels)}")

    rows: list[dict[str, float | int | str]] = []
    for query_id, group in frame.groupby("query_id", sort=False):
        ranked = group.sort_values(
            [score_column, "product_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        labels = ranked["esci_label"].astype(str).to_numpy()
        scores = ranked[score_column].to_numpy(dtype=np.float64)
        gains = np.asarray([ESCI_GAINS[label] for label in labels], dtype=np.float64)
        rows.append(
            {
                "query_id": query_id,
                "ndcg@5": _ndcg(gains, 5),
                "ndcg@10": _ndcg(gains, 10),
                "ndcg@20": _ndcg(gains, 20),
                "mrr_exact": _mrr_exact(labels),
                "exact_recall@5": _exact_recall(labels, 5),
                "exact_recall@10": _exact_recall(labels, 10),
                "exact_recall@20": _exact_recall(labels, 20),
                "pairwise_accuracy": _pairwise_accuracy(scores, gains),
                "candidate_count": int(len(group)),
            }
        )

    per_query = pd.DataFrame(rows)
    metric_columns = [
        "ndcg@5",
        "ndcg@10",
        "ndcg@20",
        "mrr_exact",
        "exact_recall@5",
        "exact_recall@10",
        "exact_recall@20",
        "pairwise_accuracy",
    ]
    summary: dict[str, float | int] = {"queries": int(len(per_query))}
    for column in metric_columns:
        summary[column] = float(per_query[column].mean()) if len(per_query) else 0.0
    return summary, per_query


def bootstrap_metric_intervals(
    per_query: pd.DataFrame,
    *,
    columns: Iterable[str],
    samples: int,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    if samples <= 0:
        return {}
    rng = np.random.default_rng(seed)
    n_queries = len(per_query)
    if n_queries == 0:
        return {}

    output: dict[str, dict[str, float]] = {}
    for column in columns:
        values = per_query[column].to_numpy(dtype=np.float64)
        bootstrap_means = np.empty(samples, dtype=np.float64)
        chunk_size = 250
        for start in range(0, samples, chunk_size):
            end = min(samples, start + chunk_size)
            indices = rng.integers(0, n_queries, size=(end - start, n_queries))
            bootstrap_means[start:end] = values[indices].mean(axis=1)
        output[column] = {
            "mean": float(values.mean()),
            "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
            "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
        }
    return output
