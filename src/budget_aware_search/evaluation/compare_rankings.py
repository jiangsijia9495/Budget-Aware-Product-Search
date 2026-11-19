from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from budget_aware_search.utils import write_json


def paired_bootstrap_delta(
    delta: np.ndarray,
    *,
    samples: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    values = np.asarray(delta, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("delta must be a non-empty one-dimensional array")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk = 200
    n = len(values)
    for start in range(0, samples, chunk):
        end = min(samples, start + chunk)
        indices = rng.integers(0, n, size=(end - start, n))
        means[start:end] = values[indices].mean(axis=1)
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    probability_positive = float(np.mean(means > 0.0))
    return low, high, probability_positive


def compare_per_query(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    metrics: tuple[str, ...],
    samples: int,
    seed: int,
) -> dict[str, object]:
    if baseline["query_id"].duplicated().any() or candidate["query_id"].duplicated().any():
        raise ValueError("Each per-query file must contain one row per query_id")
    merged = baseline[["query_id", *metrics]].merge(
        candidate[["query_id", *metrics]],
        on="query_id",
        how="inner",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    if len(merged) != len(baseline) or len(merged) != len(candidate):
        raise ValueError(
            "Baseline and candidate query sets differ: "
            f"baseline={len(baseline)}, candidate={len(candidate)}, overlap={len(merged)}"
        )

    output: dict[str, object] = {"queries": int(len(merged)), "metrics": {}}
    for index, metric in enumerate(metrics):
        base = merged[f"{metric}_baseline"].to_numpy(dtype=np.float64)
        cand = merged[f"{metric}_candidate"].to_numpy(dtype=np.float64)
        delta = cand - base
        low, high, p_positive = paired_bootstrap_delta(
            delta,
            samples=samples,
            seed=seed + index,
        )
        output["metrics"][metric] = {
            "baseline_mean": float(base.mean()),
            "candidate_mean": float(cand.mean()),
            "mean_delta": float(delta.mean()),
            "relative_delta_percent": (
                float(100.0 * delta.mean() / base.mean()) if base.mean() != 0 else None
            ),
            "paired_ci95_low": low,
            "paired_ci95_high": high,
            "bootstrap_probability_delta_gt_zero": p_positive,
            "query_win_rate": float(np.mean(delta > 0)),
            "query_tie_rate": float(np.mean(delta == 0)),
            "query_loss_rate": float(np.mean(delta < 0)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired query-level comparison of ranking runs.")
    parser.add_argument("--baseline", required=True, help="Baseline *.per_query.parquet")
    parser.add_argument("--candidate", required=True, help="Candidate *.per_query.parquet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", default="ndcg@10,mrr_exact,exact_recall@5,pairwise_accuracy")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics = tuple(item.strip() for item in args.metrics.split(",") if item.strip())
    baseline = pd.read_parquet(args.baseline)
    candidate = pd.read_parquet(args.candidate)
    missing_base = set(metrics).difference(baseline.columns)
    missing_candidate = set(metrics).difference(candidate.columns)
    if missing_base or missing_candidate:
        raise ValueError(
            f"Missing metrics; baseline={sorted(missing_base)}, "
            f"candidate={sorted(missing_candidate)}"
        )

    result = compare_per_query(
        baseline,
        candidate,
        metrics=metrics,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_json(result, args.output)
    print(f"Compared {result['queries']:,} paired queries")
    for metric, record in result["metrics"].items():
        print(
            f"{metric:>20}: {record['baseline_mean']:.6f} -> "
            f"{record['candidate_mean']:.6f}; delta={record['mean_delta']:+.6f}; "
            f"95% CI [{record['paired_ci95_low']:+.6f}, "
            f"{record['paired_ci95_high']:+.6f}]"
        )
    print(f"Saved comparison to {args.output}")


if __name__ == "__main__":
    main()
