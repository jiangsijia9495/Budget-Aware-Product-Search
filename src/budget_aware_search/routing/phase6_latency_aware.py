"""Phase 6: cost-aware routing with measured RTX-stage latency replay.

Quality is evaluated on all available validation queries using frozen phase-4
per-query metrics. Latency is replayed over the observed, independent benchmark
sample (including repeats). Latency is not an online-service measurement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from budget_aware_search.routing.adaptive_reranking import (
    calibrated_threshold,
    pareto_non_dominated,
    priority,
    query_features,
    routing_mask,
)

POLICIES = ("entropy", "margin", "top1", "entropy_per_cost", "margin_per_cost", "random")
METRICS = ("ndcg@10", "mrr_exact", "exact_recall@5", "pairwise_accuracy")


def _validate_features(val: pd.DataFrame) -> None:
    needed = {"query_id", "candidate_count", "entropy", "margin", "top1", "random"}
    needed |= {f"{metric}_{model}" for metric in METRICS for model in ("bi", "ce")}
    missing = needed.difference(val.columns)
    if missing:
        raise ValueError(f"Phase 5 features missing columns: {sorted(missing)}")
    if val.query_id.duplicated().any():
        raise ValueError("Duplicate validation query IDs")
    if len(val) == 0:
        raise ValueError("Empty validation feature table")
    vals = val[list(needed.difference({"query_id"}))].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError("Validation features contain NaN/inf")


def _validate_latency(latency: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    needed = {"query_id", "candidate_count", "bi_ms", "ce_ms"}
    missing = needed.difference(latency.columns)
    if missing:
        raise ValueError(f"Measured latency missing columns: {sorted(missing)}")
    if latency.empty:
        raise ValueError("Measured latency is empty")
    if not np.isfinite(latency[["candidate_count", "bi_ms", "ce_ms"]].to_numpy(dtype=float)).all():
        raise ValueError("Latency columns contain NaN/inf")
    if (latency[["bi_ms", "ce_ms"]] <= 0).any().any():
        raise ValueError("Latency must be > 0 ms")
    joined = latency.merge(
        val[["query_id", "candidate_count"]],
        on="query_id", how="inner", suffixes=("_timed", "_val"),
        validate="many_to_one",
    )
    if len(joined) != len(latency):
        raise ValueError("Some timed query IDs are not in validation")
    if not (joined["candidate_count_timed"].to_numpy() == joined["candidate_count_val"].to_numpy()).all():
        raise ValueError("Candidate counts changed between quality and latency benchmark")
    return joined.drop(columns=["candidate_count_val"]).rename(columns={"candidate_count_timed": "candidate_count"})


class CostModel:
    """Label-free CE latency prediction based on 5-candidate bins.

    Predicts *median* inference stage cost for routing priority only. Final
    latency reported by the sweep uses real benchmark observations, not these
    predictions. Measured query/repeat data must never supply ranking labels.
    """

    def __init__(self, measurements: pd.DataFrame, bin_width: int = 5):
        if bin_width <= 0:
            raise ValueError("bin_width must be positive")
        self.bin_width = bin_width
        per_query = measurements.groupby("query_id", sort=False).agg(
            candidate_count=("candidate_count", "first"),
            ce_ms=("ce_ms", "median"),
        )
        counts = per_query["candidate_count"].to_numpy(dtype=float)
        costs = per_query["ce_ms"].to_numpy(dtype=float)
        if not len(counts):
            raise ValueError("No latency data for cost model")
        bucket_id = np.floor(np.maximum(counts - 1, 0) / bin_width).astype(int)
        bucket_data = pd.DataFrame({"bucket": bucket_id, "cost": costs})
        model = bucket_data.groupby("bucket", sort=True)["cost"].median()
        self.centers = (model.index.to_numpy(dtype=float) + 0.5) * bin_width
        # Avoid fitting a raw noisy non-monotonic curve from very few timings.
        self.medians = np.maximum.accumulate(model.to_numpy(dtype=float))

    def predict(self, candidate_counts: np.ndarray) -> np.ndarray:
        counts = np.asarray(candidate_counts, dtype=float)
        if not np.isfinite(counts).all() or (counts <= 0).any():
            raise ValueError("Invalid candidate_count")
        return np.maximum(
            0.001,
            np.interp(counts, self.centers, self.medians,
                      left=float(self.medians[0]), right=float(self.medians[-1])),
        )


def _cost_model_diagnostics(latency: pd.DataFrame, seed: int) -> dict:
    """Hold out 20% timed queries for a no-label cost calibration check."""
    qs = latency["query_id"].drop_duplicates().to_numpy()
    if len(qs) < 20:
        return {"validation_query_count": 0, "ce_mae_ms": None, "ce_median_ape_pct": None}
    rng = np.random.default_rng(seed)
    qs = qs[rng.permutation(len(qs))]
    cut = max(1, int(round(len(qs) * .2)))
    test = latency.loc[latency.query_id.isin(set(qs[:cut].tolist()))]
    train = latency.loc[~latency.query_id.isin(set(qs[:cut].tolist()))]
    cost = CostModel(train)
    actual = test.groupby("query_id").agg(candidate_count=("candidate_count", "first"), ce_ms=("ce_ms", "median"))
    predicted = cost.predict(actual.candidate_count.to_numpy())
    errors = np.abs(predicted - actual.ce_ms.to_numpy(dtype=float))
    return {
        "validation_query_count": int(len(actual)),
        "ce_mae_ms": float(errors.mean()),
        "ce_median_ape_pct": float(100 * np.median(errors / np.maximum(actual.ce_ms.to_numpy(dtype=float), 1e-6))),
    }


def _policy_score(features: pd.DataFrame, policy: str, cost: CostModel) -> np.ndarray:
    if policy in ("entropy", "margin", "top1", "random"):
        return priority(features, policy)
    ce_ms = cost.predict(features.candidate_count.to_numpy(dtype=float))
    if policy == "entropy_per_cost":
        return features.entropy.to_numpy(dtype=float) / ce_ms
    if policy == "margin_per_cost":
        return 1.0 / ((np.maximum(features.margin.to_numpy(dtype=float), 0) + 0.01) * ce_ms)
    raise ValueError(f"Unsupported routing policy: {policy}")


def _cluster_p95_interval(timing: pd.DataFrame, routed_ids: set, *, seed: int, samples: int) -> tuple[float | None, float | None]:
    if samples <= 0:
        return None, None
    groups = [g for _, g in timing.groupby("query_id", sort=False)]
    if not groups:
        return None, None
    all_costs = [
        (g.bi_ms.to_numpy(dtype=float) + (g.query_id.iloc[0] in routed_ids) * g.ce_ms.to_numpy(dtype=float))
        for g in groups
    ]
    rng = np.random.default_rng(seed)
    n = len(groups)
    p95s = np.empty(samples, dtype=float)
    for j in range(samples):
        indices = rng.integers(0, n, size=n)
        p95s[j] = np.quantile(np.concatenate([all_costs[i] for i in indices]), 0.95)
    return float(np.quantile(p95s, .025)), float(np.quantile(p95s, .975))


def evaluate_routing(
    validation_features: pd.DataFrame,
    train_scores: pd.DataFrame,
    latency_samples: pd.DataFrame,
    *,
    seed: int = 42,
    temperature: float = .1,
    rerank_rates: tuple[float, ...] = tuple(np.linspace(0, 1, 21).tolist()),
    quality_bootstrap_samples: int = 300,
    latency_bootstrap_samples: int = 150,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    _validate_features(validation_features)
    latency = _validate_latency(latency_samples, validation_features)
    if "score" in train_scores:
        score_col = "score"
    elif "dense_score" in train_scores:
        score_col = "dense_score"
    else:
        raise ValueError("Train scores require 'score' or 'dense_score'")
    calibration = query_features(train_scores, score_col=score_col, temperature=temperature, seed=seed)
    cost_model = CostModel(latency)
    cost_diagnostics = _cost_model_diagnostics(latency, seed)
    rng = np.random.default_rng(seed)
    val = validation_features.reset_index(drop=True)
    n = len(val)
    sampled_idx = rng.integers(0, n, size=(quality_bootstrap_samples, n)) if quality_bootstrap_samples else None
    base = float(val["ndcg@10_bi"].mean())
    full = float(val["ndcg@10_ce"].mean())
    gain = full - base
    full_delta = val["ndcg@10_ce"].to_numpy() - val["ndcg@10_bi"].to_numpy()

    rows = []
    for policy in POLICIES:
        cal_priority = _policy_score(calibration, policy, cost_model)
        val_priority = _policy_score(val, policy, cost_model)
        for target in rerank_rates:
            threshold = calibrated_threshold(cal_priority, float(target))
            if target == 0:
                mask = np.zeros(n, dtype=bool)
            elif target == 1:
                mask = np.ones(n, dtype=bool)
            else:
                mask = val_priority > threshold
            routed_ids = set(val.loc[mask, "query_id"])
            repeated_mask = latency.query_id.isin(routed_ids).to_numpy(dtype=bool)
            observed_ms = latency.bi_ms.to_numpy(dtype=float) + repeated_mask * latency.ce_ms.to_numpy(dtype=float)
            row = {
                "policy": policy,
                "target_rerank_rate": float(target),
                "threshold_train_calibrated": float(threshold),
                "actual_rerank_rate": float(mask.mean()),
                "benchmark_rerank_rate": float(repeated_mask.mean()),
                "routed_queries": int(mask.sum()),
                "reranked_candidate_pairs": int(val.loc[mask, "candidate_count"].sum()),
                "mean_stage_latency_ms": float(np.mean(observed_ms)),
                "p50_stage_latency_ms": float(np.quantile(observed_ms, .5)),
                "p95_stage_latency_ms": float(np.quantile(observed_ms, .95)),
                "p99_stage_latency_ms": float(np.quantile(observed_ms, .99)),
            }
            for metric in METRICS:
                outcome = np.where(mask, val[f"{metric}_ce"].to_numpy(dtype=float), val[f"{metric}_bi"].to_numpy(dtype=float))
                row[metric] = float(outcome.mean())
            row["delta_ndcg_vs_bi"] = float(row["ndcg@10"] - base)
            row["gain_retained_pct"] = float(100 * row["delta_ndcg_vs_bi"] / gain) if gain > 0 else None
            if sampled_idx is not None:
                gain_values = np.where(mask, full_delta, 0.0)
                sampled_means = gain_values[sampled_idx].mean(axis=1)
                row["gain_ci95_low"] = float(np.quantile(sampled_means, .025))
                row["gain_ci95_high"] = float(np.quantile(sampled_means, .975))
            low, high = _cluster_p95_interval(latency, routed_ids, seed=seed, samples=latency_bootstrap_samples)
            row["p95_ci95_low_ms"] = low
            row["p95_ci95_high_ms"] = high
            rows.append(row)

    sweep = pd.DataFrame(rows)
    # Counterfactual oracle uses validation labels for diagnostics ONLY; it must
    # never be treated as a deployable router or compared to heuristics as fair.
    order = np.argsort(-full_delta, kind="mergesort")
    oracle = []
    for rate in rerank_rates:
        capacity = min(n, max(0, int(round(rate * n))))
        eligible = order[:capacity]
        oracle_gain = float(np.maximum(full_delta[eligible], 0.0).sum() / n)
        oracle.append({
            "rerank_budget": float(rate),
            "oracle_ndcg@10": base + oracle_gain,
            "oracle_gain_retained_pct": float(100 * oracle_gain / gain) if gain > 0 else None,
            "note": "label-dependent upper bound; not deployable",
        })
    diagnostic = {
        "quality_validation_queries": n,
        "latency_benchmark_queries": int(latency.query_id.nunique()),
        "latency_benchmark_measurements": int(len(latency)),
        "bi_ndcg@10": base,
        "ce_ndcg@10": full,
        "ce_gain": gain,
        "measured_latency_median_bi_ms": float(latency.bi_ms.median()),
        "measured_latency_median_ce_ms": float(latency.ce_ms.median()),
        "cost_model_diagnostics": cost_diagnostics,
        "quality_scope": "all validation queries",
        "latency_scope": "replay on measured latency subset only, not live end-to-end serving",
    }
    return sweep, pd.DataFrame(oracle), diagnostic


def choose_point(sweep: pd.DataFrame, *, max_rate: float, p95_budget_ms: float | None, target_gain_pct: float) -> dict:
    eligible = sweep.loc[sweep.actual_rerank_rate <= max_rate + 1e-12].copy()
    if p95_budget_ms is not None:
        eligible = eligible.loc[eligible.p95_stage_latency_ms <= p95_budget_ms]
    if eligible.empty:
        return {"feasible": False, "reason": "No policy meets specified rerank-rate and P95 budget"}
    eligible = eligible.sort_values(["ndcg@10", "p95_stage_latency_ms"], ascending=[False, True])
    best = eligible.iloc[0]
    return {
        "feasible": True,
        "policy": str(best.policy),
        "target_rerank_rate": float(best.target_rerank_rate),
        "actual_rerank_rate": float(best.actual_rerank_rate),
        "benchmark_rerank_rate": float(best.benchmark_rerank_rate),
        "threshold_train_calibrated": (
            float(best.threshold_train_calibrated)
            if np.isfinite(float(best.threshold_train_calibrated)) else None
        ),
        "threshold_mode": (
            "never_route" if float(best.target_rerank_rate) == 0.0
            else "always_route" if float(best.target_rerank_rate) == 1.0
            else "priority_greater_than_threshold"
        ),
        "ndcg@10": float(best["ndcg@10"]),
        "gain_retained_pct": float(best.gain_retained_pct),
        "target_gain_pct": target_gain_pct,
        "target_met": bool(best.gain_retained_pct >= target_gain_pct),
        "p95_stage_latency_ms": float(best.p95_stage_latency_ms),
        "p95_ci95_low_ms": float(best.p95_ci95_low_ms) if pd.notna(best.p95_ci95_low_ms) else None,
        "p95_ci95_high_ms": float(best.p95_ci95_high_ms) if pd.notna(best.p95_ci95_high_ms) else None,
        "selection_scope": "exploratory validation selection, test split not used",
    }


def _plot_results(sweep: pd.DataFrame, oracle: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for policy, data in sweep.groupby("policy"):
        data = data.sort_values("actual_rerank_rate")
        ax.plot(data.actual_rerank_rate, data["ndcg@10"], marker=".", label=policy)
    ax.plot(oracle.rerank_budget, oracle["oracle_ndcg@10"], linestyle="--", label="oracle (non-deployable)")
    ax.set(xlabel="Actual Cross-Encoder query rate", ylabel="Validation nDCG@10", title="Adaptive routing vs oracle upper bound")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "quality_vs_rate_with_oracle.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for policy, data in sweep.groupby("policy"):
        data = data.sort_values("p95_stage_latency_ms")
        ax.plot(data.p95_stage_latency_ms, data["ndcg@10"], marker=".", label=policy)
    ax.set(xlabel="Measured-subset P95 stage latency (ms, replay)", ylabel="Validation nDCG@10", title="Measured-stage latency vs quality")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "quality_vs_measured_p95.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase5-features", default="outputs/adaptive_reranking/query_features_and_metrics.parquet")
    parser.add_argument("--train-scored", default="outputs/biencoder_v1_random_train_scored.parquet")
    parser.add_argument("--latency", default="outputs/adaptive_reranking/latency_samples.parquet")
    parser.add_argument("--output-dir", default="outputs/phase6_latency_aware")
    parser.add_argument("--max-rerank-rate", type=float, default=.50)
    parser.add_argument("--target-gain-retention", type=float, default=95.0)
    parser.add_argument("--p95-budget-ms", type=float, default=None)
    parser.add_argument("--quality-bootstrap-samples", type=int, default=300)
    parser.add_argument("--latency-bootstrap-samples", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.max_rerank_rate <= 1:
        raise ValueError("max-rerank-rate must be within [0,1]")
    if args.p95_budget_ms is not None and args.p95_budget_ms <= 0:
        raise ValueError("p95-budget-ms must be positive")
    if not 0 <= args.target_gain_retention <= 100:
        raise ValueError("target-gain-retention must be within [0,100]")
    paths = (args.phase5_features, args.train_scored, args.latency)
    for path in paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing {path}. Re-run Phase 5 GPU benchmark or transfer that file to the project.")
    val = pd.read_parquet(args.phase5_features)
    train = pd.read_parquet(args.train_scored)
    latency = pd.read_parquet(args.latency)
    sweep, oracle, diagnostics = evaluate_routing(
        val, train, latency, seed=args.seed, temperature=args.temperature,
        quality_bootstrap_samples=args.quality_bootstrap_samples,
        latency_bootstrap_samples=args.latency_bootstrap_samples,
    )
    best = choose_point(sweep, max_rate=args.max_rerank_rate,
                        p95_budget_ms=args.p95_budget_ms,
                        target_gain_pct=args.target_gain_retention)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(output / "policy_sweep.csv", index=False)
    oracle.to_csv(output / "oracle_upper_bound.csv", index=False)
    pareto_non_dominated(sweep, cost_col="p95_stage_latency_ms", value_col="ndcg@10").to_csv(output / "pareto_frontier.csv", index=False)
    _plot_results(sweep, oracle, output)
    at_budget = oracle.iloc[(oracle.rerank_budget - args.max_rerank_rate).abs().argmin()]
    summary = {
        **diagnostics,
        "max_rerank_rate": args.max_rerank_rate,
        "target_gain_retention_pct": args.target_gain_retention,
        "target_ndcg@10": float(diagnostics["bi_ndcg@10"] + args.target_gain_retention / 100 * diagnostics["ce_gain"]),
        "p95_budget_ms": args.p95_budget_ms,
        "best_policy_under_constraints": best,
        "oracle_upper_bound_at_nearest_budget": at_budget.to_dict(),
        "warning": "Policy picked on validation; evaluate frozen policy on untouched test only once. Stage P95 is a measured-subset replay, not deployed service latency.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Saved Phase 6 outputs to {output}")


if __name__ == "__main__":
    main()
