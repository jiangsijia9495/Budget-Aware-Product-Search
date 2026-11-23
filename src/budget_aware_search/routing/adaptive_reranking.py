"""Phase 5: leakage-safe adaptive reranking quality/latency Pareto analysis.

Quality is taken directly from the already evaluated, fixed candidate-pool
per-query metrics, so the all-on/all-off endpoints exactly match Phase 4.
The router uses ONLY bi-encoder scores and query metadata; never gold labels.
Routing thresholds are calibrated from separate TRAIN query scores.

Optional measured latency samples are replayed per query:
  latency(q) = measured_bi(q) + route(q) * measured_ce(q)
This is a measured-stage latency *replay*, not an instrumented live server P95.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

METRICS = ("ndcg@10", "mrr_exact", "exact_recall@5", "pairwise_accuracy")
POLICIES = ("margin", "entropy", "top1", "random")


def _choose_score_column(df: pd.DataFrame, explicit: str | None, choices: tuple[str, ...]) -> str:
    if explicit:
        if explicit not in df:
            raise ValueError(f"Missing requested score column {explicit!r}. Available: {df.columns.tolist()}")
        return explicit
    matches = [name for name in choices if name in df]
    if not matches:
        raise ValueError(f"Cannot locate score column. Available: {df.columns.tolist()}. Pass --bi-score-column or --ce-score-column.")
    return matches[0]


def _seeded_random(key: object, seed: int) -> float:
    value = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(value[:8], "big") / float(2**64)


def query_features(scored: pd.DataFrame, score_col: str = "score", temperature: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """Compute only inference-available query difficulty proxies from bi scores."""
    needed = {"query_id", score_col}
    if not needed.issubset(scored.columns):
        raise ValueError(f"Query feature input missing: {sorted(needed - set(scored.columns))}")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    out: list[dict[str, Any]] = []
    for qid, g in scored.groupby("query_id", sort=False):
        s = np.asarray(g[score_col], dtype=np.float64)
        if len(s) == 0 or not np.all(np.isfinite(s)):
            raise ValueError(f"Query {qid!r} has invalid scores")
        ordered = np.sort(s)[::-1]
        # Stable entropy of softmax scores, normalized for different candidate counts.
        logits = (s - np.max(s)) / temperature
        p = np.exp(logits)
        p /= p.sum()
        entropy = -float(np.dot(p, np.log(np.maximum(p, 1e-15))))
        entropy /= float(np.log(len(s))) if len(s) > 1 else 1.0
        top1 = float(ordered[0])
        top_margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else 1e6
        row: dict[str, Any] = {
            "query_id": qid,
            "candidate_count": len(s),
            "margin": top_margin,
            "entropy": entropy,
            "top1": top1,
            "random": _seeded_random(qid, seed),
        }
        if "query" in g:
            row["query_char_length"] = len(str(g["query"].iloc[0]))
        out.append(row)
    return pd.DataFrame(out)


def priority(features: pd.DataFrame, policy: str) -> np.ndarray:
    if policy == "margin":
        return -features["margin"].to_numpy(dtype=float)
    if policy == "entropy":
        return features["entropy"].to_numpy(dtype=float)
    if policy == "top1":
        return -features["top1"].to_numpy(dtype=float)
    if policy == "random":
        return features["random"].to_numpy(dtype=float)
    raise ValueError(f"Unknown policy: {policy}")


def calibrated_threshold(calibration_priorities: np.ndarray, target_rate: float) -> float:
    if not 0 <= target_rate <= 1:
        raise ValueError("target_rate must be between 0 and 1")
    if target_rate == 0:
        return float("inf")
    if target_rate == 1:
        return -float("inf")
    vals = np.asarray(calibration_priorities, dtype=float)
    if len(vals) == 0:
        raise ValueError("Empty calibration scores")
    # Use strict > for threshold to avoid rerouting tied confidence values en masse.
    return float(np.quantile(vals, 1 - target_rate))


def routing_mask(features: pd.DataFrame, policy: str, threshold: float, target_rate: float) -> np.ndarray:
    if target_rate == 0:
        return np.zeros(len(features), dtype=bool)
    if target_rate == 1:
        return np.ones(len(features), dtype=bool)
    return priority(features, policy) > threshold


def validate_candidate_alignment(bi: pd.DataFrame, ce: pd.DataFrame) -> None:
    keys = ["query_id", "product_id"]
    for name, frame in (("bi", bi), ("cross", ce)):
        if not set(keys).issubset(frame.columns):
            raise ValueError(f"{name} scores missing {keys}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} contains duplicate query_id/product_id rows")
    merged = bi[keys].merge(ce[keys], on=keys, how="outer", indicator=True, validate="one_to_one")
    if not merged["_merge"].eq("both").all():
        counts = merged["_merge"].value_counts().to_dict()
        raise ValueError(f"Bi-encoder / Cross-Encoder candidate pools differ: {counts}")
    if "esci_label" in bi and "esci_label" in ce:
        label_join = bi[keys + ["esci_label"]].merge(
            ce[keys + ["esci_label"]], on=keys, suffixes=("_bi", "_ce"), validate="one_to_one"
        )
        if not label_join["esci_label_bi"].astype(str).eq(label_join["esci_label_ce"].astype(str)).all():
            raise ValueError("Label mismatch between the bi and CE scored candidate sets")


def load_metric_pairs(bi_per_query: pd.DataFrame, ce_per_query: pd.DataFrame) -> pd.DataFrame:
    for name, frame in (("bi", bi_per_query), ("cross", ce_per_query)):
        cols = {"query_id", *METRICS}
        if not cols.issubset(frame):
            raise ValueError(f"{name} per-query metrics missing: {sorted(cols - set(frame.columns))}")
        if frame["query_id"].duplicated().any():
            raise ValueError(f"{name} per-query metrics contain duplicate query_id")
    merged = bi_per_query[["query_id", *METRICS]].merge(
        ce_per_query[["query_id", *METRICS]],
        how="inner",
        on="query_id",
        suffixes=("_bi", "_ce"),
        validate="one_to_one",
    )
    if len(merged) != len(bi_per_query) or len(merged) != len(ce_per_query):
        raise ValueError("Per-query metrics do not cover exactly the same queries")
    return merged


def pareto_non_dominated(points: pd.DataFrame, cost_col: str = "p95_latency_ms", value_col: str = "ndcg@10") -> pd.DataFrame:
    """Return all globally nondominated points (cost lower, quality higher)."""
    if points.empty:
        return points.copy()
    usable = points.dropna(subset=[cost_col, value_col]).copy()
    costs = usable[cost_col].to_numpy(dtype=float)
    qualities = usable[value_col].to_numpy(dtype=float)
    mask = np.ones(len(usable), dtype=bool)
    for idx in range(len(usable)):
        dominated_by = (costs <= costs[idx] + 1e-10) & (qualities >= qualities[idx] - 1e-10)
        strictly = (costs < costs[idx] - 1e-10) | (qualities > qualities[idx] + 1e-10)
        dominated_by[idx] = False
        if np.any(dominated_by & strictly):
            mask[idx] = False
    return usable.iloc[np.flatnonzero(mask)].sort_values([cost_col, value_col])


def _percentiles_ms(values: np.ndarray) -> dict[str, float]:
    return {
        "mean_latency_ms": float(np.mean(values)),
        "p50_latency_ms": float(np.quantile(values, 0.5)),
        "p95_latency_ms": float(np.quantile(values, 0.95)),
        "p99_latency_ms": float(np.quantile(values, 0.99)),
        "estimated_serial_qps": float(1000 / np.mean(values)),
    }


def build_pareto(
    bi_scored: pd.DataFrame,
    ce_scored: pd.DataFrame,
    bi_per_query: pd.DataFrame,
    ce_per_query: pd.DataFrame,
    calibration_scored: pd.DataFrame,
    *,
    bi_score_col: str = "score",
    calibration_score_col: str = "score",
    temperature: float = 0.1,
    seed: int = 42,
    fractions: tuple[float, ...] = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1),
    latency: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return sweep rows, feature rows, and routing decisions (all per-query)."""
    validate_candidate_alignment(bi_scored, ce_scored)
    features = query_features(bi_scored, bi_score_col, temperature, seed)
    calibration_features = query_features(calibration_scored, calibration_score_col, temperature, seed)
    metrics = load_metric_pairs(bi_per_query, ce_per_query)
    joined = metrics.merge(features, on="query_id", how="inner", validate="one_to_one")
    if len(joined) != len(metrics) or len(joined) != len(features):
        raise ValueError("Scores and metrics have different query coverage")
    n = len(joined)
    if n == 0:
        raise ValueError("No validation queries")

    latency_merged = None
    if latency is not None:
        required = {"query_id", "bi_ms", "ce_ms"}
        if not required.issubset(latency):
            raise ValueError(f"Latency data missing columns {sorted(required - set(latency.columns))}")
        # Preserve every timed repetition instead of averaging away GPU jitter.
        # A single routing decision applies to every repeat for that query.
        lat_data = latency[["query_id", "bi_ms", "ce_ms"]].merge(
            joined[["query_id"]], on="query_id", how="inner", validate="many_to_one"
        )
        if lat_data.empty:
            raise ValueError("No overlapping query IDs between validation and latency benchmark")
        if not np.isfinite(lat_data[["bi_ms", "ce_ms"]].to_numpy(dtype=float)).all():
            raise ValueError("Latency values contain NaN/inf")
        if (lat_data[["bi_ms", "ce_ms"]].to_numpy(dtype=float) < 0).any():
            raise ValueError("Negative latency values are invalid")
    else:
        lat_data = None

    bi_ndcg = float(joined["ndcg@10_bi"].mean())
    ce_ndcg = float(joined["ndcg@10_ce"].mean())
    gain = ce_ndcg - bi_ndcg
    # Paired query bootstrap; the same query IDs are resampled for both models.
    # This is exploratory because we will select the routing policy on validation.
    bootstrap_rng = np.random.default_rng(seed)
    bootstrap_indices = bootstrap_rng.integers(0, n, size=(1000, n))
    sweep: list[dict[str, Any]] = []
    decision_rows: list[pd.DataFrame] = []
    for policy in POLICIES:
        calib_priorities = priority(calibration_features, policy)
        for target_rate in fractions:
            threshold = calibrated_threshold(calib_priorities, target_rate)
            mask = routing_mask(joined, policy, threshold, target_rate)
            result: dict[str, Any] = {
                "policy": policy,
                "target_rerank_rate": float(target_rate),
                "calibrated_threshold": threshold,
                "rerank_rate": float(mask.mean()),
                "routed_queries": int(mask.sum()),
                "total_queries": n,
                "reranked_candidate_pairs": int(joined.loc[mask, "candidate_count"].sum()),
            }
            for metric in METRICS:
                values = np.where(mask, joined[f"{metric}_ce"].to_numpy(), joined[f"{metric}_bi"].to_numpy())
                result[metric] = float(values.mean())
                result[f"delta_{metric}_vs_bi"] = float(values.mean() - joined[f"{metric}_bi"].mean())
            result["gain_retained_pct"] = float(100 * (result["ndcg@10"] - bi_ndcg) / gain) if gain > 1e-12 else None
            delta_per_query = np.where(
                mask,
                joined["ndcg@10_ce"].to_numpy() - joined["ndcg@10_bi"].to_numpy(),
                0.0,
            )
            paired_means = delta_per_query[bootstrap_indices].mean(axis=1)
            result["delta_ndcg@10_ci95_low"] = float(np.quantile(paired_means, 0.025))
            result["delta_ndcg@10_ci95_high"] = float(np.quantile(paired_means, 0.975))

            if lat_data is not None:
                decisions = pd.Series(mask, index=joined["query_id"])
                latency_mask = lat_data["query_id"].map(decisions).to_numpy(dtype=bool)
                per_query_cost = lat_data["bi_ms"].to_numpy() + latency_mask * lat_data["ce_ms"].to_numpy()
                result.update(_percentiles_ms(per_query_cost))
                result["latency_benchmark_queries"] = int(lat_data["query_id"].nunique())
                result["latency_measurements"] = int(len(lat_data))
                result["latency_benchmark_rerank_rate"] = float(latency_mask.mean())

            sweep.append(result)
            decision_rows.append(
                pd.DataFrame({
                    "query_id": joined["query_id"],
                    "policy": policy,
                    "target_rerank_rate": float(target_rate),
                    "rerouted": mask,
                })
            )
    return pd.DataFrame(sweep), joined, pd.concat(decision_rows, ignore_index=True)


def plot_sweep(sweep: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for x, output_name, label in (
        ("rerank_rate", "quality_vs_rerank_rate.png", "Actual Cross-Encoder rerank rate"),
        ("p95_latency_ms", "quality_vs_p95_latency.png", "P95 latency (ms), stage-replay estimate"),
    ):
        if x not in sweep:
            continue
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        for policy, g in sweep.groupby("policy"):
            g = g.sort_values(x)
            ax.plot(g[x], g["ndcg@10"], marker="o", label=policy)
        ax.set_xlabel(label)
        ax.set_ylabel("Validation nDCG@10")
        ax.set_title("Adaptive reranking quality-efficiency tradeoff")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / output_name, dpi=160)
        plt.close(fig)


def _serialize_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bi-scores", default="outputs/biencoder_v1_random_val.parquet")
    parser.add_argument("--ce-scores", default="outputs/cross_encoder_esci_v1_val.parquet")
    parser.add_argument("--bi-per-query", default="outputs/biencoder_v1_random_val.per_query.parquet")
    parser.add_argument("--ce-per-query", default="outputs/cross_encoder_esci_v1_val.per_query.parquet")
    parser.add_argument("--calibration-scores", default="outputs/biencoder_v1_random_train_scored.parquet")
    parser.add_argument("--bi-score-column", default=None)
    parser.add_argument("--calibration-score-column", default=None)
    parser.add_argument("--ce-score-column", default=None)
    parser.add_argument("--latency", default=None)
    parser.add_argument("--output-dir", default="outputs/adaptive_reranking")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--expected-val-queries", type=int, default=3133)
    parser.add_argument("--target-gain-retention", type=float, default=95)
    parser.add_argument("--max-rerank-rate", type=float, default=0.5)
    parser.add_argument("--p95-budget-ms", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bi = pd.read_parquet(args.bi_scores)
    ce = pd.read_parquet(args.ce_scores)
    bi_per_query = pd.read_parquet(args.bi_per_query)
    ce_per_query = pd.read_parquet(args.ce_per_query)
    calib = pd.read_parquet(args.calibration_scores)
    if args.expected_val_queries and bi_per_query["query_id"].nunique() != args.expected_val_queries:
        raise ValueError(f"Validation query count mismatch: expected {args.expected_val_queries}, got {bi_per_query['query_id'].nunique()}")
    bi_col = _choose_score_column(bi, args.bi_score_column, ("score", "dense_score", "cosine_score"))
    calib_col = _choose_score_column(calib, args.calibration_score_column, ("score", "dense_score", "cosine_score"))
    ce_col = _choose_score_column(ce, args.ce_score_column, ("cross_score", "score"))
    if not np.isfinite(ce[ce_col].to_numpy(dtype=float)).all():
        raise ValueError("Cross-Encoder score contains NaN/inf")
    latency = pd.read_parquet(args.latency) if args.latency else None

    sweep, features, decisions = build_pareto(
        bi, ce, bi_per_query, ce_per_query, calib,
        bi_score_col=bi_col,
        calibration_score_col=calib_col,
        temperature=args.temperature,
        seed=args.seed,
        latency=latency,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(output_dir / "pareto_sweep.csv", index=False)
    features.to_parquet(output_dir / "query_features_and_metrics.parquet", index=False)
    decisions.to_parquet(output_dir / "routing_decisions.parquet", index=False)
    cost_col = "p95_latency_ms" if latency is not None else "rerank_rate"
    pareto_non_dominated(sweep, cost_col=cost_col).to_csv(output_dir / "pareto_frontier.csv", index=False)
    plot_sweep(sweep, output_dir)

    baseline = float(features["ndcg@10_bi"].mean())
    oracle_all_ce = float(features["ndcg@10_ce"].mean())
    eligible = sweep.loc[
        (sweep["rerank_rate"] <= args.max_rerank_rate + 1e-9)
        & (sweep["gain_retained_pct"] >= args.target_gain_retention)
    ]
    if args.p95_budget_ms is not None:
        if "p95_latency_ms" not in eligible:
            raise ValueError("--p95-budget-ms requires --latency")
        eligible = eligible.loc[eligible["p95_latency_ms"] <= args.p95_budget_ms]
    recommended: dict[str, Any] | None = None
    if not eligible.empty:
        rank_by = "p95_latency_ms" if latency is not None else "rerank_rate"
        selected = eligible.sort_values([rank_by, "ndcg@10"], ascending=[True, False]).iloc[0]
        recommended = {
            "policy": str(selected["policy"]),
            "threshold": float(selected["calibrated_threshold"]) if np.isfinite(float(selected["calibrated_threshold"])) else None,
            "target_rerank_rate": float(selected["target_rerank_rate"]),
            "actual_rerank_rate": float(selected["rerank_rate"]),
            "ndcg@10": float(selected["ndcg@10"]),
            "gain_retained_pct": float(selected["gain_retained_pct"]),
            "p95_latency_ms": float(selected["p95_latency_ms"]) if latency is not None else None,
        }
    rate_constrained = sweep.loc[sweep["rerank_rate"] <= args.max_rerank_rate + 1e-9]
    if args.p95_budget_ms is not None:
        rate_constrained = rate_constrained.loc[rate_constrained["p95_latency_ms"] <= args.p95_budget_ms]
    best_at_budget = None
    if not rate_constrained.empty:
        best_row = rate_constrained.sort_values(["ndcg@10", "rerank_rate"], ascending=[False, True]).iloc[0]
        best_at_budget = {
            "policy": str(best_row["policy"]),
            "threshold": float(best_row["calibrated_threshold"]) if np.isfinite(float(best_row["calibrated_threshold"])) else None,
            "actual_rerank_rate": float(best_row["rerank_rate"]),
            "ndcg@10": float(best_row["ndcg@10"]),
            "gain_retained_pct": float(best_row["gain_retained_pct"]),
            "p95_latency_ms": float(best_row["p95_latency_ms"]) if latency is not None else None,
        }
    _serialize_json(
        {
            "queries": int(len(features)),
            "bi_ndcg@10": baseline,
            "cross_ndcg@10": oracle_all_ce,
            "full_ce_gain": oracle_all_ce - baseline,
            "routing_features_use_labels": False,
            "threshold_calibration_source": args.calibration_scores,
            "threshold_calibration_uses_labels": False,
            "latency_kind": "measured-stage replay on benchmark subset, NOT direct serving P95" if latency is not None else "not_measured",
            "target_gain_retention_percent": args.target_gain_retention,
            "max_rerank_rate": args.max_rerank_rate,
            "p95_budget_ms": args.p95_budget_ms,
            "recommendation_exploratory_only": True,
            "recommended_policy": recommended,
            "best_quality_within_budget": best_at_budget,
            "note": "Freeze the selected router and all models BEFORE evaluating untouched test queries. No performance guarantee at deployment.",
        }, output_dir / "summary.json",
    )
    print(f"Base nDCG@10={baseline:.6f}; full CE nDCG@10={oracle_all_ce:.6f}")
    print(f"Rows in sweep: {len(sweep)}; calibration: {args.calibration_scores}")
    print(f"Recommended policy: {recommended if recommended else 'None reaches the defined targets'}")
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
