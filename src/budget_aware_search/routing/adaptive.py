from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from budget_aware_search.metrics import evaluate_predictions
from budget_aware_search.utils import write_json


def merge_predictions(base: pd.DataFrame, reranker: pd.DataFrame) -> pd.DataFrame:
    keys = ["query_id", "product_id"]
    required = set(keys + ["score", "esci_label"])
    for name, frame in (("base", base), ("reranker", reranker)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} predictions missing columns: {sorted(missing)}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} predictions contain duplicate query/product pairs")

    base_columns = list(dict.fromkeys(keys + ["query", "product_text", "esci_label", "score"]))
    base_columns = [column for column in base_columns if column in base.columns]
    merged = base[base_columns].rename(columns={"score": "base_score"}).merge(
        reranker[keys + ["score"]].rename(columns={"score": "reranker_score"}),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(base) or len(merged) != len(reranker):
        raise ValueError(
            "Base and reranker predictions do not contain exactly the same query/product pairs"
        )
    return merged


def query_confidence_features(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for query_id, group in merged.groupby("query_id", sort=False):
        scores = np.sort(group["base_score"].to_numpy(dtype=np.float64))[::-1]
        margin = float(scores[0] - scores[1]) if len(scores) >= 2 else float("inf")
        centered = scores - scores.max()
        probabilities = np.exp(centered)
        probabilities /= probabilities.sum()
        entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
        normalized_entropy = entropy / np.log(len(scores)) if len(scores) > 1 else 0.0
        base_top = group.loc[group["base_score"].idxmax(), "product_id"]
        reranker_top = group.loc[group["reranker_score"].idxmax(), "product_id"]
        query_text = str(group["query"].iloc[0]) if "query" in group.columns else ""
        rows.append(
            {
                "query_id": query_id,
                "margin": margin,
                "normalized_entropy": normalized_entropy,
                "score_std": float(np.std(scores)),
                "query_length": len(query_text.split()),
                "top1_disagreement": float(base_top != reranker_top),
            }
        )
    return pd.DataFrame(rows)


def _per_query_metric(frame: pd.DataFrame, score_col: str, metric: str) -> pd.Series:
    temp = frame.rename(columns={score_col: "score"})
    _, per_query = evaluate_predictions(temp, score_col="score", bootstrap_samples=0)
    if metric not in per_query.columns:
        raise ValueError(f"Unknown routing metric {metric!r}; available: {per_query.columns.tolist()}")
    return per_query.set_index("query_id")[metric]


def fit_margin_router(
    merged: pd.DataFrame,
    base_latency_ms: float,
    reranker_latency_ms: float,
    latency_budget_ms: float,
    metric: str = "ndcg@10",
    sweep_points: int = 51,
) -> tuple[dict[str, object], pd.DataFrame]:
    features = query_confidence_features(merged).set_index("query_id")
    base_metric = _per_query_metric(merged, "base_score", metric)
    reranker_metric = _per_query_metric(merged, "reranker_score", metric)
    common_ids = features.index.intersection(base_metric.index).intersection(reranker_metric.index)
    features = features.loc[common_ids]
    base_metric = base_metric.loc[common_ids]
    reranker_metric = reranker_metric.loc[common_ids]

    max_route_rate = np.clip(
        (latency_budget_ms - base_latency_ms) / max(reranker_latency_ms, 1e-12),
        0.0,
        1.0,
    )
    rates = np.unique(np.concatenate([np.linspace(0.0, 1.0, sweep_points), [max_route_rate]]))
    ordered_ids = features.sort_values("margin", ascending=True).index.to_numpy()
    n_queries = len(ordered_ids)
    rows: list[dict[str, float]] = []
    for rate in rates:
        n_routed = min(n_queries, max(0, int(round(rate * n_queries))))
        routed_ids = set(ordered_ids[:n_routed].tolist())
        routed_mask = base_metric.index.to_series().isin(routed_ids).to_numpy()
        combined = np.where(routed_mask, reranker_metric.to_numpy(), base_metric.to_numpy())
        threshold = (
            float(features.loc[list(routed_ids), "margin"].max()) if routed_ids else float("-inf")
        )
        rows.append(
            {
                "route_rate": float(n_routed / n_queries),
                "margin_threshold": threshold,
                metric: float(combined.mean()),
                "latency_ms": float(base_latency_ms + (n_routed / n_queries) * reranker_latency_ms),
                "delta_vs_base": float(combined.mean() - base_metric.mean()),
            }
        )
    curve = pd.DataFrame(rows).sort_values(["route_rate", "latency_ms"]).drop_duplicates(
        "route_rate", keep="last"
    )
    feasible = curve.loc[curve["latency_ms"] <= latency_budget_ms + 1e-9]
    if feasible.empty:
        chosen = curve.iloc[0]
    else:
        chosen = feasible.sort_values([metric, "latency_ms"], ascending=[False, True]).iloc[0]

    policy: dict[str, object] = {
        "type": "margin_threshold",
        "feature": "margin",
        "route_when": "less_than_or_equal",
        "threshold": float(chosen["margin_threshold"]),
        "validation_route_rate": float(chosen["route_rate"]),
        "validation_metric": metric,
        "validation_metric_value": float(chosen[metric]),
        "base_latency_ms": float(base_latency_ms),
        "reranker_latency_ms": float(reranker_latency_ms),
        "latency_budget_ms": float(latency_budget_ms),
        "estimated_latency_ms": float(chosen["latency_ms"]),
    }
    return policy, curve


def apply_margin_router(merged: pd.DataFrame, policy: dict[str, object]) -> pd.DataFrame:
    if policy.get("type") != "margin_threshold":
        raise ValueError(f"Unsupported policy type: {policy.get('type')}")
    threshold = float(policy["threshold"])
    features = query_confidence_features(merged)
    routed_ids = set(features.loc[features["margin"] <= threshold, "query_id"].tolist())
    output = merged.copy()
    output["routed_to_reranker"] = output["query_id"].isin(routed_ids)
    output["score"] = np.where(
        output["routed_to_reranker"], output["reranker_score"], output["base_score"]
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit or apply a confidence-based reranking router.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--base", required=True)
    fit_parser.add_argument("--reranker", required=True)
    fit_parser.add_argument("--policy-output", required=True)
    fit_parser.add_argument("--curve-output", required=True)
    fit_parser.add_argument("--metric", default="ndcg@10")
    fit_parser.add_argument("--base-latency-ms", type=float, default=1.0)
    fit_parser.add_argument("--reranker-latency-ms", type=float, default=8.0)
    fit_parser.add_argument("--latency-budget-ms", type=float, default=5.0)
    fit_parser.add_argument("--sweep-points", type=int, default=51)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--base", required=True)
    apply_parser.add_argument("--reranker", required=True)
    apply_parser.add_argument("--policy", required=True)
    apply_parser.add_argument("--output", required=True)
    apply_parser.add_argument("--bootstrap-samples", type=int, default=0)

    args = parser.parse_args()
    base = pd.read_parquet(args.base)
    reranker = pd.read_parquet(args.reranker)
    merged = merge_predictions(base, reranker)

    if args.command == "fit":
        policy, curve = fit_margin_router(
            merged,
            base_latency_ms=args.base_latency_ms,
            reranker_latency_ms=args.reranker_latency_ms,
            latency_budget_ms=args.latency_budget_ms,
            metric=args.metric,
            sweep_points=args.sweep_points,
        )
        write_json(policy, args.policy_output)
        curve_output = Path(args.curve_output)
        curve_output.parent.mkdir(parents=True, exist_ok=True)
        curve.to_csv(curve_output, index=False)
        print(json.dumps(policy, indent=2))
    else:
        with open(args.policy, "r", encoding="utf-8") as f:
            policy = json.load(f)
        output_frame = apply_margin_router(merged, policy)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output_frame.to_parquet(output, index=False)
        summary, per_query = evaluate_predictions(
            output_frame, bootstrap_samples=args.bootstrap_samples
        )
        summary["actual_route_rate"] = float(output_frame.groupby("query_id")["routed_to_reranker"].first().mean())
        write_json(summary, output.with_suffix(".metrics.json"))
        per_query.to_parquet(output.with_suffix(".per_query.parquet"), index=False)
        print(f"Adaptive nDCG@10: {summary['metrics']['ndcg@10']['mean']:.6f}")
        print(f"Actual route rate: {summary['actual_route_rate']:.2%}")


if __name__ == "__main__":
    main()
