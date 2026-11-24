"""Freeze a Phase 7 validation-selected policy; never calibrate on official TEST.

Freeze saves: fixed threshold, self-contained latency-cost interpolation, exact
feature/decision semantics, checkpoint checksums, selection provenance, and a
strict replay of the Phase 6 validation operating point.

Apply uses only bi-encoder inference scores and the frozen artifact. It never
uses labels, Cross-Encoder predictions, or latency observations from TEST.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from budget_aware_search.routing import adaptive_reranking
from budget_aware_search.routing.phase6_latency_aware import CostModel

SCHEMA = "budget-aware-product-search.frozen-router.v1"
POLICY = "entropy_per_cost"
TEMPERATURE = 0.1
SEED = 42
FLOAT_ABS_TOL = 1e-7


def _digest_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _checkpoint_manifest(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing checkpoint: {root}")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"Checkpoint directory contains no files: {root}")
    return {
        "path": str(root),
        "files": [
            {"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": _digest_file(p)}
            for p in files
        ],
    }


def _verify_checkpoint(saved: dict[str, Any]) -> None:
    root = Path(saved["path"])
    actual = _checkpoint_manifest(root)
    if actual["files"] != saved["files"]:
        raise RuntimeError(f"Checkpoint contents changed since policy freeze: {root}")


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip()
        return {"commit": commit, "working_tree_dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}


def _safe_float(value: Any, name: str) -> float:
    answer = float(value)
    if not math.isfinite(answer):
        raise ValueError(f"{name} must be finite")
    return answer


def _find_selected_row(sweep: pd.DataFrame, selected: dict) -> pd.Series:
    required = {"policy", "target_rerank_rate", "threshold_train_calibrated", "actual_rerank_rate", "p95_stage_latency_ms", "ndcg@10"}
    if not required.issubset(sweep.columns):
        raise ValueError(f"Phase 6 sweep missing columns: {sorted(required - set(sweep.columns))}")
    subset = sweep.loc[sweep["policy"].eq(POLICY)].copy()
    if subset.empty:
        raise ValueError(f"No {POLICY} row in Phase 6 sweep")
    for key in ("ndcg@10", "actual_rerank_rate", "p95_stage_latency_ms"):
        expected = _safe_float(selected[key], key)
        subset = subset.loc[np.isclose(subset[key].astype(float), expected, rtol=0.0, atol=FLOAT_ABS_TOL)]
    if len(subset) != 1:
        raise ValueError(f"Expected exactly one Phase 6 sweep row matching Phase 7 selection; got {len(subset)}")
    row = subset.iloc[0]
    rate = _safe_float(row["target_rerank_rate"], "target_rerank_rate")
    if not 0 < rate < 1:
        raise ValueError("Expected interior fixed-threshold routing point")
    return row


def _predict_cost(counts: np.ndarray, cost_parameters: dict) -> np.ndarray:
    arr = np.asarray(counts, dtype=float)
    if not np.isfinite(arr).all() or (arr <= 0).any():
        raise ValueError("Invalid candidate counts")
    xs = np.asarray(cost_parameters["centers"], dtype=float)
    ys = np.asarray(cost_parameters["medians"], dtype=float)
    if not len(xs) or len(xs) != len(ys) or not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        raise ValueError("Frozen cost interpolation is malformed")
    if not np.all(np.diff(xs) > 0) or np.any(ys <= 0):
        raise ValueError("Frozen cost interpolation is not valid")
    return np.maximum(0.001, np.interp(arr, xs, ys, left=float(ys[0]), right=float(ys[-1])))


def decide_from_features(features: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    if artifact.get("schema") != SCHEMA or artifact.get("policy") != POLICY:
        raise ValueError("Unsupported frozen policy artifact")
    required = {"query_id", "entropy", "candidate_count"}
    if not required.issubset(features.columns):
        raise ValueError(f"Router features missing columns: {sorted(required - set(features.columns))}")
    if features["query_id"].isna().any() or features["query_id"].duplicated().any():
        raise ValueError("Query IDs must be non-null and unique")
    ent = features["entropy"].to_numpy(dtype=float)
    if not np.isfinite(ent).all() or np.any(ent < -1e-8) or np.any(ent > 1 + 1e-8):
        raise ValueError("Invalid normalized entropy")
    predicted_cost = _predict_cost(features["candidate_count"].to_numpy(dtype=float), artifact["cost_model"])
    values = ent / predicted_cost
    threshold = _safe_float(artifact["decision"]["threshold"], "threshold")
    decisions = features[["query_id", "candidate_count", "entropy"]].copy()
    decisions["predicted_ce_ms"] = predicted_cost
    decisions["routing_priority"] = values
    decisions["rerank"] = values > threshold
    return decisions


def _validate_replay(
    val: pd.DataFrame,
    latency: pd.DataFrame,
    artifact: dict,
    selected_row: pd.Series,
) -> dict[str, float]:
    required = {"query_id", "candidate_count", "entropy", "ndcg@10_bi", "ndcg@10_ce"}
    if not required.issubset(val.columns):
        raise ValueError(f"Validation feature table missing columns: {sorted(required - set(val.columns))}")
    decisions = decide_from_features(val, artifact)
    mask = decisions["rerank"].to_numpy(dtype=bool)
    val_ndcg = float(np.where(mask, val["ndcg@10_ce"], val["ndcg@10_bi"]).mean())
    val_rate = float(mask.mean())

    required_latency = {"query_id", "candidate_count", "bi_ms", "ce_ms"}
    if not required_latency.issubset(latency.columns):
        raise ValueError(f"Latency samples missing: {sorted(required_latency - set(latency.columns))}")
    if latency["query_id"].isna().any() or not set(latency["query_id"]).issubset(set(decisions["query_id"])):
        raise ValueError("Latency queries do not match validation")
    counts = latency[["query_id", "candidate_count"]].drop_duplicates()
    if counts["query_id"].duplicated().any():
        raise ValueError("Inconsistent repeated candidate counts for latency queries")
    lookup = decisions.set_index("query_id")
    for row in counts.itertuples(index=False):
        if int(lookup.loc[row.query_id, "candidate_count"]) != int(row.candidate_count):
            raise ValueError("Candidate counts changed between validation and latency benchmark")
    timed_mask = latency["query_id"].map(decisions.set_index("query_id")["rerank"])
    observed_cost = latency["bi_ms"].to_numpy(dtype=float) + timed_mask.to_numpy(dtype=bool) * latency["ce_ms"].to_numpy(dtype=float)
    if not np.isfinite(observed_cost).all() or (observed_cost <= 0).any():
        raise ValueError("Invalid latency samples")
    p95_ms = float(np.quantile(observed_cost, 0.95))

    comparisons = {
        "ndcg@10": (val_ndcg, float(selected_row["ndcg@10"])),
        "actual_rerank_rate": (val_rate, float(selected_row["actual_rerank_rate"])),
        "p95_stage_latency_ms": (p95_ms, float(selected_row["p95_stage_latency_ms"])),
    }
    for label, (observed, recorded) in comparisons.items():
        if not np.isclose(observed, recorded, rtol=0, atol=FLOAT_ABS_TOL):
            raise RuntimeError(f"Freeze replay mismatch for {label}: recomputed {observed}, Phase 6 sweep {recorded}")
    return {key: observed for key, (observed, _) in comparisons.items()}


def freeze(
    *,
    phase7_summary: Path,
    phase6_sweep: Path,
    calibration_scores: Path,
    val_features: Path,
    latency_samples: Path,
    bi_model: Path,
    ce_model: Path,
    output: Path,
    force: bool = False,
) -> dict[str, Any]:
    if output.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite frozen artifact: {output}; use a different path")
    for p in (phase7_summary, phase6_sweep, calibration_scores, val_features, latency_samples):
        if not p.is_file():
            raise FileNotFoundError(p)

    summary = json.loads(phase7_summary.read_text(encoding="utf-8"))
    selection = summary["selection"]
    best = selection["best_policy"]
    if not best or best["policy"] != POLICY:
        raise ValueError(f"Phase 7 did not select {POLICY}; selected {best}")
    if summary.get("test_evaluated") is not False:
        raise ValueError("Freeze must happen before any official TEST evaluation")
    sweep = pd.read_csv(phase6_sweep)
    row = _find_selected_row(sweep, best)
    latency = pd.read_parquet(latency_samples)
    val = pd.read_parquet(val_features)
    cost = CostModel(latency, bin_width=5)

    threshold = _safe_float(row["threshold_train_calibrated"], "threshold_train_calibrated")
    max_rate = _safe_float(selection["max_rerank_rate"], "max_rerank_rate")
    budget = _safe_float(selection["p95_stage_budget_ms"], "p95_stage_budget_ms")
    if float(row["actual_rerank_rate"]) > max_rate + 1e-12 or float(row["p95_stage_latency_ms"]) > budget + 1e-12:
        raise ValueError("Selected point does not satisfy specified validation operating constraints")

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "state": "FROZEN",
        "policy": POLICY,
        "feature_recipe": {
            "source": "bi_encoder_scores_only",
            "score_column": "score",
            "temperature": TEMPERATURE,
            "entropy_definition": "-sum(softmax(score / temperature) * log(softmax(score / temperature))) / log(candidate_count)",
            "query_feature_code_sha256": _digest_file(Path(adaptive_reranking.__file__)),
            "freeze_apply_code_sha256": _digest_file(Path(__file__)),
        },
        "cost_model": {
            "kind": "ce_median_latency_bin_interpolation",
            "bin_width": 5,
            "centers": [float(v) for v in cost.centers],
            "medians": [float(v) for v in cost.medians],
            "latency_scope": "RTX 3090 measured GPU-stage per-query replay; not online end-to-end P95",
        },
        "decision": {
            "threshold": threshold,
            "operator": ">",
            "priority": "normalized_entropy / predicted_ce_ms",
            "target_rerank_rate_for_calibration_only": float(row["target_rerank_rate"]),
            "max_rerank_rate_validation_constraint": max_rate,
            "p95_budget_ms_validation_constraint": budget,
            "do_not_recalibrate_on_test": True,
            "per_query_decisions_do_not_use_labels": True,
        },
        "validation_evidence": {
            "ndcg@10": float(row["ndcg@10"]),
            "rerank_rate": float(row["actual_rerank_rate"]),
            "p95_stage_latency_ms": float(row["p95_stage_latency_ms"]),
            "gain_retained_pct": float(best["gain_retained_pct"]),
            "quality_queries": int(summary["quality_queries"]),
            "latency_queries": int(summary["latency_benchmark_queries"]),
        },
        "models": {
            "bi_encoder": _checkpoint_manifest(bi_model),
            "cross_encoder": _checkpoint_manifest(ce_model),
        },
        "provenance": {
            "selection_summary_path": str(phase7_summary),
            "selection_summary_sha256": _digest_file(phase7_summary),
            "phase6_sweep_path": str(phase6_sweep),
            "phase6_sweep_sha256": _digest_file(phase6_sweep),
            "train_calibration_scores_path": str(calibration_scores),
            "train_calibration_scores_sha256": _digest_file(calibration_scores),
            "val_features_path": str(val_features),
            "val_features_sha256": _digest_file(val_features),
            "latency_samples_path": str(latency_samples),
            "latency_samples_sha256": _digest_file(latency_samples),
            "git": _git_metadata(),
        },
    }
    replay = _validate_replay(val, latency, artifact, row)
    for key, column in (("bi_ndcg@10", "ndcg@10_bi"), ("ce_ndcg@10", "ndcg@10_ce")):
        measured = float(val[column].mean())
        expected = float(summary[key])
        if not np.isclose(measured, expected, atol=FLOAT_ABS_TOL, rtol=0):
            raise RuntimeError(f"Champion endpoint mismatch for {key}: {measured} vs {expected}")
    artifact["validation_evidence"]["bi_ndcg@10"] = float(summary["bi_ndcg@10"])
    artifact["validation_evidence"]["ce_ndcg@10"] = float(summary["ce_ndcg@10"])
    artifact["validation_evidence"]["replay_checked"] = True
    artifact["validation_evidence"]["replay"] = replay
    artifact["artifact_sha256"] = hashlib.sha256(
        json.dumps(artifact, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(output.parent), prefix=f".{output.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(artifact, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, output)
    return artifact


def load_and_verify(path: Path, *, check_models: bool = True) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("schema") != SCHEMA or artifact.get("state") != "FROZEN":
        raise ValueError("Not a frozen Phase 8 policy")
    signature = artifact.get("artifact_sha256")
    data = dict(artifact)
    data.pop("artifact_sha256", None)
    calculated = hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()
    if signature != calculated:
        raise RuntimeError("Policy artifact content changed since freeze")
    checks = {
        "query_feature_code_sha256": _digest_file(Path(adaptive_reranking.__file__)),
        "freeze_apply_code_sha256": _digest_file(Path(__file__)),
    }
    for key, current in checks.items():
        if current != artifact["feature_recipe"][key]:
            raise RuntimeError(f"Router implementation changed since freeze: {key}")
    if check_models:
        for model in artifact["models"].values():
            _verify_checkpoint(model)
    return artifact


def apply(scored: pd.DataFrame, artifact: dict[str, Any], *, score_col: str = "score") -> pd.DataFrame:
    # Intentionally ignore all labels/CE scores if they are present on a scored table.
    if not {"query_id", score_col}.issubset(scored.columns):
        raise ValueError(f"Scored query frame needs query_id and {score_col}")
    frame = scored[["query_id", score_col]].copy()
    frame = frame.rename(columns={score_col: "score"})
    features = adaptive_reranking.query_features(
        frame, score_col="score", temperature=float(artifact["feature_recipe"]["temperature"]), seed=SEED
    )
    return decide_from_features(features, artifact)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze", help="Freeze exact Phase 7 winner and audit validation replay")
    f.add_argument("--phase7-summary", default="outputs/phase7_learned/summary.json")
    f.add_argument("--phase6-sweep", default="outputs/phase6_latency_aware/policy_sweep.csv")
    f.add_argument("--calibration-scores", default="outputs/biencoder_v1_random_train_scored.parquet")
    f.add_argument("--val-features", default="outputs/adaptive_reranking/query_features_and_metrics.parquet")
    f.add_argument("--latency", default="outputs/adaptive_reranking/latency_samples.parquet")
    f.add_argument("--bi-model", default="models/biencoder_v1_random/final")
    f.add_argument("--ce-model", default="models/cross_encoder_esci_v1/final")
    f.add_argument("--output", default="models/router_entropy_per_cost_frozen/policy.json")
    f.add_argument("--force", action="store_true")

    v = sub.add_parser("verify", help="Verify policy integrity and both model checkpoint hashes")
    v.add_argument("--policy", default="models/router_entropy_per_cost_frozen/policy.json")

    a = sub.add_parser("apply", help="Apply frozen policy to bi-encoder-scored pairs, without labels")
    a.add_argument("--policy", default="models/router_entropy_per_cost_frozen/policy.json")
    a.add_argument("--bi-scores", required=True)
    a.add_argument("--score-column", default="score")
    a.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.command == "freeze":
        artifact = freeze(
            phase7_summary=Path(args.phase7_summary),
            phase6_sweep=Path(args.phase6_sweep),
            calibration_scores=Path(args.calibration_scores),
            val_features=Path(args.val_features),
            latency_samples=Path(args.latency),
            bi_model=Path(args.bi_model),
            ce_model=Path(args.ce_model),
            output=Path(args.output),
            force=args.force,
        )
        print(f"FROZEN: {args.output}")
        print(json.dumps({"policy": artifact["policy"], "threshold": artifact["decision"]["threshold"], "validation": artifact["validation_evidence"], "sha256": artifact["artifact_sha256"]}, indent=2))
    elif args.command == "verify":
        artifact = load_and_verify(Path(args.policy))
        print(f"PASS: frozen policy, both checkpoints, and router feature code verified; policy_sha256={artifact['artifact_sha256']}")
    elif args.command == "apply":
        artifact = load_and_verify(Path(args.policy))
        if not Path(args.bi_scores).is_file():
            raise FileNotFoundError(args.bi_scores)
        decisions = apply(pd.read_parquet(args.bi_scores), artifact, score_col=args.score_column)
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        decisions.to_parquet(target, index=False)
        observed_rate = float(decisions.rerank.mean())
        print(f"Saved {len(decisions)} query decisions: {target}; rerank rate {observed_rate:.3%}")
        if observed_rate > artifact["decision"]["max_rerank_rate_validation_constraint"]:
            print("WARNING: test rerank rate exceeds validation operating constraint; do not tune threshold on test")
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
