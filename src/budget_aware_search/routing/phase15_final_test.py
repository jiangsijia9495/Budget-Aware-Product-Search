"""Phase 15: frozen-policy test evaluation (no fitting or threshold selection).

Preflight verifies the frozen model/policy hashes, query-disjoint splits and
stable test candidate identities. Prepare-routed derives decisions exclusively
from bi-encoder scores via Phase 8 freeze_policy.apply and extracts *all*
candidates of the routed queries. Finalize validates candidate alignment,
computes test metrics and paired CIs, and optionally replays measured RTX-stage
latency on a test subset. Never tune or calibrate using the test results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from budget_aware_search.routing import freeze_policy
from budget_aware_search.routing.adaptive_reranking import validate_candidate_alignment

METRICS = ("ndcg@10", "mrr_exact", "exact_recall@5", "pairwise_accuracy")


def _json_save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _keycheck(frame: pd.DataFrame, name: str) -> None:
    keys = ["query_id", "product_id"]
    missing = set(keys) - set(frame)
    if missing:
        raise ValueError(f"{name} missing columns {sorted(missing)}")
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"{name} has null or duplicate query-product identities")


def _same_keys(a: pd.DataFrame, b: pd.DataFrame, name: str) -> None:
    _keycheck(a, "test")
    _keycheck(b, name)
    check = a[["query_id", "product_id"]].merge(
        b[["query_id", "product_id"]], on=["query_id", "product_id"],
        how="outer", validate="one_to_one", indicator=True,
    )
    if not check["_merge"].eq("both").all():
        raise ValueError(f"Candidate set mismatch between test and {name}: {check['_merge'].value_counts().to_dict()}")


def _validate_decisions(bi: pd.DataFrame, decisions: pd.DataFrame, policy: dict) -> pd.DataFrame:
    fresh = freeze_policy.apply(bi, policy, score_col="score")
    required = {"query_id", "rerank", "routing_priority", "predicted_ce_ms"}
    if not required.issubset(decisions.columns):
        raise ValueError(f"Decisions missing {sorted(required - set(decisions))}")
    if decisions.query_id.isna().any() or decisions.query_id.duplicated().any():
        raise ValueError("Decisions have duplicate or null query IDs")
    a = fresh.merge(decisions, on="query_id", how="outer", indicator=True, validate="one_to_one", suffixes=("_fresh", "_saved"))
    if not a["_merge"].eq("both").all():
        raise ValueError("Decisions do not cover exactly the scored test queries")
    if not (a["rerank_fresh"].to_numpy(dtype=bool) == a["rerank_saved"].to_numpy(dtype=bool)).all():
        raise ValueError("Saved test decisions differ from the frozen router decision")
    for column in ("routing_priority", "predicted_ce_ms"):
        if not np.allclose(a[f"{column}_fresh"], a[f"{column}_saved"], rtol=1e-10, atol=1e-10):
            raise ValueError(f"Saved decisions contain modified {column}")
    return fresh


def _load_test(path: Path) -> pd.DataFrame:
    test = pd.read_parquet(path)
    needed = {"query_id", "query", "product_id", "product_text", "esci_label"}
    missing = needed - set(test)
    if missing:
        raise ValueError(f"Test parquet missing {sorted(missing)}")
    _keycheck(test, "test")
    if set(test.esci_label.dropna().astype(str)) - {"E", "S", "C", "I"}:
        raise ValueError("Unexpected ESCI relevance labels")
    if test.esci_label.isna().any():
        raise ValueError("Cannot compute held-out quality without relevance labels")
    return test


def preflight(*, policy_file: Path, data_dir: Path, output_dir: Path) -> dict:
    policy = freeze_policy.load_and_verify(policy_file)
    train_path, val_path, test_path = (data_dir / f"{part}.parquet" for part in ("train", "val", "test"))
    for p in (train_path, val_path, test_path):
        if not p.exists():
            raise FileNotFoundError(p)
    ids = {}
    for part, path in zip(("train", "val", "test"), (train_path, val_path, test_path)):
        # Read only query IDs for split-disjointness checks. Never use test labels to set policy.
        ids[part] = set(pd.read_parquet(path, columns=["query_id"]).query_id.tolist())
    if ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"]:
        raise RuntimeError("Query overlap detected between train/val/test: refusing final evaluation")
    if not ids["test"]:
        raise ValueError("Empty TEST split")
    expected = policy["validation_evidence"].get("quality_queries")
    if expected is not None and len(ids["val"]) != int(expected):
        raise RuntimeError(f"Validation query count changed since freeze: {len(ids['val'])} != {expected}")
    _load_test(test_path)

    provenance = {
        "schema": "budget-aware-product-search.final-test.v1",
        "status": "PREPARED_NO_TEST_METRICS",
        "frozen_policy_sha256": policy["artifact_sha256"],
        "test_data_sha256": _sha256(test_path),
        "test_queries": len(ids["test"]),
        "train_queries": len(ids["train"]),
        "val_queries": len(ids["val"]),
        "policy": policy["policy"],
        "threshold": float(policy["decision"]["threshold"]),
        "data_dir": str(data_dir),
        "frozen_policy_path": str(policy_file),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != provenance:
            raise RuntimeError("Existing final-test run manifest differs; refusing to mix test artifacts")
    else:
        # Do not silently adopt files from a different run.
        if any(output_dir.glob("*.parquet")) or (output_dir / "final_report.json").exists():
            raise RuntimeError("Final-test outputs exist without a matching manifest; inspect/move them first")
        _json_save(manifest_path, provenance)
    return provenance


def prepare_routed(*, policy_file: Path, test_path: Path, bi_path: Path,
                   decisions_path: Path, routed_path: Path, manifest_path: Path) -> dict:
    policy = freeze_policy.load_and_verify(policy_file)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["frozen_policy_sha256"] != policy["artifact_sha256"] or manifest["test_data_sha256"] != _sha256(test_path):
        raise RuntimeError("Test split or frozen policy changed since preflight")
    test = _load_test(test_path)
    bi = pd.read_parquet(bi_path)
    _same_keys(test, bi, "bi")
    if "score" not in bi or not np.isfinite(bi["score"].to_numpy(dtype=float)).all():
        raise ValueError("Bi-encoder test scores missing or invalid")
    decisions = pd.read_parquet(decisions_path)
    actual = _validate_decisions(bi, decisions, policy)
    selected = set(actual.loc[actual.rerank, "query_id"])
    routed = test.loc[test.query_id.isin(selected)].copy()
    if routed.empty:
        raise ValueError("No test queries selected for CE; this runner expects an interior routing point")
    routed_path.parent.mkdir(parents=True, exist_ok=True)
    if routed_path.exists():
        existing = pd.read_parquet(routed_path)
        _same_keys(routed, existing, "existing routed candidates")
    else:
        routed.to_parquet(routed_path, index=False)
    return {"test_queries": int(test.query_id.nunique()), "routed_queries": int(len(selected)),
            "rerank_rate": float(len(selected) / test.query_id.nunique()),
            "routed_pairs": len(routed), "routed_data": str(routed_path)}


def _score_only(frame: pd.DataFrame, score_col: str, outname: str) -> pd.DataFrame:
    required = {"query_id", "product_id", score_col}
    if not required.issubset(frame.columns):
        raise ValueError(f"{outname} missing columns: {sorted(required - set(frame.columns))}")
    if not np.isfinite(frame[score_col].to_numpy(dtype=float)).all():
        raise ValueError(f"Nonfinite score in {outname}")
    return frame[["query_id", "product_id", score_col]].copy()


def _paired_compare(baseline: pd.DataFrame, candidate: pd.DataFrame, samples: int = 5000, seed: int = 42) -> dict:
    merged = baseline.merge(candidate, on="query_id", suffixes=("_baseline", "_candidate"), validate="one_to_one")
    if len(merged) != len(baseline) or len(merged) != len(candidate):
        raise RuntimeError("Unmatched query IDs in paired bootstrap comparison")
    rng = np.random.default_rng(seed)
    result = {"queries": len(merged), "metrics": {}}
    for metric in METRICS:
        diffs = merged[f"{metric}_candidate"].to_numpy(float) - merged[f"{metric}_baseline"].to_numpy(float)
        boot = np.empty(samples, dtype=float)
        for start in range(0, samples, 128):
            end = min(samples, start + 128)
            idx = rng.integers(0, len(diffs), size=(end - start, len(diffs)))
            boot[start:end] = diffs[idx].mean(axis=1)
        result["metrics"][metric] = {
            "baseline_mean": float(merged[f"{metric}_baseline"].mean()),
            "candidate_mean": float(merged[f"{metric}_candidate"].mean()),
            "mean_delta": float(diffs.mean()),
            "paired_ci95_low": float(np.quantile(boot, 0.025)),
            "paired_ci95_high": float(np.quantile(boot, 0.975)),
            "query_win_rate": float(np.mean(diffs > 1e-12)),
            "query_tie_rate": float(np.mean(np.abs(diffs) <= 1e-12)),
            "query_loss_rate": float(np.mean(diffs < -1e-12)),
        }
    return result


def _stage_latency(latency_path: Path, decisions: pd.DataFrame, test: pd.DataFrame,
                   samples: int = 300, seed: int = 42) -> dict:
    lat = pd.read_parquet(latency_path)
    required = {"query_id", "candidate_count", "bi_ms", "ce_ms"}
    if not required.issubset(lat):
        raise ValueError(f"Latency file missing columns: {sorted(required - set(lat.columns))}")
    if lat.empty or not set(lat.query_id).issubset(set(test.query_id)):
        raise ValueError("Invalid / non-test latency query IDs")
    counts = test.groupby("query_id").size()
    if not (lat.groupby("query_id")["candidate_count"].nunique() == 1).all():
        raise ValueError("Inconsistent measured candidate counts across repetitions")
    for row in lat[["query_id", "candidate_count"]].drop_duplicates().itertuples(index=False):
        if int(counts.loc[row.query_id]) != int(row.candidate_count):
            raise ValueError("Test latency candidates differ from scored test pool")
    rerank = decisions.set_index("query_id")["rerank"]
    lat = lat.copy()
    lat["rerank"] = lat.query_id.map(rerank)
    if lat.rerank.isna().any():
        raise ValueError("Missing routing decision for latency queries")
    for col in ("bi_ms", "ce_ms"):
        if not np.isfinite(lat[col].to_numpy(float)).all() or (lat[col] < 0).any():
            raise ValueError("Invalid measured stage latency")
    lat["adaptive_ms"] = lat.bi_ms + lat.rerank.to_numpy(bool) * lat.ce_ms
    lat["full_ce_ms"] = lat.bi_ms + lat.ce_ms
    result = {"scope": "test-subset RTX 3090 measured GPU-stage replay; not online end-to-end latency",
              "test_benchmark_queries": int(lat.query_id.nunique()),
              "test_benchmark_measurements": int(len(lat)),
              "benchmark_rerank_rate": float(lat.rerank.mean())}
    for name, col in (("bi_only", "bi_ms"), ("adaptive", "adaptive_ms"), ("full_ce", "full_ce_ms")):
        values = lat[col].to_numpy(float)
        result[name] = {"mean_ms": float(values.mean()), "p50_ms": float(np.quantile(values, .5)),
                        "p95_ms": float(np.quantile(values, .95)), "p99_ms": float(np.quantile(values, .99))}
    rng = np.random.default_rng(seed)
    grouped = [g.adaptive_ms.to_numpy(float) for _, g in lat.groupby("query_id", sort=False)]
    p95s = []
    for _ in range(samples):
        drawn = rng.integers(0, len(grouped), size=len(grouped))
        p95s.append(np.quantile(np.concatenate([grouped[i] for i in drawn]), .95))
    result["adaptive"]["p95_ci95_low_ms"] = float(np.quantile(p95s, .025))
    result["adaptive"]["p95_ci95_high_ms"] = float(np.quantile(p95s, .975))
    return result



def audit_ce_score_repeatability(
    routed: pd.DataFrame,
    full: pd.DataFrame,
) -> dict:
    """Audit two FP16 CE inference passes on the same query-product identities.

    This does NOT inspect ESCI labels and does NOT change scores or routing.
    Differences are expected when batch composition changes dynamic padding.
    We allow only small distribution-wide numerical drift; pronounced drift
    remains a hard error and requires inspecting the saved diagnostics.
    """
    _keycheck(routed, "routed CE")
    _keycheck(full, "full CE")
    if "cross_score" not in routed.columns or "cross_score" not in full.columns:
        raise ValueError("Both CE predictions must contain cross_score")

    pairs = routed[["query_id", "product_id", "cross_score"]].merge(
        full[["query_id", "product_id", "cross_score"]],
        how="left",
        on=["query_id", "product_id"],
        suffixes=("_routed", "_full"),
        validate="one_to_one",
        indicator=True,
    )
    if not pairs["_merge"].eq("both").all():
        raise RuntimeError("Routed CE contains candidates missing in full CE")

    r = pairs["cross_score_routed"].to_numpy(dtype=float)
    f = pairs["cross_score_full"].to_numpy(dtype=float)
    if not np.isfinite(r).all() or not np.isfinite(f).all():
        raise RuntimeError("CE predictions contain non-finite scores")
    if len(r) == 0:
        raise RuntimeError("No selected CE candidates to compare")
    absdiff = np.abs(r - f)

    # Expected gain from 4-class probabilities is in [0, 1]. Using absolute
    # scale is more interpretable than an rtol test near zero. These are
    # fixed inference-integrity tolerances, NOT model-selection parameters.
    # Large differences indicate changed models/tokenization or corrupt files.
    tolerances = {"mean_abs": 5e-4, "p99_abs": 5e-3, "max_abs": 2e-2}
    stats = {
        "rows": int(len(pairs)),
        "queries": int(pairs["query_id"].nunique()),
        "mean_abs_score_difference": float(np.mean(absdiff)),
        "p95_abs_score_difference": float(np.quantile(absdiff, 0.95)),
        "p99_abs_score_difference": float(np.quantile(absdiff, 0.99)),
        "max_abs_score_difference": float(np.max(absdiff)),
        "fraction_abs_difference_above_1e-4": float(np.mean(absdiff > 1e-4)),
        "fraction_abs_difference_above_1e-3": float(np.mean(absdiff > 1e-3)),
        "tolerances": tolerances,
    }

    # Ranking-disagreement diagnostics use only scores and candidate identities.
    # They do not use ground-truth labels. The *actual routed* CE predictions
    # remain the source of adaptive ranking, even when near-tie ranks differ.
    top1_disagreements = 0
    top10_set_disagreements = 0
    for _, group in pairs.groupby("query_id", sort=False):
        a = group.sort_values(
            ["cross_score_routed", "product_id"],
            ascending=[False, True], kind="mergesort"
        )["product_id"].astype(str).tolist()
        b = group.sort_values(
            ["cross_score_full", "product_id"],
            ascending=[False, True], kind="mergesort"
        )["product_id"].astype(str).tolist()
        top1_disagreements += bool(a[0] != b[0])
        top10_set_disagreements += bool(set(a[:10]) != set(b[:10]))
    stats["top1_disagreement_queries"] = int(top1_disagreements)
    stats["top10_set_disagreement_queries"] = int(top10_set_disagreements)

    passed = (
        stats["mean_abs_score_difference"] <= tolerances["mean_abs"]
        and stats["p99_abs_score_difference"] <= tolerances["p99_abs"]
        and stats["max_abs_score_difference"] <= tolerances["max_abs"]
    )
    stats["passed_numeric_consistency"] = bool(passed)
    stats["interpretation"] = (
        "Minor repeated-inference numerical drift; evaluate adaptive with routed scores "
        "and full-CE reference with full scores."
        if passed else
        "Score drift is too large to attribute safely to batch/FP16 noise; "
        "check checkpoint, model arguments, preprocessing and cached artifacts."
    )
    return stats

def finalize(*, policy_file: Path, test_path: Path, bi_path: Path, decisions_path: Path,
             routed_ce_path: Path, full_ce_path: Path, latency_path: Path | None,
             output_dir: Path, bootstrap_samples: int = 5000) -> dict:
    from budget_aware_search.cross_encoder_metrics import evaluate_scored_candidates
    final_path = output_dir / "final_report.json"
    if final_path.exists():
        raise FileExistsError(f"Final TEST report already exists: {final_path}; do not rerun/select on test")
    policy = freeze_policy.load_and_verify(policy_file)
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["frozen_policy_sha256"] != policy["artifact_sha256"] or manifest["test_data_sha256"] != _sha256(test_path):
        raise RuntimeError("Frozen policy or TEST data changed since preflight")

    test = _load_test(test_path)
    bi = pd.read_parquet(bi_path)
    _same_keys(test, bi, "bi")
    if "esci_label" in bi:
        validate_candidate_alignment(test, bi)
    decisions = _validate_decisions(bi, pd.read_parquet(decisions_path), policy)
    selected_queries = set(decisions.loc[decisions.rerank, "query_id"])
    expected = test.loc[test.query_id.isin(selected_queries)]
    ce_routed = pd.read_parquet(routed_ce_path)
    _same_keys(expected, ce_routed, "routed CE")
    if "esci_label" in ce_routed:
        validate_candidate_alignment(expected, ce_routed)
    if set(ce_routed.query_id) != selected_queries:
        raise ValueError("CE scored an incorrect subset of test queries")
    ce_full = pd.read_parquet(full_ce_path)
    _same_keys(test, ce_full, "full CE")
    if "esci_label" in ce_full:
        validate_candidate_alignment(test, ce_full)

    scored = test.copy()
    scored = scored.merge(_score_only(bi, "score", "bi"), on=["query_id", "product_id"], how="left", validate="one_to_one")
    scored = scored.merge(_score_only(ce_routed, "cross_score", "routed CE"), on=["query_id", "product_id"], how="left", validate="one_to_one")
    scored = scored.merge(_score_only(ce_full, "cross_score", "full CE").rename(columns={"cross_score": "full_ce_score"}),
                          on=["query_id", "product_id"], how="left", validate="one_to_one")
    if scored["score"].isna().any() or scored["full_ce_score"].isna().any():
        raise RuntimeError("Missing bi or full CE candidate scores")
    mask = scored.query_id.isin(selected_queries)
    if scored.loc[mask, "cross_score"].isna().any():
        raise RuntimeError("Missing routed Cross-Encoder scores")
    # Audit the two inference passes; FP16 outputs need not be bit-identical
    # when routed/full candidate batches have different dynamic padding.
    # IMPORTANT: keep the routed scores for adaptive ranking; do not silently
    # replace them with the full-CE scores just to make the test pass.
    ce_repeatability = audit_ce_score_repeatability(ce_routed, ce_full)
    _json_save(output_dir / "ce_inference_repeatability.json", ce_repeatability)
    if not ce_repeatability["passed_numeric_consistency"]:
        raise RuntimeError(
            "Routed/full CE numeric consistency FAILED: "
            f"mean_abs={ce_repeatability['mean_abs_score_difference']:.6g}, "
            f"p99_abs={ce_repeatability['p99_abs_score_difference']:.6g}, "
            f"max_abs={ce_repeatability['max_abs_score_difference']:.6g}, "
            f"top1_disagreement_queries={ce_repeatability['top1_disagreement_queries']}. "
            "Inspect outputs/final_test/ce_inference_repeatability.json; "
            "do not change the frozen policy or use test labels to repair predictions."
        )
    scored["adaptive_score"] = np.where(mask, scored["cross_score"], scored["score"])
    scored["rerank"] = mask
    reports = {}
    per_query = {}
    for name, col in (("bi", "score"), ("adaptive", "adaptive_score"), ("full_ce", "full_ce_score")):
        reports[name], per_query[name] = evaluate_scored_candidates(scored, score_column=col)
    cmp_bi = _paired_compare(per_query["bi"], per_query["adaptive"], samples=bootstrap_samples)
    cmp_ce = _paired_compare(per_query["adaptive"], per_query["full_ce"], samples=bootstrap_samples)

    latency = _stage_latency(latency_path, decisions, test) if latency_path is not None else None
    gain_denom = reports["full_ce"]["ndcg@10"] - reports["bi"]["ndcg@10"]
    retained = (
        float(100.0 * (reports["adaptive"]["ndcg@10"] - reports["bi"]["ndcg@10"]) / gain_denom)
        if abs(gain_denom) > 1e-12 else None
    )
    budget = float(policy["decision"]["p95_budget_ms_validation_constraint"])
    rate_cap = float(policy["decision"]["max_rerank_rate_validation_constraint"])
    measured_p95 = latency["adaptive"]["p95_ms"] if latency else None
    report = {
        "status": "FINAL_HELD_OUT_TEST_EVALUATED",
        "no_test_fitting_or_threshold_selection": True,
        "frozen_policy_sha256": policy["artifact_sha256"],
        "policy": policy["policy"],
        "threshold": float(policy["decision"]["threshold"]),
        "test_queries": int(test.query_id.nunique()),
        "test_candidates": int(len(test)),
        "routed_queries": int(len(selected_queries)),
        "rerank_rate": float(len(selected_queries) / test.query_id.nunique()),
        "gain_retained_pct_vs_full_ce": retained,
        "frozen_validation_budget": {
            "max_rerank_rate": rate_cap,
            "max_p95_stage_latency_ms": budget,
        },
        "test_budget_checks": {
            "rerank_rate_within_validation_cap": float(len(selected_queries) / test.query_id.nunique()) <= rate_cap,
            "p95_stage_latency_within_validation_budget": (measured_p95 <= budget if measured_p95 is not None else None),
        },
        "metrics": reports,
        "paired_adaptive_vs_bi": cmp_bi,
        "paired_full_ce_vs_adaptive": cmp_ce,
        "measured_latency": latency,
        "ce_inference_repeatability": ce_repeatability,
        "validation_reference_only": policy["validation_evidence"],
        "limitations": "ESCI judged candidate-pool reranking (not full-catalog retrieval); measured-stage latency replay excludes FAISS, networking and concurrent serving; do not retune from test results.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(output_dir / "adaptive_test_scored.parquet", index=False)
    for name, frame in per_query.items():
        frame.to_parquet(output_dir / f"{name}_test.per_query.parquet", index=False)
    _json_save(output_dir / "adaptive_vs_bi.paired.json", cmp_bi)
    _json_save(output_dir / "full_ce_vs_adaptive.paired.json", cmp_ce)
    _json_save(final_path, report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    cmd = ap.add_subparsers(dest="command", required=True)
    p = cmd.add_parser("preflight")
    p.add_argument("--policy", default="models/router_entropy_per_cost_frozen/policy.json")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--output-dir", default="outputs/final_test")
    p = cmd.add_parser("prepare-routed")
    p.add_argument("--policy", default="models/router_entropy_per_cost_frozen/policy.json")
    p.add_argument("--test", default="data/processed/test.parquet")
    p.add_argument("--bi-scores", default="outputs/final_test/biencoder_v1_test.parquet")
    p.add_argument("--decisions", default="outputs/final_test/router_decisions.parquet")
    p.add_argument("--output", default="data/processed/phase15_routed/test.parquet")
    p.add_argument("--manifest", default="outputs/final_test/run_manifest.json")
    p = cmd.add_parser("finalize")
    p.add_argument("--policy", default="models/router_entropy_per_cost_frozen/policy.json")
    p.add_argument("--test", default="data/processed/test.parquet")
    p.add_argument("--bi-scores", default="outputs/final_test/biencoder_v1_test.parquet")
    p.add_argument("--decisions", default="outputs/final_test/router_decisions.parquet")
    p.add_argument("--routed-ce", default="outputs/final_test/cross_encoder_routed_test.parquet")
    p.add_argument("--full-ce", default="outputs/final_test/cross_encoder_full_test.parquet")
    p.add_argument("--latency", default="outputs/final_test/latency_samples.parquet")
    p.add_argument("--output-dir", default="outputs/final_test")
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    args = ap.parse_args()
    if args.command == "preflight":
        print(json.dumps(preflight(policy_file=Path(args.policy), data_dir=Path(args.data_dir), output_dir=Path(args.output_dir)), indent=2))
    elif args.command == "prepare-routed":
        print(json.dumps(prepare_routed(policy_file=Path(args.policy), test_path=Path(args.test),
                 bi_path=Path(args.bi_scores), decisions_path=Path(args.decisions),
                 routed_path=Path(args.output), manifest_path=Path(args.manifest)), indent=2))
    else:
        print(json.dumps(finalize(policy_file=Path(args.policy), test_path=Path(args.test),
                 bi_path=Path(args.bi_scores), decisions_path=Path(args.decisions),
                 routed_ce_path=Path(args.routed_ce), full_ce_path=Path(args.full_ce),
                 latency_path=(Path(args.latency) if args.latency else None),
                 output_dir=Path(args.output_dir), bootstrap_samples=args.bootstrap_samples), indent=2))


if __name__ == "__main__":
    main()
