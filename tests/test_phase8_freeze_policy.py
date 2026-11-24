import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from budget_aware_search.routing import adaptive_reranking
from budget_aware_search.routing.freeze_policy import (
    POLICY,
    SCHEMA,
    _find_selected_row,
    _predict_cost,
    apply,
    decide_from_features,
    freeze,
    load_and_verify,
)
from budget_aware_search.routing.phase6_latency_aware import CostModel


def _minimal_policy(threshold=0.12):
    return {
        "schema": SCHEMA,
        "state": "FROZEN",
        "policy": POLICY,
        "cost_model": {"centers": [2.5, 7.5], "medians": [10.0, 20.0], "bin_width": 5},
        "decision": {"threshold": threshold, "operator": ">"},
        "feature_recipe": {"temperature": 0.1},
    }


def test_route_is_strict_threshold_and_uses_frozen_cost_interpolation():
    frame = pd.DataFrame({"query_id": ["a", "b", "c"], "entropy": [0.75, 0.50, 0.75], "candidate_count": [2, 2, 7]})
    policy = _minimal_policy(0.05)
    decisions = decide_from_features(frame, policy)
    assert decisions.rerank.tolist() == [True, False, False]  # 0.50/10 exactly ties threshold
    assert decisions.predicted_ce_ms.to_list()[0] == pytest.approx(10)
    assert decisions.predicted_ce_ms.to_list()[2] > 10


def test_apply_ignores_labels_and_ce_scores():
    scored = pd.DataFrame({
        "query_id": [1, 1, 2, 2],
        "score": [0.95, 0.89, 0.51, 0.50],
        "esci_label": ["E", "I", "E", "I"],
        "cross_score": [0.1, 0.9, 0.1, 0.9],
    })
    artifact = _minimal_policy(0.05)
    first = apply(scored, artifact)
    scored["esci_label"] = ["I", "E", "I", "E"]
    scored["cross_score"] = [0.9, 0.1, 0.9, 0.1]
    second = apply(scored, artifact)
    pd.testing.assert_frame_equal(first, second)
    assert first.query_id.nunique() == 2


def test_selected_phase7_row_requires_exact_phase6_match():
    sweep = pd.DataFrame({
        "policy": ["entropy_per_cost", "entropy_per_cost", "entropy"],
        "target_rerank_rate": [.40, .45, .45],
        "threshold_train_calibrated": [.065, .06151835348444984, .8],
        "actual_rerank_rate": [.4, .4468560485, .44],
        "p95_stage_latency_ms": [19.2, 19.7502611637, 19.3],
        "ndcg@10": [.744, .74712180136, .745],
    })
    selected = {"ndcg@10": .7471218013621901, "actual_rerank_rate": .4468560485157996, "p95_stage_latency_ms": 19.750261163711546}
    row = _find_selected_row(sweep, selected)
    assert row["threshold_train_calibrated"] == pytest.approx(.06151835348444984)
    with pytest.raises(ValueError, match="exactly one"):
        _find_selected_row(sweep, {**selected, "ndcg@10": .748})


def test_reject_bad_cost_values():
    with pytest.raises(ValueError, match="Invalid candidate"):
        _predict_cost(np.array([-2.0]), _minimal_policy()["cost_model"])
    with pytest.raises(ValueError, match="malformed"):
        _predict_cost(np.array([4]), {"centers": [2.0], "medians": [np.nan]})


def test_full_freeze_replays_validation_and_checks_integrity(tmp_path, monkeypatch):
    # Build fake validation metadata and timed rows; no GPU, no model inference.
    scored = pd.DataFrame([
        {"query_id": i, "score": score}
        for i in range(1, 21)
        for score in (0.93, 0.50 + i * 0.019)
    ])
    features = adaptive_reranking.query_features(scored, score_col="score")
    features["ndcg@10_bi"] = np.arange(20) / 30.0
    features["ndcg@10_ce"] = features["ndcg@10_bi"] + 0.05
    latency = pd.DataFrame([
        {"query_id": i, "candidate_count": 2, "bi_ms": 3.5, "ce_ms": 15.0}
        for i in range(1, 21) for _ in range(3)
    ])
    cost = CostModel(latency)
    p = features["entropy"].to_numpy(dtype=float) / cost.predict(features.candidate_count.to_numpy(dtype=float))
    threshold = float(np.quantile(p, .55))
    mask = p > threshold
    rate = float(mask.mean())
    ndcg = float(np.where(mask, features["ndcg@10_ce"], features["ndcg@10_bi"]).mean())
    p95 = float(np.quantile(3.5 + mask.astype(float).repeat(3) * 15, .95))

    root = tmp_path
    json_path = root / "summary.json"
    sweep_path = root / "sweep.csv"
    calibration_path = root / "train_scored.parquet"
    calibration_path.write_bytes(b"train calibrated scores")
    val_path = root / "val.parquet"
    latency_path = root / "latency.parquet"
    for x in (val_path, latency_path):
        x.write_bytes(b"test fixture")
    json_path.write_text(json.dumps({
        "test_evaluated": False,
        "quality_queries": 20,
        "latency_benchmark_queries": 20,
        "bi_ndcg@10": float(features["ndcg@10_bi"].mean()),
        "ce_ndcg@10": float(features["ndcg@10_ce"].mean()),
        "selection": {
            "max_rerank_rate": .5,
            "p95_stage_budget_ms": 20.0,
            "best_policy": {"policy": POLICY, "ndcg@10": ndcg, "actual_rerank_rate": rate, "p95_stage_latency_ms": p95, "gain_retained_pct": 50},
        },
    }))
    pd.DataFrame([{
        "policy": POLICY,
        "target_rerank_rate": .45,
        "threshold_train_calibrated": threshold,
        "actual_rerank_rate": rate,
        "p95_stage_latency_ms": p95,
        "ndcg@10": ndcg,
    }]).to_csv(sweep_path, index=False)
    bi_model, ce_model = root / "bi", root / "ce"
    for model in (bi_model, ce_model):
        model.mkdir()
        (model / "config.json").write_text('{"ok":true}')

    original_read = pd.read_parquet
    def fake_read(path, *args, **kwargs):
        if str(path) == str(val_path):
            return features
        if str(path) == str(latency_path):
            return latency
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", fake_read)

    output = root / "policy.json"
    frozen = freeze(phase7_summary=json_path, phase6_sweep=sweep_path, calibration_scores=calibration_path, val_features=val_path,
                    latency_samples=latency_path, bi_model=bi_model, ce_model=ce_model, output=output)
    assert frozen["state"] == "FROZEN"
    assert frozen["validation_evidence"]["replay_checked"] is True
    assert load_and_verify(output)["artifact_sha256"] == frozen["artifact_sha256"]
    with pytest.raises(FileExistsError):
        freeze(phase7_summary=json_path, phase6_sweep=sweep_path, calibration_scores=calibration_path, val_features=val_path,
               latency_samples=latency_path, bi_model=bi_model, ce_model=ce_model, output=output)
    (bi_model / "config.json").write_text('{"tampered":true}')
    with pytest.raises(RuntimeError, match="Checkpoint contents changed"):
        load_and_verify(output)
