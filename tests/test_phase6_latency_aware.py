import numpy as np
import pandas as pd
import pytest

from budget_aware_search.routing.phase6_latency_aware import (
    CostModel,
    choose_point,
    evaluate_routing,
    _validate_latency,
)


def fixture():
    val_rows = []
    train_rows = []
    latency_rows = []
    for qid in range(1, 11):
        n = 5 if qid <= 5 else 10
        val_rows.append({
            "query_id": qid,
            "candidate_count": n,
            "entropy": .5 + .03 * qid,
            "margin": .1 / qid,
            "top1": .8 - .02 * qid,
            "random": qid / 10,
            "ndcg@10_bi": .6,
            "ndcg@10_ce": .7 + .02 * (qid % 2),
            "mrr_exact_bi": .5,
            "mrr_exact_ce": .6,
            "exact_recall@5_bi": .4,
            "exact_recall@5_ce": .45,
            "pairwise_accuracy_bi": .55,
            "pairwise_accuracy_ce": .61,
        })
        for j in range(n):
            train_rows.append({
                "query_id": qid + 100,
                "query": f"query {qid}",
                "product_id": f"p{qid}-{j}",
                "score": .9 - .05 * j,
            })
        for repeat in range(2):
            latency_rows.append({
                "query_id": qid,
                "repeat": repeat,
                "candidate_count": n,
                "bi_ms": 3.0,
                "ce_ms": float(15 if n == 5 else 30),
            })
    return pd.DataFrame(val_rows), pd.DataFrame(train_rows), pd.DataFrame(latency_rows)


def test_cost_model_larger_groups_cost_more():
    _, _, latency = fixture()
    model = CostModel(latency)
    assert model.predict(np.array([5, 10]))[0] <= model.predict(np.array([5, 10]))[1]


def test_endpoints_and_measured_latency_replay():
    val, train, latency = fixture()
    sweep, oracle, diag = evaluate_routing(
        val, train, latency,
        rerank_rates=(0, .5, 1),
        quality_bootstrap_samples=20,
        latency_bootstrap_samples=10,
    )
    entropy = sweep.loc[sweep.policy.eq("entropy")].set_index("target_rerank_rate")
    assert entropy.loc[0, "ndcg@10"] == pytest.approx(.6)
    assert entropy.loc[1, "ndcg@10"] == pytest.approx(float(val["ndcg@10_ce"].mean()))
    assert entropy.loc[0, "p95_stage_latency_ms"] == pytest.approx(3.0)
    assert entropy.loc[1, "p95_stage_latency_ms"] == pytest.approx(33.0)
    assert len(sweep) == 6 * 3
    assert float(oracle.iloc[1]["oracle_ndcg@10"]) >= float(entropy.loc[.5, "ndcg@10"])
    assert diag["latency_benchmark_queries"] == 10


def test_mismatch_query_candidate_counts_rejected():
    val, _, latency = fixture()
    latency.loc[0, "candidate_count"] = 99
    with pytest.raises(ValueError, match="Candidate counts changed"):
        _validate_latency(latency, val)


def test_p95_constraint_protects_selection():
    sweep = pd.DataFrame({
        "policy": ["entropy", "entropy_per_cost"],
        "target_rerank_rate": [.5, .5],
        "actual_rerank_rate": [.5, .5],
        "benchmark_rerank_rate": [.5, .5],
        "threshold_train_calibrated": [.7, .8],
        "ndcg@10": [.75, .745],
        "gain_retained_pct": [60, 55],
        "p95_stage_latency_ms": [35, 25],
        "p95_ci95_low_ms": [33, 23],
        "p95_ci95_high_ms": [37, 27],
    })
    best = choose_point(sweep, max_rate=.5, p95_budget_ms=28, target_gain_pct=95)
    assert best["policy"] == "entropy_per_cost"
    assert not best["target_met"]
    no_solution = choose_point(sweep, max_rate=.5, p95_budget_ms=20, target_gain_pct=95)
    assert not no_solution["feasible"]
