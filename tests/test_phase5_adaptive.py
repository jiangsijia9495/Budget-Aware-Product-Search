import numpy as np
import pandas as pd
import pytest

from budget_aware_search.routing.adaptive_reranking import (
    build_pareto,
    query_features,
    validate_candidate_alignment,
    pareto_non_dominated,
)


def fake_data():
    products = [("p1", "E"), ("p2", "I"), ("p3", "S")]
    rows = []
    for qid in range(1, 11):
        for rank, (pid, label) in enumerate(products):
            rows.append(
                {"query_id": qid, "query": f"query {qid}", "product_id": pid,
                 "product_text": f"product {pid}", "esci_label": label,
                 "score": 0.7 - 0.1 * rank + qid * 0.001}
            )
    bi = pd.DataFrame(rows)
    ce = bi.drop(columns=["score"]).copy()
    ce["cross_score"] = np.linspace(0, 1, len(ce))
    pm_bi = pd.DataFrame(
        {"query_id": range(1, 11), "ndcg@10": [0.6] * 10,
         "mrr_exact": [0.5] * 10, "exact_recall@5": [0.4] * 10,
         "pairwise_accuracy": [0.6] * 10}
    )
    pm_ce = pm_bi.copy()
    pm_ce["ndcg@10"] = 0.8
    pm_ce["mrr_exact"] = 0.75
    pm_ce["exact_recall@5"] = 0.7
    pm_ce["pairwise_accuracy"] = 0.65
    return bi, ce, pm_bi, pm_ce


def test_ends_exactly_match_phase4_metrics():
    bi, ce, m_bi, m_ce = fake_data()
    sweep, features, decisions = build_pareto(bi, ce, m_bi, m_ce, bi)
    for policy in ("margin", "entropy", "top1", "random"):
        part = sweep.loc[sweep["policy"].eq(policy)].set_index("target_rerank_rate")
        assert part.loc[0.0, "ndcg@10"] == pytest.approx(0.6)
        assert part.loc[1.0, "ndcg@10"] == pytest.approx(0.8)
        assert part.loc[0.0, "rerank_rate"] == 0
        assert part.loc[1.0, "rerank_rate"] == 1
        assert part.loc[0.0, "mrr_exact"] == pytest.approx(0.5)
        assert part.loc[1.0, "mrr_exact"] == pytest.approx(0.75)
    assert len(features) == 10
    assert len(decisions) == 4 * 11 * 10


def test_latencies_are_substituted_only_for_routed_queries():
    bi, ce, m_bi, m_ce = fake_data()
    latency = pd.DataFrame({"query_id": range(1, 11), "bi_ms": [4.0] * 10, "ce_ms": [10.0] * 10})
    sweep, _, _ = build_pareto(bi, ce, m_bi, m_ce, bi, latency=latency)
    part = sweep.loc[sweep["policy"].eq("random")].set_index("target_rerank_rate")
    assert part.loc[0.0, "p95_latency_ms"] == pytest.approx(4.0)
    assert part.loc[1.0, "p95_latency_ms"] == pytest.approx(14.0)
    assert part.loc[0.0, "mean_latency_ms"] == pytest.approx(4.0)
    assert part.loc[1.0, "mean_latency_ms"] == pytest.approx(14.0)


def test_mismatched_candidates_raise():
    bi, ce, m_bi, m_ce = fake_data()
    ce = ce.iloc[:-1]
    with pytest.raises(ValueError, match="candidate pools differ"):
        validate_candidate_alignment(bi, ce)


def test_query_features_are_label_independent():
    bi, _, _, _ = fake_data()
    no_labels = bi.drop(columns=["esci_label"])
    f1 = query_features(bi)
    f2 = query_features(no_labels)
    pd.testing.assert_frame_equal(f1, f2)
    assert (f1["margin"] > 0).all()


def test_pareto_removes_dominated_points():
    frame = pd.DataFrame({"policy": ["A", "B", "C", "D"],
                          "p95_latency_ms": [5, 8, 9, 10],
                          "ndcg@10": [0.70, 0.80, 0.78, 0.82]})
    frontier = pareto_non_dominated(frame)
    assert frontier["policy"].tolist() == ["A", "B", "D"]
