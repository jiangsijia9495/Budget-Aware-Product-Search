import pandas as pd

from budget_aware_search.routing.adaptive import (
    apply_margin_router,
    fit_margin_router,
    merge_predictions,
)


def _predictions(scores):
    return pd.DataFrame(
        {
            "query_id": [1, 1, 2, 2],
            "query": ["q1", "q1", "q2", "q2"],
            "product_id": ["a", "b", "c", "d"],
            "esci_label": ["E", "I", "E", "I"],
            "score": scores,
        }
    )


def test_router_fit_and_apply():
    base = _predictions([0.9, 0.1, 0.49, 0.51])
    reranker = _predictions([0.9, 0.1, 0.9, 0.1])
    merged = merge_predictions(base, reranker)
    policy, curve = fit_margin_router(
        merged,
        base_latency_ms=1.0,
        reranker_latency_ms=4.0,
        latency_budget_ms=3.0,
        sweep_points=3,
    )
    output = apply_margin_router(merged, policy)
    assert not curve.empty
    assert "score" in output
    assert output.groupby("query_id")["routed_to_reranker"].first().mean() <= 0.5
