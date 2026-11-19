import pandas as pd

from budget_aware_search.metrics import evaluate_predictions


def _frame(scores):
    return pd.DataFrame(
        {
            "query_id": [1, 1, 1, 1],
            "product_id": ["a", "b", "c", "d"],
            "esci_label": ["E", "S", "C", "I"],
            "score": scores,
        }
    )


def test_perfect_ranking_has_unit_ndcg():
    summary, _ = evaluate_predictions(_frame([4.0, 3.0, 2.0, 1.0]))
    assert summary["metrics"]["ndcg@10"]["mean"] == 1.0
    assert summary["metrics"]["mrr_exact"]["mean"] == 1.0


def test_reversed_ranking_is_worse():
    perfect, _ = evaluate_predictions(_frame([4.0, 3.0, 2.0, 1.0]))
    reversed_result, _ = evaluate_predictions(_frame([1.0, 2.0, 3.0, 4.0]))
    assert reversed_result["metrics"]["ndcg@10"]["mean"] < perfect["metrics"]["ndcg@10"]["mean"]
