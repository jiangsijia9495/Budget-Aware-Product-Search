import numpy as np
import pandas as pd
import pytest

from budget_aware_search.cross_encoder_metrics import (
    evaluate_scored_candidates,
    expected_gain_scores,
)


def test_expected_gain_score_uses_esci_gains():
    probabilities = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert expected_gain_scores(probabilities).tolist() == [1.0, 0.1, 0.01, 0.0]


def test_perfect_ranking_has_perfect_ndcg_and_pairwise_accuracy():
    frame = pd.DataFrame(
        {
            "query_id": [1, 1, 1, 1],
            "product_id": ["e", "s", "c", "i"],
            "esci_label": ["E", "S", "C", "I"],
            "score": [4.0, 3.0, 2.0, 1.0],
        }
    )
    summary, per_query = evaluate_scored_candidates(frame, score_column="score")
    assert summary["ndcg@10"] == pytest.approx(1.0)
    assert summary["mrr_exact"] == pytest.approx(1.0)
    assert summary["exact_recall@5"] == pytest.approx(1.0)
    assert summary["pairwise_accuracy"] == pytest.approx(1.0)
    assert len(per_query) == 1


def test_worse_ranking_lowers_metrics():
    frame = pd.DataFrame(
        {
            "query_id": [1, 1, 1, 1],
            "product_id": ["e", "s", "c", "i"],
            "esci_label": ["E", "S", "C", "I"],
            "score": [1.0, 2.0, 3.0, 4.0],
        }
    )
    summary, _ = evaluate_scored_candidates(frame, score_column="score")
    assert summary["ndcg@10"] < 1.0
    assert summary["mrr_exact"] == pytest.approx(0.25)
    assert summary["pairwise_accuracy"] == pytest.approx(0.0)
