import numpy as np
import pandas as pd

from budget_aware_search.evaluation.compare_rankings import (
    compare_per_query,
    paired_bootstrap_delta,
)


def test_paired_bootstrap_positive_delta():
    low, high, probability = paired_bootstrap_delta(
        np.array([0.01, 0.02, 0.03, 0.04]), samples=1000, seed=1
    )
    assert low > 0
    assert high > low
    assert probability == 1.0


def test_compare_per_query_reports_mean_delta():
    baseline = pd.DataFrame(
        {
            "query_id": [1, 2],
            "ndcg@10": [0.5, 0.7],
        }
    )
    candidate = pd.DataFrame(
        {
            "query_id": [1, 2],
            "ndcg@10": [0.6, 0.8],
        }
    )
    result = compare_per_query(
        baseline,
        candidate,
        metrics=("ndcg@10",),
        samples=1000,
        seed=2,
    )
    assert result["queries"] == 2
    assert abs(result["metrics"]["ndcg@10"]["mean_delta"] - 0.1) < 1e-12
