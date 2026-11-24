import numpy as np
import pandas as pd

from budget_aware_search.routing.phase15_final_test import audit_ce_score_repeatability


def _frames():
    routed = pd.DataFrame({
        "query_id": [1, 1, 1, 2, 2],
        "product_id": ["a", "b", "c", "x", "y"],
        "cross_score": [0.92, 0.60, 0.10, 0.8000, 0.2000],
    })
    full = pd.DataFrame({
        "query_id": [2, 1, 1, 2, 1],
        "product_id": ["y", "c", "a", "x", "b"],
        "cross_score": [0.2001, 0.1001, 0.9199, 0.8001, 0.5999],
    })
    return routed, full


def test_small_fp16_like_drift_passes():
    routed, full = _frames()
    stats = audit_ce_score_repeatability(routed, full)
    assert stats["passed_numeric_consistency"]
    assert stats["top1_disagreement_queries"] == 0
    assert stats["top10_set_disagreement_queries"] == 0
    assert stats["max_abs_score_difference"] < 2e-4


def test_large_drift_is_not_silently_ignored():
    routed, full = _frames()
    full.loc[full.product_id.eq("a"), "cross_score"] = 0.70
    stats = audit_ce_score_repeatability(routed, full)
    assert not stats["passed_numeric_consistency"]


def test_missing_routed_candidate_is_rejected():
    routed, full = _frames()
    full = full.loc[~full.product_id.eq("b")]
    try:
        audit_ce_score_repeatability(routed, full)
    except RuntimeError as e:
        assert "missing" in str(e).lower()
    else:
        raise AssertionError("Expected missing-candidate failure")


def test_wrong_candidate_scores_can_be_detected_without_labels():
    routed, full = _frames()
    routed.loc[routed.product_id.eq("x"), "cross_score"] = 0.02
    stats = audit_ce_score_repeatability(routed, full)
    assert stats["top1_disagreement_queries"] >= 1
    assert not stats["passed_numeric_consistency"]
