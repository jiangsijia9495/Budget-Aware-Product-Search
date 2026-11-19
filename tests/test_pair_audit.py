import pandas as pd

from budget_aware_search.training.audit_pairs import audit_pairs


def test_pair_audit_passes_valid_rows():
    frame = pd.DataFrame(
        {
            "query_id": [1],
            "anchor": ["red shoe"],
            "positive": ["red running shoe"],
            "negative": ["coffee mug"],
            "positive_product_id": ["a"],
            "negative_product_id": ["b"],
            "positive_label": ["E"],
            "negative_label": ["I"],
        }
    )
    result = audit_pairs(frame)
    assert result["passed"] is True
    assert result["same_product_rows"] == 0


def test_pair_audit_rejects_equal_gain_negative():
    frame = pd.DataFrame(
        {
            "query_id": [1],
            "anchor": ["red shoe"],
            "positive": ["red running shoe"],
            "negative": ["another red running shoe"],
            "positive_product_id": ["a"],
            "negative_product_id": ["b"],
            "positive_label": ["S"],
            "negative_label": ["S"],
        }
    )
    result = audit_pairs(frame)
    assert result["passed"] is False
    assert result["negative_gain_ge_positive_gain_rows"] == 1
