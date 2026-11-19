import pandas as pd

from budget_aware_search.training.build_pairs import build_biencoder_pairs


def test_build_pairs_uses_exact_positive_and_irrelevant_negative():
    frame = pd.DataFrame(
        {
            "query_id": [1, 1, 1],
            "query": ["red shoe"] * 3,
            "product_id": ["a", "b", "c"],
            "product_text": ["red shoe", "blue shoe", "coffee mug"],
            "esci_label": ["E", "S", "I"],
        }
    )
    pairs = build_biencoder_pairs(frame, seed=3)
    assert len(pairs) == 1
    assert pairs.iloc[0]["positive_label"] == "E"
    assert pairs.iloc[0]["negative_label"] == "I"


def test_build_pairs_never_uses_substitute_as_v1_negative():
    frame = pd.DataFrame(
        {
            "query_id": [1, 1, 1, 1],
            "query": ["red shoe"] * 4,
            "product_id": ["a", "b", "c", "d"],
            "product_text": ["red shoe", "similar shoe", "shoe accessory", "coffee mug"],
            "esci_label": ["E", "S", "C", "I"],
        }
    )
    pairs = build_biencoder_pairs(frame, seed=3, negatives_per_positive=2)
    assert set(pairs["negative_label"]) == {"I", "C"}


def test_build_pairs_substitute_positive_does_not_use_substitute_negative():
    frame = pd.DataFrame(
        {
            "query_id": [1, 1, 1],
            "query": ["rare product"] * 3,
            "product_id": ["a", "b", "c"],
            "product_text": ["substitute one", "substitute two", "irrelevant"],
            "esci_label": ["S", "S", "I"],
        }
    )
    pairs = build_biencoder_pairs(frame, seed=3)
    assert len(pairs) == 1
    assert pairs.iloc[0]["positive_label"] == "S"
    assert pairs.iloc[0]["negative_label"] == "I"
