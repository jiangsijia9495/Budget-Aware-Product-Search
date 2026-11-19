import pandas as pd

from budget_aware_search.data import split_official_train_by_query


def test_query_level_split_has_no_leakage():
    rows = []
    for query_id in range(20):
        split = "train" if query_id < 15 else "test"
        for product_idx in range(3):
            rows.append(
                {
                    "query_id": query_id,
                    "product_id": f"p-{query_id}-{product_idx}",
                    "split": split,
                }
            )
    frame = pd.DataFrame(rows)
    train, val, test = split_official_train_by_query(frame, 0.2, seed=7)
    assert set(train.query_id).isdisjoint(set(val.query_id))
    assert set(train.query_id).isdisjoint(set(test.query_id))
    assert set(val.query_id).isdisjoint(set(test.query_id))
