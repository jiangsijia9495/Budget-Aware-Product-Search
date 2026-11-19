import numpy as np

from budget_aware_search.baselines.bm25 import SparseBM25


def test_bm25_prefers_lexically_matching_product():
    model = SparseBM25.fit(
        product_ids=["red", "blue", "mug"],
        documents=[
            "title red waterproof running shoes",
            "title blue cotton shirt",
            "title ceramic coffee mug",
        ],
    )
    scores = model.score_candidates("red running shoe", ["mug", "blue", "red"])
    assert int(np.argmax(scores)) == 2


def test_bm25_save_load_round_trip(tmp_path):
    model = SparseBM25.fit(
        product_ids=["a", "b"],
        documents=["red running shoe", "ceramic mug"],
    )
    model.save(tmp_path)
    loaded = SparseBM25.load(tmp_path)
    original = model.score_candidates("red shoe", ["a", "b"])
    restored = loaded.score_candidates("red shoe", ["a", "b"])
    np.testing.assert_allclose(restored, original)
