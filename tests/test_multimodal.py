import numpy as np
import pandas as pd

from budget_aware_search.multimodal.fuse_sqid import add_sqid_features


def test_add_sqid_features_computes_cosine_scores():
    predictions = pd.DataFrame(
        {
            "query_id": [1],
            "product_id": ["p1"],
            "score": [0.5],
            "esci_label": ["E"],
        }
    )
    product = pd.DataFrame(
        {
            "product_id": ["p1"],
            "clip_text_features": [np.array([1.0, 0.0])],
            "clip_image_features": [np.array([0.0, 1.0])],
        }
    )
    query = pd.DataFrame(
        {
            "query_id": ["1"],
            "clip_text_features": [np.array([1.0, 0.0])],
        }
    )
    output, coverage = add_sqid_features(predictions, product, query)
    assert output.iloc[0]["clip_text_score"] == 1.0
    assert output.iloc[0]["clip_image_score"] == 0.0
    assert coverage["row_vector_coverage"] == 1.0
