from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from budget_aware_search.labels import ESCI_GAINS
from budget_aware_search.metrics import evaluate_predictions
from budget_aware_search.utils import write_json


def _stack_vectors(series: pd.Series, name: str) -> np.ndarray:
    try:
        matrix = np.stack([np.asarray(value, dtype=np.float32) for value in series])
    except Exception as exc:
        raise ValueError(f"Could not stack vector column {name!r}") from exc
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D vectors in {name!r}, got shape {matrix.shape}")
    return matrix


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def add_sqid_features(
    predictions: pd.DataFrame,
    product_features: pd.DataFrame,
    query_features: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    required_product = {"product_id", "clip_text_features", "clip_image_features"}
    required_query = {"query_id", "clip_text_features"}
    if missing := required_product.difference(product_features.columns):
        raise ValueError(f"SQID product features missing columns: {sorted(missing)}")
    if missing := required_query.difference(query_features.columns):
        raise ValueError(f"SQID query features missing columns: {sorted(missing)}")

    product_features = product_features.drop_duplicates("product_id").reset_index(drop=True)
    query_features = query_features.drop_duplicates("query_id").reset_index(drop=True)
    product_ids = product_features["product_id"].astype(str).to_numpy()
    query_ids = query_features["query_id"].astype(str).to_numpy()
    product_text = _normalize_rows(
        _stack_vectors(product_features["clip_text_features"], "clip_text_features")
    )
    product_image = _normalize_rows(
        _stack_vectors(product_features["clip_image_features"], "clip_image_features")
    )
    query_text = _normalize_rows(
        _stack_vectors(query_features["clip_text_features"], "clip_text_features")
    )
    if not (product_text.shape[1] == product_image.shape[1] == query_text.shape[1]):
        raise ValueError(
            "SQID query/product embedding dimensions do not agree: "
            f"{query_text.shape[1]}, {product_text.shape[1]}, {product_image.shape[1]}"
        )

    product_to_row = {pid: idx for idx, pid in enumerate(product_ids)}
    query_to_row = {qid: idx for idx, qid in enumerate(query_ids)}
    output = predictions.copy()
    product_rows = output["product_id"].astype(str).map(product_to_row)
    query_rows = output["query_id"].astype(str).map(query_to_row)
    valid = product_rows.notna() & query_rows.notna()

    output["clip_text_score"] = np.nan
    output["clip_image_score"] = np.nan
    output["product_text_image_agreement"] = np.nan
    p_rows = product_rows.loc[valid].astype(int).to_numpy()
    q_rows = query_rows.loc[valid].astype(int).to_numpy()
    valid_index = output.index[valid]
    output.loc[valid_index, "clip_text_score"] = np.einsum(
        "ij,ij->i", query_text[q_rows], product_text[p_rows]
    )
    output.loc[valid_index, "clip_image_score"] = np.einsum(
        "ij,ij->i", query_text[q_rows], product_image[p_rows]
    )
    output.loc[valid_index, "product_text_image_agreement"] = np.einsum(
        "ij,ij->i", product_text[p_rows], product_image[p_rows]
    )
    coverage = {
        "row_vector_coverage": float(valid.mean()),
        "query_vector_coverage": float(output.loc[valid, "query_id"].nunique() / output["query_id"].nunique()),
        "product_vector_coverage": float(output.loc[valid, "product_id"].nunique() / output["product_id"].nunique()),
    }
    return output, coverage


def query_split(
    frame: pd.DataFrame,
    seed: int,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> pd.Series:
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be less than 1")
    query_ids = frame["query_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(query_ids)
    n_train = int(round(len(query_ids) * train_fraction))
    n_val = int(round(len(query_ids) * validation_fraction))
    train_ids = set(query_ids[:n_train].tolist())
    val_ids = set(query_ids[n_train : n_train + n_val].tolist())
    return frame["query_id"].map(
        lambda query_id: "train" if query_id in train_ids else ("val" if query_id in val_ids else "test")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a custom query-disjoint SQID feature-fusion ranker."
    )
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--product-features", required=True)
    parser.add_argument("--query-features", required=True)
    parser.add_argument("--output-dir", default="outputs/sqid_fusion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args()

    base = pd.read_parquet(args.base_predictions).copy()
    if "score" not in base.columns:
        raise ValueError("Base predictions must contain a score column")
    base = base.rename(columns={"score": "base_score"})
    product_features = pd.read_parquet(args.product_features)
    query_features = pd.read_parquet(args.query_features)
    enriched, coverage = add_sqid_features(base, product_features, query_features)
    enriched["custom_split"] = query_split(enriched, seed=args.seed)
    enriched["target_gain"] = enriched["esci_label"].map(ESCI_GAINS).astype(float)

    feature_columns = [
        "base_score",
        "clip_text_score",
        "clip_image_score",
        "product_text_image_agreement",
    ]
    train = enriched.loc[enriched["custom_split"] == "train"]
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=args.max_iter,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=args.seed,
    )
    model.fit(train[feature_columns], train["target_gain"])
    enriched["score"] = model.predict(enriched[feature_columns]).clip(0.0, 1.0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "fusion_model.joblib")
    enriched.to_parquet(output_dir / "all_predictions.parquet", index=False)

    summary: dict[str, object] = {
        "warning": (
            "SQID mainly covers the official ESCI test queries. This script creates a new "
            "query-disjoint train/validation/test split inside that subset; its metrics are not "
            "directly comparable to the official ESCI test benchmark."
        ),
        "coverage": coverage,
        "features": feature_columns,
        "split_queries": enriched.groupby("custom_split")["query_id"].nunique().to_dict(),
        "metrics": {},
    }
    for split in ("val", "test"):
        split_frame = enriched.loc[enriched["custom_split"] == split].copy()
        split_metrics: dict[str, object] = {}
        for method, column in {
            "base_text": "base_score",
            "clip_text": "clip_text_score",
            "clip_image": "clip_image_score",
            "fusion": "score",
        }.items():
            evaluable = split_frame.copy()
            if evaluable[column].notna().sum() == 0:
                continue
            # Preserve the original candidate set. Missing visual features rank last rather
            # than disappearing from the denominator and inflating nDCG.
            evaluable[column] = evaluable[column].fillna(-1.0e9)
            method_summary, per_query = evaluate_predictions(evaluable, score_col=column)
            split_metrics[method] = method_summary["metrics"]
            per_query.to_parquet(output_dir / f"{split}_{method}_per_query.parquet", index=False)
        summary["metrics"][split] = split_metrics
        split_frame.to_parquet(output_dir / f"{split}_predictions.parquet", index=False)

    write_json(summary, output_dir / "summary.json")
    test_ndcg = summary["metrics"]["test"].get("fusion", {}).get("ndcg@10", {}).get("mean")
    print(f"SQID vector coverage: {coverage['row_vector_coverage']:.2%}")
    if test_ndcg is not None:
        print(f"Custom-split fusion nDCG@10: {test_ndcg:.6f}")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
