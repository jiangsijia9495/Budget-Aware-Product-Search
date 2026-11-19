from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from budget_aware_search.metrics import evaluate_predictions
from budget_aware_search.utils import write_json


def _require_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required. Install with: pip install -e '.[ml]'"
        ) from exc
    return SentenceTransformer


def _model_cache_name(model_name: str) -> str:
    readable = model_name.replace("/", "__").replace(" ", "_")
    digest = hashlib.sha1(model_name.encode("utf-8")).hexdigest()[:8]
    return f"{readable}__{digest}"


def encode_catalog(
    model,
    catalog: pd.DataFrame,
    batch_size: int,
    normalize_embeddings: bool,
) -> np.ndarray:
    return np.asarray(
        model.encode(
            catalog["product_text"].fillna("").astype(str).tolist(),
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )


def score_candidate_pool(
    model,
    frame: pd.DataFrame,
    catalog_ids: np.ndarray,
    catalog_embeddings: np.ndarray,
    batch_size: int,
    normalize_embeddings: bool,
) -> pd.DataFrame:
    query_table = frame[["query_id", "query"]].drop_duplicates("query_id").reset_index(drop=True)
    query_embeddings = np.asarray(
        model.encode(
            query_table["query"].astype(str).tolist(),
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )
    query_to_row = {
        query_id: index for index, query_id in enumerate(query_table["query_id"].tolist())
    }
    product_to_row = {str(pid): index for index, pid in enumerate(catalog_ids)}

    query_rows = frame["query_id"].map(query_to_row).to_numpy()
    product_rows = frame["product_id"].astype(str).map(product_to_row)
    if product_rows.isna().any():
        missing = frame.loc[product_rows.isna(), "product_id"].astype(str).unique()[:10]
        raise ValueError(f"Products missing from embedding catalog, examples: {missing.tolist()}")
    product_rows_np = product_rows.astype(int).to_numpy()
    scores = np.einsum(
        "ij,ij->i",
        query_embeddings[query_rows],
        catalog_embeddings[product_rows_np],
        optimize=True,
    )
    output = frame.copy()
    output["score"] = scores.astype(np.float32)
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrained dense candidate-pool baseline.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cache-dir", default="models/dense_cache")
    parser.add_argument("--output", default="outputs/dense_val.parquet")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--reencode-catalog", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    SentenceTransformer = _require_sentence_transformers()
    processed_dir = Path(args.processed_dir)
    cache_dir = Path(args.cache_dir) / _model_cache_name(args.model)
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = cache_dir / "catalog_embeddings.npy"
    ids_path = cache_dir / "catalog_product_ids.npy"

    model = SentenceTransformer(args.model, device=args.device)
    catalog = pd.read_parquet(processed_dir / "catalog.parquet")
    normalize = not args.no_normalize

    if args.reencode_catalog or not embeddings_path.exists() or not ids_path.exists():
        print(f"Encoding {len(catalog):,} products with {args.model}...")
        catalog_embeddings = encode_catalog(model, catalog, args.batch_size, normalize)
        catalog_ids = catalog["product_id"].astype(str).to_numpy(dtype=str)
        np.save(embeddings_path, catalog_embeddings)
        np.save(ids_path, catalog_ids)
        write_json(
            {
                "model": args.model,
                "products": int(len(catalog_ids)),
                "dimension": int(catalog_embeddings.shape[1]),
                "normalized": normalize,
            },
            cache_dir / "metadata.json",
        )
    else:
        catalog_embeddings = np.load(embeddings_path, mmap_mode="r")
        catalog_ids = np.load(ids_path, allow_pickle=False)
        expected_ids = catalog["product_id"].astype(str).to_numpy(dtype=str)
        if len(catalog_ids) != len(expected_ids) or not np.array_equal(catalog_ids, expected_ids):
            raise ValueError(
                "Cached catalog embeddings do not match the current catalog. "
                "Use --reencode-catalog or a different --cache-dir."
            )

    frame = pd.read_parquet(processed_dir / f"{args.split}.parquet")
    scored = score_candidate_pool(
        model=model,
        frame=frame,
        catalog_ids=catalog_ids,
        catalog_embeddings=catalog_embeddings,
        batch_size=args.batch_size,
        normalize_embeddings=normalize,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(output, index=False)
    summary, per_query = evaluate_predictions(
        scored, bootstrap_samples=args.bootstrap_samples
    )
    write_json(summary, output.with_suffix(".metrics.json"))
    per_query.to_parquet(output.with_suffix(".per_query.parquet"), index=False)
    print(f"Dense baseline nDCG@10: {summary['metrics']['ndcg@10']['mean']:.6f}")
    print(f"Saved predictions to {output}")


if __name__ == "__main__":
    main()
