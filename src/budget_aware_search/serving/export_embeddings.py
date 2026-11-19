from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from budget_aware_search.serving.matrix_io import write_float32_matrix
from budget_aware_search.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Export normalized embeddings for C++ FAISS benchmarks.")
    parser.add_argument("--catalog", default="data/processed/catalog.parquet")
    parser.add_argument("--queries", default="data/processed/val.parquet")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", default="serving_artifacts/embeddings")
    parser.add_argument("--max-products", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install ML dependencies with: pip install -e '.[ml]'") from exc

    catalog = pd.read_parquet(args.catalog).drop_duplicates("product_id")
    queries = pd.read_parquet(args.queries)[["query_id", "query"]].drop_duplicates("query_id")
    if args.max_products:
        catalog = catalog.head(args.max_products)
    if args.max_queries:
        queries = queries.head(args.max_queries)

    model = SentenceTransformer(args.model, device=args.device)
    product_embeddings = np.asarray(
        model.encode(
            catalog["product_text"].fillna("").astype(str).tolist(),
            batch_size=args.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )
    query_embeddings = np.asarray(
        model.encode(
            queries["query"].astype(str).tolist(),
            batch_size=args.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_float32_matrix(output_dir / "catalog.f32mat", product_embeddings)
    write_float32_matrix(output_dir / "queries.f32mat", query_embeddings)
    catalog["product_id"].astype(str).to_csv(
        output_dir / "catalog_ids.txt", index=False, header=False
    )
    queries["query_id"].astype(str).to_csv(
        output_dir / "query_ids.txt", index=False, header=False
    )
    write_json(
        {
            "model": args.model,
            "catalog_rows": int(len(catalog)),
            "query_rows": int(len(queries)),
            "dimension": int(product_embeddings.shape[1]),
            "normalized": True,
            "format": "4-byte magic + uint64 rows + uint64 columns + row-major values",
        },
        output_dir / "metadata.json",
    )
    print(f"Exported C++ benchmark matrices to {output_dir}")


if __name__ == "__main__":
    main()
