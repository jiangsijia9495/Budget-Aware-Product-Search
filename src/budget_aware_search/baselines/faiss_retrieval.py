from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from budget_aware_search.labels import ESCI_GAINS
from budget_aware_search.metrics import dcg_at_k
from budget_aware_search.utils import write_json


def _imports():
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install ML dependencies with: pip install -e '.[ml]'") from exc
    return faiss, SentenceTransformer


def _ndcg_from_gains(ranked_gains: list[float], ideal_gains: list[float], k: int) -> float:
    denominator = dcg_at_k(np.sort(np.asarray(ideal_gains))[::-1], k)
    return dcg_at_k(np.asarray(ranked_gains), k) / denominator if denominator > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full-catalog dense retrieval with incomplete-judgment diagnostics."
    )
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--catalog-embeddings", required=True)
    parser.add_argument("--catalog-ids", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--index", choices=["flat", "hnsw"], default="flat")
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-search", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/full_catalog_retrieval")
    args = parser.parse_args()

    faiss, SentenceTransformer = _imports()
    judgments = pd.read_parquet(args.judgments)
    queries = judgments[["query_id", "query"]].drop_duplicates("query_id").reset_index(drop=True)
    catalog_embeddings = np.asarray(np.load(args.catalog_embeddings, mmap_mode="r"), dtype=np.float32)
    catalog_ids = np.load(args.catalog_ids, allow_pickle=False).astype(str)
    if len(catalog_ids) != catalog_embeddings.shape[0]:
        raise ValueError("Catalog ids and embeddings have different row counts")

    dimension = catalog_embeddings.shape[1]
    if args.index == "flat":
        index = faiss.IndexFlatIP(dimension)
    else:
        index = faiss.IndexHNSWFlat(dimension, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = args.ef_search
    build_start = time.perf_counter()
    index.add(catalog_embeddings)
    build_seconds = time.perf_counter() - build_start

    model = SentenceTransformer(args.model, device=args.device)
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
    search_start = time.perf_counter()
    distances, indices = index.search(query_embeddings, args.top_k)
    search_seconds = time.perf_counter() - search_start

    judgments_by_query: dict[object, dict[str, str]] = {}
    for query_id, group in judgments.groupby("query_id", sort=False):
        judgments_by_query[query_id] = dict(
            zip(group["product_id"].astype(str), group["esci_label"].astype(str))
        )

    metric_rows: list[dict[str, float | int | object]] = []
    result_rows: list[dict[str, float | int | object]] = []
    for query_row, query_id in enumerate(queries["query_id"].tolist()):
        judged = judgments_by_query[query_id]
        retrieved_ids = catalog_ids[indices[query_row]]
        retrieved_labels = [judged.get(pid) for pid in retrieved_ids]
        retrieved_gains = [ESCI_GAINS.get(label, 0.0) for label in retrieved_labels]
        known_exact = {pid for pid, label in judged.items() if label == "E"}
        retrieved_exact = sum(pid in known_exact for pid in retrieved_ids)
        condensed_gains = [ESCI_GAINS[label] for label in retrieved_labels if label is not None]
        ideal_gains = [ESCI_GAINS[label] for label in judged.values()]
        metric_rows.append(
            {
                "query_id": query_id,
                "known_exact_recall@k": (
                    retrieved_exact / len(known_exact) if known_exact else 0.0
                ),
                "known_exact_success@k": float(retrieved_exact > 0),
                "judged_fraction@k": float(np.mean([label is not None for label in retrieved_labels])),
                "lower_bound_ndcg@k": _ndcg_from_gains(
                    retrieved_gains, ideal_gains, args.top_k
                ),
                "condensed_ndcg@k": _ndcg_from_gains(
                    condensed_gains, ideal_gains, args.top_k
                ),
            }
        )
        for rank, (product_id, score, label) in enumerate(
            zip(retrieved_ids, distances[query_row], retrieved_labels), start=1
        ):
            result_rows.append(
                {
                    "query_id": query_id,
                    "rank": rank,
                    "product_id": product_id,
                    "score": float(score),
                    "esci_label_if_judged": label,
                }
            )

    metrics = pd.DataFrame(metric_rows)
    results = pd.DataFrame(result_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(output_dir / "per_query_metrics.parquet", index=False)
    results.to_parquet(output_dir / "retrieved_products.parquet", index=False)
    summary = {
        "warning": (
            "ESCI judgments cover only the original candidate pool. Unjudged catalog products are "
            "unknown, not necessarily irrelevant. known_exact_recall and lower_bound_ndcg should "
            "therefore be interpreted as incomplete-judgment diagnostics, not official retrieval metrics."
        ),
        "index": args.index,
        "catalog_products": int(len(catalog_ids)),
        "queries": int(len(queries)),
        "top_k": args.top_k,
        "build_seconds": build_seconds,
        "search_seconds": search_seconds,
        "queries_per_second": float(len(queries) / max(search_seconds, 1e-12)),
        "metrics": {
            column: float(metrics[column].mean())
            for column in metrics.columns
            if column != "query_id"
        },
    }
    write_json(summary, output_dir / "summary.json")
    print(f"Searched {len(queries):,} queries over {len(catalog_ids):,} products")
    print(f"Known-exact recall@{args.top_k}: {summary['metrics']['known_exact_recall@k']:.6f}")
    print(f"Judged fraction@{args.top_k}: {summary['metrics']['judged_fraction@k']:.6f}")


if __name__ == "__main__":
    main()
