"""Benchmark real single-query GPU forward latencies for an adaptive reranker.

Measures first-stage query embedding + cosine scores against *precomputed*
candidate embeddings, and second-stage Cross-Encoder scoring of ALL candidates
for that query in one batch. GPU synchronization surrounds every timed stage.
Excludes catalog embedding build, checkpoint load, network and FAISS retrieval.

The output is used by adaptive_reranking.py for *measured-stage replay* across
policies. The replay is not a direct online service benchmark.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def _ms() -> float:
    return time.perf_counter() * 1000.0


def run_benchmark(
    scored: pd.DataFrame,
    bi_model_path: str,
    ce_model_path: str,
    *,
    device: str = "cuda",
    max_queries: int = 200,
    repeats: int = 2,
    warmups: int = 4,
    seed: int = 42,
    catalog_batch_size: int = 256,
    ce_max_length: int = 256,
) -> tuple[pd.DataFrame, dict]:
    try:
        import torch
        from sentence_transformers import CrossEncoder, SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Requires sentence-transformers and torch in your budget environment") from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if max_queries < 1 or repeats < 1 or warmups < 0:
        raise ValueError("max_queries, repeats must be >=1; warmups >=0")
    required = {"query_id", "query", "product_id", "product_text"}
    if not required.issubset(scored):
        raise ValueError(f"Missing scored validation columns: {sorted(required - set(scored.columns))}")

    ids = scored["query_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    chosen_ids = ids[rng.permutation(len(ids))[:min(max_queries, len(ids))]]
    chosen_set = set(chosen_ids.tolist())
    subset = scored.loc[scored["query_id"].isin(chosen_set)].copy()
    groups = [group for _, group in subset.groupby("query_id", sort=False)]
    if not groups:
        raise ValueError("No candidate groups selected")

    # Model and catalog embedding initialization must be OUTSIDE timed path.
    bi = SentenceTransformer(bi_model_path, device=device)
    ce = CrossEncoder(ce_model_path, device=device, max_length=ce_max_length)
    catalog = subset["product_text"].fillna("").astype(str).drop_duplicates().tolist()
    product_embedding = bi.encode(
        catalog, batch_size=catalog_batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    )
    embedding_lookup = {text: idx for idx, text in enumerate(catalog)}
    _synchronize(device)

    def one_query(group: pd.DataFrame) -> tuple[float, float]:
        q = str(group["query"].iloc[0])
        products = group["product_text"].fillna("").astype(str).tolist()
        candidate_emb = product_embedding[[embedding_lookup[t] for t in products]]

        _synchronize(device)
        start = _ms()
        query_emb = bi.encode(
            [q], batch_size=1, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )[0]
        scores = candidate_emb @ query_emb
        # Include difficulty feature overhead in first-stage latency.
        if len(scores) >= 2:
            top2 = np.partition(scores, -2)[-2:]
            _ = float(np.max(top2) - np.min(top2))
        _synchronize(device)
        bi_ms = _ms() - start

        pair_inputs = [(q, text) for text in products]
        _synchronize(device)
        start = _ms()
        ce.predict(
            pair_inputs,
            batch_size=max(1, len(pair_inputs)),
            show_progress_bar=False,
            apply_softmax=False,
            convert_to_numpy=True,
        )
        _synchronize(device)
        ce_ms = _ms() - start
        return bi_ms, ce_ms

    # Explicit warm-up, not included in results.
    warm_group = groups[0]
    for _ in range(warmups):
        one_query(warm_group)

    records = []
    for repeat in range(repeats):
        random_order = rng.permutation(len(groups))
        for i in random_order:
            group = groups[int(i)]
            bi_ms, ce_ms = one_query(group)
            records.append(
                {
                    "query_id": group["query_id"].iloc[0],
                    "repeat": repeat,
                    "candidate_count": len(group),
                    "bi_ms": bi_ms,
                    "ce_ms": ce_ms,
                }
            )
    result = pd.DataFrame(records)
    summary = {
        "benchmark_queries": int(len(groups)),
        "measurements": int(len(result)),
        "repeats": int(repeats),
        "warmups": int(warmups),
        "device": device,
        "gpu": str(torch.cuda.get_device_name(0)) if device.startswith("cuda") else None,
        "biencoder_model": bi_model_path,
        "crossencoder_model": ce_model_path,
        "p50_bi_ms": float(result["bi_ms"].quantile(0.50)),
        "p95_bi_ms": float(result["bi_ms"].quantile(0.95)),
        "p50_ce_ms": float(result["ce_ms"].quantile(0.50)),
        "p95_ce_ms": float(result["ce_ms"].quantile(0.95)),
        "scope": (
            "Per-query serial processing. Candidate embeddings precomputed. "
            "Bi time includes one query encoding, candidate dot products, and margin; "
            "CE time includes tokenization and one batch for all candidates. "
            "Excludes startup, candidate retrieval/FAISS, network and model loading. "
            "Not QPS under concurrent serving."
        ),
    }
    return result, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-val", default="outputs/biencoder_v1_random_val.parquet")
    parser.add_argument("--bi-model", default="models/biencoder_v1_random/final")
    parser.add_argument("--ce-model", default="models/cross_encoder_esci_v1/final")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--catalog-batch-size", type=int, default=256)
    parser.add_argument("--ce-max-length", type=int, default=256)
    parser.add_argument("--output", default="outputs/adaptive_reranking/latency_samples.parquet")
    args = parser.parse_args()
    data = pd.read_parquet(args.scored_val)
    records, summary = run_benchmark(
        data, args.bi_model, args.ce_model,
        device=args.device,
        max_queries=args.max_queries,
        repeats=args.repeats,
        warmups=args.warmups,
        seed=args.seed,
        catalog_batch_size=args.catalog_batch_size,
        ce_max_length=args.ce_max_length,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records.to_parquet(output, index=False)
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {len(records)} measurements: {output}")


if __name__ == "__main__":
    main()
