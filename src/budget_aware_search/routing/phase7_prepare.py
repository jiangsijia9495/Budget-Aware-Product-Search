"""Create query-disjoint shadow-ranker training and router-label holdout sets.

Only the official TRAIN split is partitioned. Validation and test are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _hash_query_ids(ids: list[str]) -> str:
    data = "\n".join(sorted(ids)).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def split_shadow_data(
    train: pd.DataFrame,
    v1_pairs: pd.DataFrame,
    *,
    holdout_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    for name, table, required in (
        ("train", train, {"query_id", "query", "product_id", "product_text", "esci_label"}),
        ("v1_pairs", v1_pairs, {"query_id", "anchor", "positive", "negative"}),
    ):
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"{name} missing {sorted(missing)}")
    if not (0.05 <= holdout_fraction <= 0.5):
        raise ValueError("holdout_fraction must be within [0.05, 0.5]")
    ids = train["query_id"].drop_duplicates().to_numpy()
    if len(ids) < 50:
        raise ValueError("Need >=50 queries for a meaningful shadow holdout")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_holdout = int(round(len(ids) * holdout_fraction))
    holdout_ids = set(ids[:n_holdout].tolist())
    fit_ids = set(ids[n_holdout:].tolist())
    shadow_train = train.loc[train.query_id.isin(fit_ids)].copy()
    holdout = train.loc[train.query_id.isin(holdout_ids)].copy()
    shadow_pairs = v1_pairs.loc[v1_pairs.query_id.isin(fit_ids)].copy()
    if not shadow_pairs.query_id.isin(holdout_ids).eq(False).all():
        raise AssertionError("Pair leakage into shadow holdout")
    if not len(shadow_train) or not len(holdout) or not len(shadow_pairs):
        raise ValueError("Empty split or shadow pair dataset")
    if not set(v1_pairs.query_id.unique()).issubset(set(train.query_id.unique())):
        raise ValueError("v1 pair query IDs are not entirely in train split")
    manifest = {
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "shadow_train_queries": len(fit_ids),
        "router_label_queries": len(holdout_ids),
        "shadow_train_rows": len(shadow_train),
        "router_label_rows": len(holdout),
        "shadow_train_pair_rows": len(shadow_pairs),
        "shadow_train_ids_sha256": _hash_query_ids([str(q) for q in fit_ids]),
        "router_label_ids_sha256": _hash_query_ids([str(q) for q in holdout_ids]),
        "query_disjoint": not bool(fit_ids.intersection(holdout_ids)),
        "warning": "Models for router-label predictions must be fitted only on shadow_train. Do not substitute full-data champion predictions for router supervision.",
    }
    return shadow_train, holdout, shadow_pairs, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/processed/train.parquet")
    parser.add_argument("--v1-pairs", default="data/processed/biencoder_pairs_v1.parquet")
    parser.add_argument("--output-dir", default="data/processed/phase7")
    parser.add_argument("--holdout-fraction", type=float, default=.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    shadow_train, holdout, shadow_pairs, manifest = split_shadow_data(
        pd.read_parquet(args.train), pd.read_parquet(args.v1_pairs),
        holdout_fraction=args.holdout_fraction, seed=args.seed,
    )
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    shadow_train.to_parquet(root / "shadow_train.parquet", index=False)
    holdout.to_parquet(root / "router_holdout.parquet", index=False)
    shadow_pairs.to_parquet(root / "shadow_pairs.parquet", index=False)
    # Existing dense and CE scoring CLIs accept --processed-dir/--split,
    # so this separate directory contains only the held-out scoring pool.
    scoring = root / "router_scoring"
    scoring.mkdir(parents=True, exist_ok=True)
    holdout.to_parquet(scoring / "train.parquet", index=False)
    # The dense scorer loads both <split>.parquet AND catalog.parquet
    # from --processed-dir. Reuse the official catalog without copying it.
    catalog = Path(args.train).resolve().parent / "catalog.parquet"
    if not catalog.is_file():
        raise FileNotFoundError(
            f"Missing {catalog}; Phase 7 must reuse the original dense-scoring catalog"
        )
    catalog_link = scoring / "catalog.parquet"
    if not catalog_link.exists():
        if catalog_link.is_symlink():
            catalog_link.unlink()  # repair a broken link
        catalog_link.symlink_to(os.path.relpath(catalog, scoring.resolve()))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
