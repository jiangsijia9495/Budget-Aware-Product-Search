from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from budget_aware_search.labels import validate_esci_labels
from budget_aware_search.text import build_product_text
from budget_aware_search.utils import load_yaml, set_seed, write_json

EXAMPLE_COLUMNS = {
    "example_id",
    "query",
    "query_id",
    "product_id",
    "product_locale",
    "esci_label",
    "small_version",
    "large_version",
    "split",
}
PRODUCT_COLUMNS = {
    "product_id",
    "product_locale",
    "product_title",
    "product_description",
    "product_bullet_point",
    "product_brand",
    "product_color",
}


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def load_and_merge_esci(
    examples_path: str | Path,
    products_path: str | Path,
    locale: str = "us",
    use_small_version: bool = True,
    product_text_fields: list[str] | None = None,
    max_product_text_chars: int = 1600,
) -> pd.DataFrame:
    examples_path = Path(examples_path)
    products_path = Path(products_path)
    if not examples_path.exists():
        raise FileNotFoundError(
            f"Missing examples parquet: {examples_path}. Run scripts/download_esci.sh first."
        )
    if not products_path.exists():
        raise FileNotFoundError(
            f"Missing products parquet: {products_path}. Run scripts/download_esci.sh first."
        )

    examples = pd.read_parquet(examples_path)
    products = pd.read_parquet(products_path)
    _require_columns(examples, EXAMPLE_COLUMNS, "examples parquet")
    _require_columns(products, PRODUCT_COLUMNS, "products parquet")

    version_column = "small_version" if use_small_version else "large_version"
    examples = examples.loc[
        (examples["product_locale"] == locale) & (examples[version_column] == 1)
    ].copy()
    if examples.empty:
        raise ValueError(
            f"No rows remain after filtering locale={locale!r}, {version_column}=1."
        )

    examples["source_rank"] = examples.groupby("query_id", sort=False).cumcount() + 1
    products = products.loc[products["product_locale"] == locale].copy()

    duplicate_product_rows = products.duplicated(["product_locale", "product_id"], keep=False)
    if duplicate_product_rows.any():
        # Product metadata should be one row per locale/product. Keep the most complete row.
        metadata_fields = [
            "product_title",
            "product_description",
            "product_bullet_point",
            "product_brand",
            "product_color",
        ]
        products["_non_null_fields"] = products[metadata_fields].notna().sum(axis=1)
        products = (
            products.sort_values("_non_null_fields", ascending=False)
            .drop_duplicates(["product_locale", "product_id"], keep="first")
            .drop(columns="_non_null_fields")
        )

    merged = examples.merge(
        products,
        how="left",
        on=["product_locale", "product_id"],
        validate="many_to_one",
    )
    validate_esci_labels(set(merged["esci_label"].dropna().unique()))

    if merged["product_title"].isna().mean() > 0.01:
        missing_rate = merged["product_title"].isna().mean()
        raise ValueError(
            f"Unexpectedly high missing product-title rate after merge: {missing_rate:.2%}. "
            "Check that the examples and products files come from the same ESCI release."
        )

    fields = product_text_fields or [
        "product_title",
        "product_brand",
        "product_color",
        "product_bullet_point",
        "product_description",
    ]
    merged["product_text"] = merged.apply(
        lambda row: build_product_text(row, fields, max_product_text_chars), axis=1
    )
    merged["query"] = merged["query"].fillna("").astype(str).str.strip()
    merged["product_id"] = merged["product_id"].astype(str)
    return merged


def split_official_train_by_query(
    merged: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")

    official_train = merged.loc[merged["split"] == "train"].copy()
    official_test = merged.loc[merged["split"] == "test"].copy()
    if official_train.empty or official_test.empty:
        raise ValueError("Expected both official train and test rows in the ESCI data.")

    query_ids = official_train["query_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(query_ids)
    n_val = max(1, int(round(len(query_ids) * validation_fraction)))
    val_ids = set(query_ids[:n_val].tolist())

    train = official_train.loc[~official_train["query_id"].isin(val_ids)].copy()
    val = official_train.loc[official_train["query_id"].isin(val_ids)].copy()
    test = official_test.copy()

    _assert_query_disjoint(train, val, test)
    return train, val, test


def _assert_query_disjoint(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    train_ids = set(train["query_id"].unique())
    val_ids = set(val["query_id"].unique())
    test_ids = set(test["query_id"].unique())
    overlaps = {
        "train_val": len(train_ids.intersection(val_ids)),
        "train_test": len(train_ids.intersection(test_ids)),
        "val_test": len(val_ids.intersection(test_ids)),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Query leakage detected: {overlaps}")


def _sample_queries(df: pd.DataFrame, max_queries: int | None, seed: int) -> pd.DataFrame:
    if max_queries is None:
        return df
    unique_ids = df["query_id"].drop_duplicates().to_numpy()
    if len(unique_ids) <= max_queries:
        return df
    rng = np.random.default_rng(seed)
    selected = set(rng.choice(unique_ids, size=max_queries, replace=False).tolist())
    return df.loc[df["query_id"].isin(selected)].copy()


def _summary(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "queries": int(df["query_id"].nunique()),
        "products": int(df["product_id"].nunique()),
        "avg_candidates_per_query": float(df.groupby("query_id").size().mean()),
        "label_counts": {
            str(k): int(v) for k, v in df["esci_label"].value_counts().sort_index().items()
        },
        "missing_product_text_rate": float(df["product_text"].eq("").mean()),
    }


def prepare_dataset(
    config: dict[str, Any],
    output_dir: str | Path = "data/processed",
    debug_max_queries: int | None = None,
) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    data_cfg = config["data"]

    merged = load_and_merge_esci(
        examples_path=data_cfg["examples_path"],
        products_path=data_cfg["products_path"],
        locale=data_cfg.get("locale", "us"),
        use_small_version=bool(data_cfg.get("use_small_version", True)),
        product_text_fields=list(data_cfg.get("product_text_fields", [])) or None,
        max_product_text_chars=int(data_cfg.get("max_product_text_chars", 1600)),
    )
    train, val, test = split_official_train_by_query(
        merged,
        validation_fraction=float(data_cfg.get("validation_fraction", 0.15)),
        seed=seed,
    )

    train = _sample_queries(train, debug_max_queries, seed)
    val = _sample_queries(val, debug_max_queries, seed + 1)
    test = _sample_queries(test, debug_max_queries, seed + 2)
    _assert_query_disjoint(train, val, test)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keep_columns = [
        "example_id",
        "query_id",
        "query",
        "product_id",
        "product_locale",
        "esci_label",
        "source_rank",
        "product_title",
        "product_brand",
        "product_color",
        "product_bullet_point",
        "product_description",
        "product_text",
    ]

    for split_name, frame in (("train", train), ("val", val), ("test", test)):
        frame = frame[keep_columns].copy()
        frame.to_parquet(output_dir / f"{split_name}.parquet", index=False)

    # The product catalog is assumed known at serving time. It contains no query labels.
    catalog = (
        merged[["product_id", "product_locale", "product_text"]]
        .drop_duplicates("product_id", keep="first")
        .reset_index(drop=True)
    )
    catalog.to_parquet(output_dir / "catalog.parquet", index=False)

    summary = {
        "configuration": {
            "locale": data_cfg.get("locale", "us"),
            "use_small_version": bool(data_cfg.get("use_small_version", True)),
            "validation_fraction": float(data_cfg.get("validation_fraction", 0.15)),
            "seed": seed,
            "debug_max_queries": debug_max_queries,
        },
        "train": _summary(train),
        "val": _summary(val),
        "test": _summary(test),
        "catalog_products": int(catalog["product_id"].nunique()),
    }
    write_json(summary, output_dir / "dataset_summary.json")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare query-disjoint ESCI splits.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument(
        "--debug-max-queries",
        type=int,
        default=None,
        help="Keep at most this many queries in each split for a quick smoke test.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_yaml(args.config)
    summary = prepare_dataset(config, args.output_dir, args.debug_max_queries)
    print("Prepared ESCI data successfully:")
    for split in ("train", "val", "test"):
        info = summary[split]
        print(
            f"  {split:>5}: {info['queries']:,} queries, "
            f"{info['rows']:,} query-product rows"
        )
    print(f"  catalog: {summary['catalog_products']:,} unique products")


if __name__ == "__main__":
    main()
