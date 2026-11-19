from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from budget_aware_search.serving.matrix_io import write_int64_matrix
from budget_aware_search.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize query/product pairs for C++ ONNX benchmarking.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", default="serving_artifacts/onnx_inputs")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=10000)
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install ML dependencies with: pip install -e '.[ml]'") from exc

    frame = pd.read_parquet(args.input).head(args.max_rows)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    encoded = tokenizer(
        frame["query"].astype(str).tolist(),
        frame["product_text"].fillna("").astype(str).tolist(),
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="np",
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    available_inputs = []
    for name in ("input_ids", "attention_mask", "token_type_ids"):
        if name in encoded:
            write_int64_matrix(output_dir / f"{name}.i64mat", encoded[name])
            available_inputs.append(name)
    write_json(
        {
            "rows": int(len(frame)),
            "max_length": args.max_length,
            "tokenizer": args.tokenizer,
            "inputs": available_inputs,
        },
        output_dir / "metadata.json",
    )
    print(f"Prepared {len(frame):,} tokenized pairs in {output_dir}")


if __name__ == "__main__":
    main()
