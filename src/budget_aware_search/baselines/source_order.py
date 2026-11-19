from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from budget_aware_search.metrics import evaluate_predictions
from budget_aware_search.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the source candidate order baseline.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    if "source_rank" not in frame.columns:
        raise ValueError("Input must contain source_rank")
    frame = frame.copy()
    frame["score"] = -frame["source_rank"].astype(float)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    summary, per_query = evaluate_predictions(
        frame, bootstrap_samples=args.bootstrap_samples
    )
    write_json(summary, output.with_suffix(".metrics.json"))
    per_query.to_parquet(output.with_suffix(".per_query.parquet"), index=False)
    print(f"Source-order nDCG@10: {summary['metrics']['ndcg@10']['mean']:.6f}")


if __name__ == "__main__":
    main()
