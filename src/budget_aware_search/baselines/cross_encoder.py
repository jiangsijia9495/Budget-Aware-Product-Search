from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from budget_aware_search.cross_encoder_metrics import (
    ESCI_GAINS,
    LABEL_ORDER,
    bootstrap_metric_intervals,
    evaluate_scored_candidates,
    expected_gain_scores,
)


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _output_sidecar(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score an ESCI candidate set with a scalar or four-class Cross-Encoder."
    )
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        import torch
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "Missing Cross-Encoder dependencies. Run: pip install -e '.[ml,dev]'"
        ) from exc

    input_path = Path(args.processed_dir) / f"{args.split}.parquet"
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    frame = pd.read_parquet(input_path)
    required = {"query_id", "query", "product_id", "product_text", "esci_label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")

    if args.max_queries is not None:
        query_ids = frame["query_id"].drop_duplicates().head(args.max_queries)
        frame = frame.loc[frame["query_id"].isin(set(query_ids.tolist()))].copy()

    model_kwargs: dict[str, object] = {}
    if args.device.startswith("cuda") and torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
    model = CrossEncoder(
        args.model,
        device=args.device,
        max_length=args.max_length,
        model_kwargs=model_kwargs,
    )

    pairs = list(zip(frame["query"].astype(str), frame["product_text"].astype(str), strict=True))
    start = time.perf_counter()
    if model.num_labels == 1:
        predictions = model.predict(
            pairs,
            batch_size=args.batch_size,
            show_progress_bar=True,
            activation_fn=torch.nn.Identity(),
            apply_softmax=False,
            convert_to_numpy=True,
        )
        scores = np.asarray(predictions, dtype=np.float64).reshape(-1)
        frame["cross_score"] = scores
        score_definition = "raw scalar Cross-Encoder logit"
    elif model.num_labels == len(LABEL_ORDER):
        probabilities = model.predict(
            pairs,
            batch_size=args.batch_size,
            show_progress_bar=True,
            apply_softmax=True,
            convert_to_numpy=True,
        )
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.shape != (len(frame), len(LABEL_ORDER)):
            raise ValueError(
                f"Expected probability matrix {(len(frame), len(LABEL_ORDER))}, "
                f"got {probabilities.shape}"
            )
        for index, label in enumerate(LABEL_ORDER):
            frame[f"prob_{label}"] = probabilities[:, index]
        frame["cross_score"] = expected_gain_scores(probabilities)
        score_definition = "P(E) + 0.1*P(S) + 0.01*P(C)"
    else:
        raise ValueError(
            f"Expected model.num_labels to be 1 or {len(LABEL_ORDER)}, got {model.num_labels}"
        )
    inference_seconds = time.perf_counter() - start

    summary, per_query = evaluate_scored_candidates(frame, score_column="cross_score")
    bootstrap = bootstrap_metric_intervals(
        per_query,
        columns=["ndcg@10", "mrr_exact", "exact_recall@5", "pairwise_accuracy"],
        samples=args.bootstrap_samples,
        seed=args.seed,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    per_query_path = _output_sidecar(output, ".per_query.parquet")
    per_query.to_parquet(per_query_path, index=False)
    metrics_path = _output_sidecar(output, ".metrics.json")
    metrics = {
        **summary,
        "bootstrap": bootstrap,
        "model": args.model,
        "num_labels": int(model.num_labels),
        "score_definition": score_definition,
        "gain_mapping": ESCI_GAINS,
        "rows": int(len(frame)),
        "inference_seconds": inference_seconds,
        "pairs_per_second": float(len(frame) / max(inference_seconds, 1e-9)),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
    }
    _write_json(metrics, metrics_path)

    print(json.dumps(metrics, indent=2))
    print(f"Saved predictions to {output}")
    print(f"Saved per-query metrics to {per_query_path}")


if __name__ == "__main__":
    main()
