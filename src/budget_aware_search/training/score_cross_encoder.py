from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from budget_aware_search.labels import CLASS_GAINS, ID_TO_CLASS
from budget_aware_search.metrics import evaluate_predictions
from budget_aware_search.utils import write_json


def _load_model(path: str, device: str):
    try:
        from sentence_transformers.cross_encoder import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required. Install with: pip install -e '.[ml]'"
        ) from exc
    return CrossEncoder(path, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score ESCI candidates with a cross-encoder.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    model = _load_model(args.model, args.device)
    pairs = list(
        zip(
            frame["query"].astype(str).tolist(),
            frame["product_text"].fillna("").astype(str).tolist(),
        )
    )
    logits = np.asarray(
        model.predict(
            pairs,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
    )
    if logits.ndim != 2 or logits.shape[1] != 4:
        raise ValueError(f"Expected logits of shape [N, 4], got {logits.shape}")
    probabilities = softmax(logits, axis=1)
    gains = np.asarray(CLASS_GAINS, dtype=np.float32)

    output_frame = frame.copy()
    output_frame["score"] = probabilities @ gains
    for class_id in range(4):
        output_frame[f"p_{ID_TO_CLASS[class_id]}"] = probabilities[:, class_id]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_parquet(output, index=False)
    summary, per_query = evaluate_predictions(
        output_frame, bootstrap_samples=args.bootstrap_samples
    )
    write_json(summary, output.with_suffix(".metrics.json"))
    per_query.to_parquet(output.with_suffix(".per_query.parquet"), index=False)
    print(f"Cross-encoder nDCG@10: {summary['metrics']['ndcg@10']['mean']:.6f}")
    print(f"Saved predictions to {output}")


if __name__ == "__main__":
    main()
