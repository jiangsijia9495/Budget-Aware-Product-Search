from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import pandas as pd

from budget_aware_search.cross_encoder_metrics import LABEL_ORDER, LABEL_TO_ID


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a four-class ESCI Cross-Encoder on query-product pairs."
    )
    parser.add_argument("--train", default="data/processed/train.parquet")
    parser.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--output-dir", default="models/cross_encoder_esci_v1")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--max-train-queries", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import sentence_transformers
        import torch
        import transformers
        from datasets import Dataset
        from sentence_transformers.cross_encoder import (
            CrossEncoder,
            CrossEncoderTrainer,
            CrossEncoderTrainingArguments,
            losses,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Missing Cross-Encoder dependencies. Run: pip install -e '.[ml,dev]'"
        ) from exc

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_path = Path(args.train)
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    frame = pd.read_parquet(train_path)
    required = {"query_id", "query", "product_text", "esci_label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Training data is missing columns: {sorted(missing)}")

    frame = frame.dropna(subset=["query", "product_text", "esci_label"]).copy()
    frame["query"] = frame["query"].astype(str)
    frame["product_text"] = frame["product_text"].astype(str)
    frame["esci_label"] = frame["esci_label"].astype(str)
    unknown_labels = set(frame["esci_label"].unique()).difference(LABEL_TO_ID)
    if unknown_labels:
        raise ValueError(f"Unknown ESCI labels: {sorted(unknown_labels)}")

    if args.max_train_queries is not None:
        query_ids = (
            frame[["query_id"]]
            .drop_duplicates()
            .sample(
                n=min(args.max_train_queries, frame["query_id"].nunique()),
                random_state=args.seed,
            )["query_id"]
        )
        frame = frame.loc[frame["query_id"].isin(set(query_ids.tolist()))].copy()

    frame["label"] = frame["esci_label"].map(LABEL_TO_ID).astype(int)
    dataset = Dataset.from_dict(
        {
            "query": frame["query"].tolist(),
            "product_text": frame["product_text"].tolist(),
            "label": frame["label"].tolist(),
        }
    ).select_columns(["query", "product_text", "label"])

    id_to_label = {index: label for index, label in enumerate(LABEL_ORDER)}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = CrossEncoder(
        args.base_model,
        num_labels=len(LABEL_ORDER),
        max_length=args.max_length,
        model_kwargs={
            "torch_dtype": torch.float32,
            "ignore_mismatched_sizes": True,
        },
        config_kwargs={
            "id2label": id_to_label,
            "label2id": LABEL_TO_ID,
            "problem_type": "single_label_classification",
        },
    )
    loss = losses.CrossEntropyLoss(model)

    use_cuda = torch.cuda.is_available()
    use_bf16 = bool(args.precision == "bf16" and use_cuda and torch.cuda.is_bf16_supported())
    use_fp16 = bool(args.precision == "fp16" and use_cuda)
    if args.precision == "bf16" and use_cuda and not use_bf16:
        raise RuntimeError("bf16 was requested but this CUDA device does not support bf16")

    training_args = CrossEncoderTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=use_cuda,
        save_strategy="epoch",
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=max(1, args.logging_steps),
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=True,
    )

    run_summary = {
        "base_model": args.base_model,
        "rows": int(len(frame)),
        "queries": int(frame["query_id"].nunique()),
        "label_counts": frame["esci_label"].value_counts().to_dict(),
        "label_order": list(LABEL_ORDER),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_length": args.max_length,
        "precision": args.precision,
        "seed": args.seed,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "cuda_available": bool(use_cuda),
        "gpu": torch.cuda.get_device_name(0) if use_cuda else None,
    }
    _write_json(run_summary, output_dir / "run_config.json")

    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=loss,
    )

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    train_result = trainer.train()
    wall_clock_seconds = time.perf_counter() - start

    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir), create_model_card=False)
    _write_json(
        {
            "label_order": list(LABEL_ORDER),
            "label_to_id": LABEL_TO_ID,
            "expected_gain": {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0},
        },
        final_dir / "esci_label_mapping.json",
    )

    run_summary.update(
        {
            "wall_clock_seconds": wall_clock_seconds,
            "peak_gpu_memory_gib": (
                float(torch.cuda.max_memory_allocated() / (1024**3)) if use_cuda else 0.0
            ),
            "train_runtime_seconds": float(
                train_result.metrics.get("train_runtime", wall_clock_seconds)
            ),
            "train_samples_per_second": float(
                train_result.metrics.get("train_samples_per_second", float("nan"))
            ),
            "train_steps_per_second": float(
                train_result.metrics.get("train_steps_per_second", float("nan"))
            ),
            "final_train_loss": float(
                train_result.metrics.get("train_loss", float("nan"))
            ),
        }
    )
    _write_json(run_summary, final_dir / "training_summary.json")
    print(f"Saved fine-tuned Cross-Encoder to {final_dir}")


if __name__ == "__main__":
    main()
