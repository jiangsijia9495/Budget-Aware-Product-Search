from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from budget_aware_search.utils import set_seed, write_json


def _imports():
    try:
        from datasets import Dataset
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
            losses,
        )
        try:
            from sentence_transformers.training_args import BatchSamplers
        except ImportError:
            from sentence_transformers.sentence_transformer.training_args import BatchSamplers
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install with: pip install -e '.[ml]'"
        ) from exc
    return Dataset, SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses, BatchSamplers


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill cross-encoder scores into a bi-encoder.")
    parser.add_argument("--teacher-scores", required=True)
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output-dir", default="models/distilled_biencoder")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Dataset, SentenceTransformer, Trainer, TrainingArguments, losses, BatchSamplers = _imports()
    set_seed(args.seed)
    frame = pd.read_parquet(args.teacher_scores)
    required = {"query", "product_text", "score"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Teacher scores missing columns: {sorted(missing)}")
    if args.max_rows and len(frame) > args.max_rows:
        frame = frame.sample(args.max_rows, random_state=args.seed)

    dataset_frame = pd.DataFrame(
        {
            "sentence1": frame["query"].astype(str),
            "sentence2": frame["product_text"].fillna("").astype(str),
            "label": frame["score"].clip(0.0, 1.0).astype(float),
        }
    )
    dataset = Dataset.from_pandas(dataset_frame, preserve_index=False)
    model = SentenceTransformer(args.base_model)
    loss = losses.CoSENTLoss(model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        fp16=args.precision == "fp16",
        bf16=args.precision == "bf16",
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=max(10, min(100, len(dataset) // max(args.batch_size * 20, 1))),
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=loss,
    )
    trainer.train()
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    write_json(
        {
            "base_model": args.base_model,
            "teacher_rows": int(len(dataset)),
            "loss": "CoSENTLoss",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "precision": args.precision,
        },
        final_dir / "distillation_summary.json",
    )
    print(f"Saved distilled bi-encoder to {final_dir}")


if __name__ == "__main__":
    main()
