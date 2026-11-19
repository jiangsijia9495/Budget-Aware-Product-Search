from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from budget_aware_search.labels import CLASS_TO_ID, ID_TO_CLASS
from budget_aware_search.utils import set_seed, write_json


def _imports():
    try:
        import torch
        from datasets import Dataset
        from sentence_transformers.cross_encoder import (
            CrossEncoder,
            CrossEncoderTrainer,
            CrossEncoderTrainingArguments,
            losses,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install with: pip install -e '.[ml]'"
        ) from exc
    return torch, Dataset, CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments, losses


def _to_dataset(frame: pd.DataFrame, Dataset):
    data = pd.DataFrame(
        {
            "query": frame["query"].astype(str),
            "product": frame["product_text"].fillna("").astype(str),
            "label": frame["esci_label"].map(CLASS_TO_ID).astype(int),
        }
    )
    return Dataset.from_pandas(data.reset_index(drop=True), preserve_index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 4-class ESCI cross-encoder reranker.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--base-model", default="microsoft/MiniLM-L12-H384-uncased")
    parser.add_argument("--output-dir", default="models/cross_encoder")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-validation-rows", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    args = parser.parse_args()

    torch, Dataset, CrossEncoder, Trainer, TrainingArguments, losses = _imports()
    set_seed(args.seed)
    train = pd.read_parquet(args.train)
    validation = pd.read_parquet(args.validation)
    if args.max_train_rows and len(train) > args.max_train_rows:
        train = train.sample(args.max_train_rows, random_state=args.seed)
    if args.max_validation_rows and len(validation) > args.max_validation_rows:
        validation = validation.sample(args.max_validation_rows, random_state=args.seed)

    train_dataset = _to_dataset(train, Dataset)
    validation_dataset = _to_dataset(validation, Dataset)
    model = CrossEncoder(
        args.base_model,
        num_labels=4,
        max_length=args.max_length,
    )

    counts = train["esci_label"].value_counts()
    raw_weights = np.asarray(
        [1.0 / np.sqrt(max(int(counts.get(ID_TO_CLASS[i], 0)), 1)) for i in range(4)],
        dtype=np.float32,
    )
    raw_weights /= raw_weights.mean()
    class_weights = torch.tensor(raw_weights, dtype=torch.float32)
    loss = losses.CrossEntropyLoss(model, weight=class_weights)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = max(1, len(train_dataset) // max(args.batch_size, 1))
    eval_steps = max(100, min(1000, steps_per_epoch // 5 or 100))
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        fp16=args.precision == "fp16",
        bf16=args.precision == "bf16",
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=max(10, eval_steps // 5),
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=4,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        loss=loss,
    )
    trainer.train()
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    write_json(
        {
            "base_model": args.base_model,
            "class_to_id": CLASS_TO_ID,
            "class_weights": raw_weights.tolist(),
            "train_rows": int(len(train_dataset)),
            "validation_rows": int(len(validation_dataset)),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "precision": args.precision,
            "max_length": args.max_length,
        },
        final_dir / "training_summary.json",
    )
    print(f"Saved cross-encoder reranker to {final_dir}")


if __name__ == "__main__":
    main()
