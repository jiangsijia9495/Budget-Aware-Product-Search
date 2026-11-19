from __future__ import annotations

import argparse
import importlib.metadata
import platform
import subprocess
import time
from pathlib import Path

import pandas as pd

from budget_aware_search.utils import set_seed, write_json


def _imports():
    try:
        import torch
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
    return (
        torch,
        Dataset,
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
        BatchSamplers,
    )


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a bi-encoder on ESCI triples.")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output-dir", default="models/biencoder")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--loss",
        choices=["mnrl", "cached-mnrl"],
        default="mnrl",
        help="Use standard MultipleNegativesRankingLoss or its memory-efficient cached form.",
    )
    parser.add_argument(
        "--cached-mini-batch-size",
        type=int,
        default=32,
        help="Internal embedding mini-batch used only by cached-mnrl.",
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    (
        torch,
        Dataset,
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
        BatchSamplers,
    ) = _imports()
    set_seed(args.seed)

    if args.batch_size <= 1:
        raise ValueError("batch-size must be greater than 1 for in-batch negatives")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup-ratio must be in [0, 1)")
    if args.precision in {"fp16", "bf16"} and not torch.cuda.is_available():
        raise RuntimeError(f"{args.precision} training requires CUDA")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support bf16; use --precision fp16")
    if args.gradient_accumulation_steps > 1 and args.loss == "mnrl":
        print(
            "WARNING: gradient accumulation increases the optimizer effective batch size but "
            "does not enlarge the per-step in-batch negative pool. Prefer a larger physical "
            "batch or --loss cached-mnrl when possible."
        )

    pairs = pd.read_parquet(args.pairs)
    required = {"anchor", "positive", "negative"}
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"Training pairs are missing columns: {sorted(missing)}")
    if args.max_train_pairs and len(pairs) > args.max_train_pairs:
        pairs = pairs.sample(args.max_train_pairs, random_state=args.seed).reset_index(drop=True)

    dataset = Dataset.from_pandas(
        pairs[["anchor", "positive", "negative"]].reset_index(drop=True),
        preserve_index=False,
    )

    model = SentenceTransformer(args.base_model)
    model.max_seq_length = args.max_seq_length
    if args.loss == "cached-mnrl":
        loss = losses.CachedMultipleNegativesRankingLoss(
            model,
            mini_batch_size=args.cached_mini_batch_size,
        )
    else:
        loss = losses.MultipleNegativesRankingLoss(model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_steps = max(1, min(50, len(dataset) // max(args.batch_size * 10, 1)))

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=args.precision == "fp16",
        bf16=args.precision == "bf16",
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=logging_steps,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=args.num_workers,
        dataloader_drop_last=True,
        run_name=output_dir.name,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=loss,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    wall_clock_seconds = time.perf_counter() - started

    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    peak_gpu_memory_gib = None
    gpu_name = None
    if torch.cuda.is_available():
        peak_gpu_memory_gib = float(torch.cuda.max_memory_allocated() / (1024**3))
        gpu_name = torch.cuda.get_device_name(0)

    write_json(
        {
            "base_model": args.base_model,
            "training_pairs": int(len(dataset)),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "precision": args.precision,
            "max_seq_length": args.max_seq_length,
            "loss": args.loss,
            "cached_mini_batch_size": (
                args.cached_mini_batch_size if args.loss == "cached-mnrl" else None
            ),
            "seed": args.seed,
            "wall_clock_seconds": wall_clock_seconds,
            "peak_gpu_memory_gib": peak_gpu_memory_gib,
            "gpu_name": gpu_name,
            "train_metrics": dict(train_result.metrics),
            "git_commit": _git_commit(),
            "python_version": platform.python_version(),
            "package_versions": {
                "torch": _version("torch"),
                "sentence-transformers": _version("sentence-transformers"),
                "transformers": _version("transformers"),
                "datasets": _version("datasets"),
            },
        },
        final_dir / "training_summary.json",
    )
    print(f"Saved fine-tuned bi-encoder to {final_dir}")
    print(f"Wall-clock training time: {wall_clock_seconds / 60.0:.2f} minutes")
    if peak_gpu_memory_gib is not None:
        print(f"Peak allocated GPU memory: {peak_gpu_memory_gib:.2f} GiB")


if __name__ == "__main__":
    main()
