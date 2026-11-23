#!/usr/bin/env bash
set -euo pipefail

mkdir -p models outputs

python -m budget_aware_search.training.train_cross_encoder \
  --train data/processed/train.parquet \
  --base-model cross-encoder/ms-marco-MiniLM-L6-v2 \
  --output-dir models/cross_encoder_esci_smoke \
  --epochs 1 \
  --batch-size 32 \
  --learning-rate 2e-5 \
  --max-length 256 \
  --precision bf16 \
  --max-train-queries 500 \
  --seed 42

python -m budget_aware_search.baselines.cross_encoder \
  --processed-dir data/processed \
  --split val \
  --model models/cross_encoder_esci_smoke/final \
  --device cuda \
  --batch-size 128 \
  --max-length 256 \
  --max-queries 100 \
  --bootstrap-samples 0 \
  --output outputs/cross_encoder_esci_smoke_val.parquet
