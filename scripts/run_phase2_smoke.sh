#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PAIRS="data/processed/biencoder_pairs_v1.parquet"

if [[ ! -f "${PAIRS}" ]]; then
  "${PYTHON_BIN}" -m budget_aware_search.training.build_pairs \
    --input data/processed/train.parquet \
    --output "${PAIRS}" \
    --max-positives-per-query 1 \
    --negatives-per-positive 1 \
    --negative-label-priority I,C \
    --seed 42
fi

"${PYTHON_BIN}" -m budget_aware_search.training.audit_pairs --pairs "${PAIRS}"
"${PYTHON_BIN}" -m budget_aware_search.training.train_biencoder \
  --pairs "${PAIRS}" \
  --base-model sentence-transformers/all-MiniLM-L6-v2 \
  --output-dir models/biencoder_v1_smoke \
  --epochs 1 \
  --batch-size 64 \
  --learning-rate 2e-5 \
  --precision bf16 \
  --loss mnrl \
  --max-train-pairs 1024 \
  --seed 42

echo "Smoke test passed if models/biencoder_v1_smoke/final exists."
