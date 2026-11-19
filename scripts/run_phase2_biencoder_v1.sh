#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PAIRS="${PAIRS:-data/processed/biencoder_pairs_v1.parquet}"
MODEL_DIR="${MODEL_DIR:-models/biencoder_v1_random}"
PREDICTIONS="${PREDICTIONS:-outputs/biencoder_v1_random_val.parquet}"
BATCH_SIZE="${BATCH_SIZE:-128}"
PRECISION="${PRECISION:-bf16}"
EPOCHS="${EPOCHS:-1}"

if [[ -n "${BASELINE_PER_QUERY:-}" ]]; then
  BASELINE="${BASELINE_PER_QUERY}"
elif [[ -f outputs/dense_pretrained_val.per_query.parquet ]]; then
  BASELINE="outputs/dense_pretrained_val.per_query.parquet"
elif [[ -f outputs/dense_val.per_query.parquet ]]; then
  BASELINE="outputs/dense_val.per_query.parquet"
else
  echo "Could not find pretrained dense per-query results."
  echo "Set BASELINE_PER_QUERY=/path/to/dense_pretrained_val.per_query.parquet"
  exit 2
fi

echo "[1/5] Building query-balanced random-negative triplets"
"${PYTHON_BIN}" -m budget_aware_search.training.build_pairs \
  --input data/processed/train.parquet \
  --output "${PAIRS}" \
  --max-positives-per-query 1 \
  --negatives-per-positive 1 \
  --negative-label-priority I,C \
  --seed 42

echo "[2/5] Auditing triplets"
"${PYTHON_BIN}" -m budget_aware_search.training.audit_pairs \
  --pairs "${PAIRS}"

echo "[3/5] Training bi-encoder v1"
"${PYTHON_BIN}" -m budget_aware_search.training.train_biencoder \
  --pairs "${PAIRS}" \
  --base-model sentence-transformers/all-MiniLM-L6-v2 \
  --output-dir "${MODEL_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --learning-rate 2e-5 \
  --max-seq-length 256 \
  --precision "${PRECISION}" \
  --loss mnrl \
  --seed 42

echo "[4/5] Scoring the unchanged validation candidate pool"
"${PYTHON_BIN}" -m budget_aware_search.baselines.dense \
  --processed-dir data/processed \
  --split val \
  --model "${MODEL_DIR}/final" \
  --device cuda \
  --batch-size 512 \
  --cache-dir models/dense_cache \
  --reencode-catalog \
  --output "${PREDICTIONS}" \
  --bootstrap-samples 1000

echo "[5/5] Paired comparison against the pretrained dense baseline"
"${PYTHON_BIN}" -m budget_aware_search.evaluation.compare_rankings \
  --baseline "${BASELINE}" \
  --candidate "${PREDICTIONS%.parquet}.per_query.parquet" \
  --output "${PREDICTIONS%.parquet}.vs_pretrained.json" \
  --bootstrap-samples 5000

echo "Phase 2 v1 completed. Key files:"
echo "  ${PAIRS%.parquet}.summary.json"
echo "  ${PAIRS%.parquet}.audit.json"
echo "  ${MODEL_DIR}/final/training_summary.json"
echo "  ${PREDICTIONS%.parquet}.metrics.json"
echo "  ${PREDICTIONS%.parquet}.vs_pretrained.json"
