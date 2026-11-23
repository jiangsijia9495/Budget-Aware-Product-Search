#!/usr/bin/env bash
set -euo pipefail

V1_PER_QUERY="${V1_PER_QUERY:-outputs/biencoder_v1_random_val.per_query.parquet}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
PRECISION="${PRECISION:-bf16}"
EPOCHS="${EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-256}"

for required in data/processed/train.parquet data/processed/val.parquet "$V1_PER_QUERY"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

mkdir -p models outputs

ZERO_OUTPUT=outputs/cross_encoder_ms_marco_zero_val.parquet
MODEL_DIR=models/cross_encoder_esci_v1
FT_OUTPUT=outputs/cross_encoder_esci_v1_val.parquet

if [[ ! -f "$ZERO_OUTPUT" ]]; then
  echo "[1/5] Zero-shot MS MARCO Cross-Encoder on the fixed validation candidate pool"
  python -m budget_aware_search.baselines.cross_encoder \
    --processed-dir data/processed \
    --split val \
    --model cross-encoder/ms-marco-MiniLM-L6-v2 \
    --device cuda \
    --batch-size "$EVAL_BATCH_SIZE" \
    --max-length "$MAX_LENGTH" \
    --bootstrap-samples 1000 \
    --output "$ZERO_OUTPUT"
else
  echo "[1/5] Reusing $ZERO_OUTPUT"
fi

python -m budget_aware_search.evaluation.compare_rankings \
  --baseline "$V1_PER_QUERY" \
  --candidate "${ZERO_OUTPUT%.parquet}.per_query.parquet" \
  --output outputs/cross_encoder_ms_marco_zero_val.vs_v1.json \
  --bootstrap-samples 5000

if [[ ! -d "$MODEL_DIR/final" ]]; then
  echo "[2/5] Fine-tune four-class ESCI Cross-Encoder"
  python -m budget_aware_search.training.train_cross_encoder \
    --train data/processed/train.parquet \
    --base-model cross-encoder/ms-marco-MiniLM-L6-v2 \
    --output-dir "$MODEL_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --max-length "$MAX_LENGTH" \
    --precision "$PRECISION" \
    --seed 42
else
  echo "[2/5] Reusing $MODEL_DIR/final"
fi

if [[ ! -f "$FT_OUTPUT" ]]; then
  echo "[3/5] Evaluate fine-tuned Cross-Encoder"
  python -m budget_aware_search.baselines.cross_encoder \
    --processed-dir data/processed \
    --split val \
    --model "$MODEL_DIR/final" \
    --device cuda \
    --batch-size "$EVAL_BATCH_SIZE" \
    --max-length "$MAX_LENGTH" \
    --bootstrap-samples 1000 \
    --output "$FT_OUTPUT"
else
  echo "[3/5] Reusing $FT_OUTPUT"
fi

echo "[4/5] Paired bootstrap: fine-tuned Cross-Encoder versus champion v1 bi-encoder"
python -m budget_aware_search.evaluation.compare_rankings \
  --baseline "$V1_PER_QUERY" \
  --candidate "${FT_OUTPUT%.parquet}.per_query.parquet" \
  --output outputs/cross_encoder_esci_v1_val.vs_biencoder_v1.json \
  --bootstrap-samples 5000

echo "[5/5] Finished. Review:"
echo "  outputs/cross_encoder_ms_marco_zero_val.metrics.json"
echo "  outputs/cross_encoder_ms_marco_zero_val.vs_v1.json"
echo "  $MODEL_DIR/final/training_summary.json"
echo "  outputs/cross_encoder_esci_v1_val.metrics.json"
echo "  outputs/cross_encoder_esci_v1_val.vs_biencoder_v1.json"
