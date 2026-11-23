#!/usr/bin/env bash
set -euo pipefail

# Avoid the multi-GPU/DataParallel issue encountered during Phase 4.
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MPLBACKEND=Agg
mkdir -p outputs/adaptive_reranking

BI_SCORES="${BI_SCORES:-outputs/biencoder_v1_random_val.parquet}"
CE_SCORES="${CE_SCORES:-outputs/cross_encoder_esci_v1_val.parquet}"
BI_PER_QUERY="${BI_PER_QUERY:-outputs/biencoder_v1_random_val.per_query.parquet}"
CE_PER_QUERY="${CE_PER_QUERY:-outputs/cross_encoder_esci_v1_val.per_query.parquet}"
CALIBRATION_SCORES="${CALIBRATION_SCORES:-outputs/biencoder_v1_random_train_scored.parquet}"
LATENCY_FILE="outputs/adaptive_reranking/latency_samples.parquet"

for file in "$BI_SCORES" "$CE_SCORES" "$BI_PER_QUERY" "$CE_PER_QUERY" "$CALIBRATION_SCORES"; do
  if [[ ! -f "$file" ]]; then
    echo "Missing required input: $file" >&2
    echo "For train calibration scores, rerun Phase 3 train-split scoring with frozen v1." >&2
    exit 1
  fi
done

if [[ "${SKIP_LATENCY:-0}" != "1" ]]; then
  echo "[1/2] Measure real single-query GPU latency (precomputed candidates)"
  python -m budget_aware_search.routing.benchmark_latency \
    --scored-val "$BI_SCORES" \
    --bi-model models/biencoder_v1_random/final \
    --ce-model models/cross_encoder_esci_v1/final \
    --device cuda \
    --max-queries "${BENCHMARK_QUERIES:-200}" \
    --repeats "${BENCHMARK_REPEATS:-2}" \
    --output "$LATENCY_FILE"
  LATENCY_ARGS=(--latency "$LATENCY_FILE")
else
  echo "[1/2] Skipping latency benchmark; quality-vs-rerank-rate only"
  LATENCY_ARGS=()
fi

echo "[2/2] Calibrate routing thresholds on TRAIN; evaluate on fixed VAL"
python -m budget_aware_search.routing.adaptive_reranking \
  --bi-scores "$BI_SCORES" \
  --ce-scores "$CE_SCORES" \
  --bi-per-query "$BI_PER_QUERY" \
  --ce-per-query "$CE_PER_QUERY" \
  --calibration-scores "$CALIBRATION_SCORES" \
  --expected-val-queries 3133 \
  --target-gain-retention 95 \
  --max-rerank-rate 0.50 \
  --output-dir outputs/adaptive_reranking \
  "${LATENCY_ARGS[@]}"

echo "See:"
echo "  outputs/adaptive_reranking/summary.json"
echo "  outputs/adaptive_reranking/pareto_sweep.csv"
echo "  outputs/adaptive_reranking/pareto_frontier.csv"
echo "  outputs/adaptive_reranking/quality_vs_rerank_rate.png"
echo "  outputs/adaptive_reranking/quality_vs_p95_latency.png (if latency enabled)"
