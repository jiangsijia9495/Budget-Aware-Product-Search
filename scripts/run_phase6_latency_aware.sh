#!/usr/bin/env bash
set -euo pipefail

export MPLBACKEND=Agg
GPU_ID="${GPU_ID:-0}"
LATENCY_FILE="${LATENCY_FILE:-outputs/adaptive_reranking/latency_samples.parquet}"
mkdir -p outputs/adaptive_reranking outputs/phase6_latency_aware

if [[ ! -f "$LATENCY_FILE" || "${FORCE_BENCHMARK:-0}" == "1" ]]; then
  echo "[1/2] Measure real RTX GPU stage latency (serial; excludes network and retrieval)"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python -m budget_aware_search.routing.benchmark_latency \
    --scored-val outputs/biencoder_v1_random_val.parquet \
    --bi-model models/biencoder_v1_random/final \
    --ce-model models/cross_encoder_esci_v1/final \
    --device cuda \
    --max-queries "${BENCHMARK_QUERIES:-500}" \
    --repeats "${BENCHMARK_REPEATS:-3}" \
    --output "$LATENCY_FILE"
else
  echo "[1/2] Reusing measured latency: $LATENCY_FILE"
fi

if [[ ! -f outputs/adaptive_reranking/query_features_and_metrics.parquet ]]; then
  echo "Missing Phase 5 validation features. Re-run Phase 5 quality analysis first." >&2
  exit 1
fi

echo "[2/2] Evaluate cost-aware routers and measured-stage latency Pareto"
args=(
  --phase5-features outputs/adaptive_reranking/query_features_and_metrics.parquet
  --train-scored outputs/biencoder_v1_random_train_scored.parquet
  --latency "$LATENCY_FILE"
  --output-dir outputs/phase6_latency_aware
  --max-rerank-rate "${MAX_RERANK_RATE:-0.50}"
  --target-gain-retention "${TARGET_GAIN_RETENTION:-95}"
)
if [[ -n "${P95_BUDGET_MS:-}" ]]; then
  args+=(--p95-budget-ms "$P95_BUDGET_MS")
fi
python -m budget_aware_search.routing.phase6_latency_aware "${args[@]}"
