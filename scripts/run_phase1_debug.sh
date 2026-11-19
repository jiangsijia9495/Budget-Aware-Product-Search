#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python -m budget_aware_search.data \
  --config configs/base.yaml \
  --output-dir data/processed \
  --debug-max-queries 300

python -m budget_aware_search.baselines.source_order \
  --input data/processed/val.parquet \
  --output outputs/source_order_val.parquet

python -m budget_aware_search.baselines.bm25 \
  --processed-dir data/processed \
  --split val \
  --index-dir models/bm25_debug \
  --output outputs/bm25_val.parquet \
  --refit

python -m budget_aware_search.baselines.dense \
  --processed-dir data/processed \
  --split val \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --device cuda \
  --cache-dir models/dense_cache_debug \
  --output outputs/dense_val.parquet

echo "Phase 1 debug run complete. Inspect outputs/*.metrics.json"
