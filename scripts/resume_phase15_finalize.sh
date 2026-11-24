#!/usr/bin/env bash
set -euo pipefail
mkdir -p outputs/final_test
python -m budget_aware_search.routing.freeze_policy verify \
  --policy models/router_entropy_per_cost_frozen/policy.json
python -m budget_aware_search.routing.phase15_final_test finalize \
  --policy models/router_entropy_per_cost_frozen/policy.json \
  --test data/processed/test.parquet \
  --bi-scores outputs/final_test/biencoder_v1_test.parquet \
  --decisions outputs/final_test/router_decisions.parquet \
  --routed-ce outputs/final_test/cross_encoder_routed_test.parquet \
  --full-ce outputs/final_test/cross_encoder_full_test.parquet \
  --latency outputs/final_test/latency_samples.parquet \
  --output-dir outputs/final_test \
  --bootstrap-samples 5000
