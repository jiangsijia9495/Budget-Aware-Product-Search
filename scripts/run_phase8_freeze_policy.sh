#!/usr/bin/env bash
set -euo pipefail

# Does not train models or access official TEST.
python -m budget_aware_search.routing.freeze_policy freeze \
  --phase7-summary outputs/phase7_learned/summary.json \
  --phase6-sweep outputs/phase6_latency_aware/policy_sweep.csv \
  --calibration-scores outputs/biencoder_v1_random_train_scored.parquet \
  --val-features outputs/adaptive_reranking/query_features_and_metrics.parquet \
  --latency outputs/adaptive_reranking/latency_samples.parquet \
  --bi-model models/biencoder_v1_random/final \
  --ce-model models/cross_encoder_esci_v1/final \
  --output models/router_entropy_per_cost_frozen/policy.json

python -m budget_aware_search.routing.freeze_policy verify \
  --policy models/router_entropy_per_cost_frozen/policy.json
