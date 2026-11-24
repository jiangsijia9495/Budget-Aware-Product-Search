#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MPLBACKEND=Agg
mkdir -p outputs/phase7_shadow outputs/phase7_learned models/phase7_shadow data/processed/phase7

for f in data/processed/train.parquet data/processed/biencoder_pairs_v1.parquet \
         outputs/adaptive_reranking/query_features_and_metrics.parquet \
         outputs/adaptive_reranking/latency_samples.parquet \
         outputs/phase6_latency_aware/policy_sweep.csv; do
  if [[ ! -f "$f" ]]; then
    echo "Missing input: $f" >&2
    exit 1
  fi
done

echo "[1/6] Create query-disjoint shadow train and router-label holdout from official TRAIN"
if [[ ! -f data/processed/phase7/manifest.json ]]; then
  python -m budget_aware_search.routing.phase7_prepare \
    --train data/processed/train.parquet \
    --v1-pairs data/processed/biencoder_pairs_v1.parquet \
    --output-dir data/processed/phase7 \
    --holdout-fraction 0.20 --seed 42
else
  echo "Reusing fixed Phase 7 holdout split"
fi

# Phase 7 dense scoring needs catalog.parquet inside router_scoring.
# Also repair existing Phase 7 runs where the split manifest causes step 1
# to be skipped. No model retraining or split regeneration is needed.
SCORING_DIR=data/processed/phase7/router_scoring
if [[ ! -f "$SCORING_DIR/train.parquet" ]]; then
  if [[ -f data/processed/phase7/router_holdout.parquet ]]; then
    mkdir -p "$SCORING_DIR"
    ln -s ../router_holdout.parquet "$SCORING_DIR/train.parquet"
  else
    echo "Missing Phase 7 held-out query data; cannot resume safely" >&2
    exit 1
  fi
fi
if [[ ! -f "$SCORING_DIR/catalog.parquet" ]]; then
  if [[ ! -f data/processed/catalog.parquet ]]; then
    echo "Missing data/processed/catalog.parquet; cannot score shadow holdout" >&2
    exit 1
  fi
  mkdir -p "$SCORING_DIR"
  if [[ -L "$SCORING_DIR/catalog.parquet" ]]; then
    rm "$SCORING_DIR/catalog.parquet"
  fi
  ln -s ../../catalog.parquet "$SCORING_DIR/catalog.parquet"
fi

echo "[2/6] Train shadow bi-encoder ONLY on shadow_train queries"
if [[ ! -f models/phase7_shadow/bi/final/modules.json ]]; then
  python -m budget_aware_search.training.train_biencoder \
    --pairs data/processed/phase7/shadow_pairs.parquet \
    --base-model sentence-transformers/all-MiniLM-L6-v2 \
    --output-dir models/phase7_shadow/bi \
    --epochs 1 --batch-size "${BI_BATCH_SIZE:-128}" \
    --learning-rate 2e-5 --max-seq-length 256 \
    --precision bf16 --loss mnrl --seed 42
fi

 echo "[3/6] Train shadow Cross-Encoder ONLY on shadow_train queries"
if [[ ! -f models/phase7_shadow/ce/final/config.json ]]; then
  python -m budget_aware_search.training.train_cross_encoder \
    --train data/processed/phase7/shadow_train.parquet \
    --base-model cross-encoder/ms-marco-MiniLM-L6-v2 \
    --output-dir models/phase7_shadow/ce \
    --epochs 1 --batch-size "${CE_BATCH_SIZE:-64}" \
    --learning-rate 2e-5 --max-length 256 \
    --precision bf16 --seed 42
fi

 echo "[4/6] OOS shadow bi scores and per-query quality (no model trained on these queries)"
if [[ ! -f outputs/phase7_shadow/bi_router_fit.parquet ]]; then
  python -m budget_aware_search.baselines.dense \
    --processed-dir data/processed/phase7/router_scoring --split train \
    --model models/phase7_shadow/bi/final --device cuda \
    --batch-size "${EVAL_BATCH_SIZE:-512}" \
    --cache-dir models/phase7_shadow/dense_cache --reencode-catalog \
    --output outputs/phase7_shadow/bi_router_fit.parquet \
    --bootstrap-samples 0
fi

 echo "[5/6] OOS shadow CE scores and per-query quality"
if [[ ! -f outputs/phase7_shadow/ce_router_fit.parquet ]]; then
  python -m budget_aware_search.baselines.cross_encoder \
    --processed-dir data/processed/phase7/router_scoring --split train \
    --model models/phase7_shadow/ce/final --device cuda \
    --batch-size "${CE_EVAL_BATCH_SIZE:-128}" --max-length 256 \
    --bootstrap-samples 0 \
    --output outputs/phase7_shadow/ce_router_fit.parquet
fi

 echo "[6/6] Train learned gain router, compare with Phase 6, plot Pareto"
python -m budget_aware_search.routing.phase7_learned_router \
  --p95-budget-ms "${P95_BUDGET_MS:-20}" \
  --max-rate "${MAX_RERANK_RATE:-0.5}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-1000}" \
  --output-dir outputs/phase7_learned

echo "Review outputs/phase7_learned/summary.json"
echo "Review outputs/phase7_learned/combined_pareto_frontier.csv"
