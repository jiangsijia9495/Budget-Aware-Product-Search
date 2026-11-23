# Phase 5 — Adaptive Reranking & Latency–Quality Pareto

This patch adds two new modules and does **not** change trained models, data splits, candidate sets, ranking metric implementations, or prior outputs.

## Scope

The experiment is **ESCI judged-candidate-pool reranking**, not catalog-wide ANN retrieval or production serving.

* Fixed validation: 3,133 query IDs.
* Baseline: `outputs/biencoder_v1_random_val.per_query.parquet` (nDCG@10 ~0.719433).
* Reranker: `outputs/cross_encoder_esci_v1_val.per_query.parquet` (nDCG@10 ~0.736121).
* All intermediate policies combine the existing per-query metrics. **No Cross-Encoder retraining is required.**
* Difficulty signals use only bi-encoder query-product scores, **never relevance labels**.
* Confidence thresholds for each target reranking rate are fit on the **training** query scores. The actual validation reranking rate can differ from the target.
* The best policy is chosen on validation and therefore is exploratory. Freeze it before touching the official test set.

## Install

From the user's existing repository root:

```bash
unzip -o phase5-adaptive-reranking.zip -d .
pip install -e ".[ml,dev]"
pip install matplotlib
pytest -q tests/test_phase5_adaptive.py
```

Make sure your existing environment includes `numpy`, `pandas`, `pyarrow`, `torch`, and `sentence-transformers`.

## Inputs

Required preexisting files:

```text
outputs/biencoder_v1_random_val.parquet
outputs/biencoder_v1_random_val.per_query.parquet
outputs/cross_encoder_esci_v1_val.parquet
outputs/cross_encoder_esci_v1_val.per_query.parquet
outputs/biencoder_v1_random_train_scored.parquet
models/biencoder_v1_random/final/
models/cross_encoder_esci_v1/final/
```

If the TRAIN scores file is missing, regenerate it using the frozen v1 model. It is the **train** split, not validation:

```bash
python -m budget_aware_search.baselines.dense \
  --processed-dir data/processed \
  --split train \
  --model models/biencoder_v1_random/final \
  --device cuda \
  --batch-size 512 \
  --cache-dir models/dense_cache_v1_train \
  --reencode-catalog \
  --output outputs/biencoder_v1_random_train_scored.parquet \
  --bootstrap-samples 0
```

If score column names differ from `score`/`cross_score`, override them in the analysis CLI via `--bi-score-column`, `--ce-score-column`, and `--calibration-score-column`.

## Run

First validate everything and generate the quality vs rerank-rate curve **without GPU**:

```bash
SKIP_LATENCY=1 bash scripts/run_phase5_adaptive.sh \
  2>&1 | tee outputs/phase5_quality_only.log
```

Then run a measured, single-GPU benchmark and latency-quality analysis:

```bash
GPU_ID=0 BENCHMARK_QUERIES=200 BENCHMARK_REPEATS=2 \
  bash scripts/run_phase5_adaptive.sh \
  2>&1 | tee outputs/phase5_quality_latency.log
```

Increase `BENCHMARK_QUERIES` to 500–1000 for a less noisy P95 estimate after the initial run. Benchmark should be done on a largely idle GPU and ideally repeated at different times.

To impose a specific latency budget (e.g., P95 <= 150ms), run:

```bash
python -m budget_aware_search.routing.adaptive_reranking \
  --bi-scores outputs/biencoder_v1_random_val.parquet \
  --ce-scores outputs/cross_encoder_esci_v1_val.parquet \
  --bi-per-query outputs/biencoder_v1_random_val.per_query.parquet \
  --ce-per-query outputs/cross_encoder_esci_v1_val.per_query.parquet \
  --calibration-scores outputs/biencoder_v1_random_train_scored.parquet \
  --latency outputs/adaptive_reranking/latency_samples.parquet \
  --p95-budget-ms 150
```

## Policies and accounting

Each threshold is calibrated only using train-query score distributions; harder queries are reranked:

* `margin`: small top-1/top-2 bi-score gap.
* `entropy`: large entropy of softmax-normalized bi-scores, temperature 0.1 (a *heuristic*, not calibrated probability).
* `top1`: low best bi-encoder score.
* `random`: seeded hash-query control.

Routing rates are swept from 0% to 100% in 10% steps. Report `rerank_rate`, not just `target_rerank_rate`. Do not use per-query CE gains in deployment routing.

**Latency instrumentation**: score the same sampled val queries using the frozen v1 bi-encoder and frozen Cross-Encoder. Candidate product embeddings are precomputed out of the timed region; bi timing includes query encoding and local candidate scoring; CE timing includes tokenization and a single batch of all judged candidates; GPU synchronization surrounds each timed stage. Policy latency is an **offline replay** of separately measured stages, with P95 computed over the timed query/repeat observations. These are *not* real server P95/QPS numbers and exclude loading, networking, retrieval index search, concurrency, and queueing.

## Files produced

```text
outputs/adaptive_reranking/latency_samples.parquet
outputs/adaptive_reranking/latency_samples.summary.json
outputs/adaptive_reranking/pareto_sweep.csv
outputs/adaptive_reranking/pareto_frontier.csv
outputs/adaptive_reranking/summary.json
outputs/adaptive_reranking/query_features_and_metrics.parquet
outputs/adaptive_reranking/routing_decisions.parquet
outputs/adaptive_reranking/quality_vs_rerank_rate.png
outputs/adaptive_reranking/quality_vs_p95_latency.png
```

## Decision criterion

Original models:

```text
bi-encoder nDCG@10:   0.719433
Cross-Encoder nDCG@10: 0.736121
```

To retain 95% of the CE **incremental** quality gain over the bi-encoder, the target is:

```text
0.719433 + 0.95 * (0.736121 - 0.719433) = ~0.735287
```

Try to reach it while reranking <=50% of queries. If the goal is infeasible, report the best validated point under <=50% rather than pretending it worked.

Select the route *using validation only*, then lock `policy`, `threshold`, models, temperature, candidate pool and latency protocol before an untouched test run. Treat multiple validation-based policy trials as exploratory; a final test should be run once.
