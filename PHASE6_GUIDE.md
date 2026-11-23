# Phase 6: Measured-latency-aware adaptive reranking

## What this adds

No model retraining. Reuse the accepted bi-encoder v1 and Cross-Encoder checkpoint,
Phase 5's correctly audited all-query quality metrics, and **actual RTX 3090 stage
latency observations**.

- Baseline router: normalized score entropy / margin / top-1 / seeded random.
- New router: entropy / predicted CE stage milliseconds (cost-aware).
- New router: inverse margin / predicted CE stage milliseconds (cost-aware).
- Diagnostic **oracle upper bound** (uses validation labels; never deploy it).
- Train-query-score calibration of thresholds; no relevance labels are used for
  deployment routing decisions.
- Full 3,133-query validation quality with paired-query bootstrap intervals.
- P50/P95/P99 latency **replay on measured RTX 3090 benchmark queries**, with
  95% query-cluster bootstrap CI for P95.
- Non-dominated quality-vs-P95 frontier; best policy under both a max rerank
  rate and optional P95 budget.

## Accuracy provenance

Use the checkpoint provenance-verified scores. The currently audited CE endpoint
is **0.771906437** nDCG@10, not the historical 0.736121 result. Bi-encoder
v1 is **0.719433362**. Thus 95% incremental gain retention corresponds to
approximately **0.769283** nDCG@10.

Audit the score/metric artifacts first if any checkpoint or validation file was
regenerated:

```bash
python -m budget_aware_search.routing.audit_phase4_artifacts
```

## Installation

Extract `phase6-latency-aware-routing.zip` in the existing project root:

```bash
unzip -o phase6-latency-aware-routing.zip -d .
pip install matplotlib
pytest -q tests/test_phase6_latency_aware.py
```

This adds only a new module, runner, test and guide; it does not overwrite
Phase 4 or Phase 5 code or models.

## Run using already measured RTX 3090 latency

Expected prior output:

```text
outputs/adaptive_reranking/latency_samples.parquet
outputs/adaptive_reranking/query_features_and_metrics.parquet
outputs/biencoder_v1_random_train_scored.parquet
```

Then:

```bash
bash scripts/run_phase6_latency_aware.sh \
  2>&1 | tee outputs/phase6_latency_aware.log
```

If the timing parquet is missing, the script automatically runs a real GPU
benchmark with `BENCHMARK_QUERIES=500` and `BENCHMARK_REPEATS=3`, using
`GPU_ID=0` by default. For stronger P95 estimates, regenerate using 1000 or
more diverse queries:

```bash
GPU_ID=0 FORCE_BENCHMARK=1 BENCHMARK_QUERIES=1000 BENCHMARK_REPEATS=3 \
  bash scripts/run_phase6_latency_aware.sh \
  2>&1 | tee outputs/phase6_latency_aware_repeat.log
```

For an explicit stage-latency budget (illustrative; choose from actual measured
baseline latencies, not an arbitrary requirement):

```bash
P95_BUDGET_MS=150 MAX_RERANK_RATE=0.5 \
  bash scripts/run_phase6_latency_aware.sh
```

The budget is applied to P95 of the **measured-subset offline stage replay**.

## Outputs

```text
outputs/phase6_latency_aware/
    policy_sweep.csv
    pareto_frontier.csv
    oracle_upper_bound.csv
    quality_vs_rate_with_oracle.png
    quality_vs_measured_p95.png
    summary.json
```

Read `summary.json` first: it contains measured latency sample coverage,
observed medians, the cost-model holdout error and the selected policy.

## Interpretation and next experiment

First examine the oracle upper bound at 50% budget:

- If oracle retention <95%, the 95%-at-50% target is unattainable even with
  per-query relevance labels and unlimited routing intelligence under the
  current two fixed rankers.
- If oracle retention >=95% but the heuristics fall short, the problem is
  **router informativeness**, not a lack of CE ranking quality. Then train a
  *gain-prediction router* with proper train/validation splits or introduce
  model distillation; do not make the router depend on evaluation labels.
- If entropy-per-cost is non-dominated and improves P95, compare its bootstrap
  uncertainty and freeze the selected configuration before the official test.

### Timing limitations

Candidate embeddings are precomputed out of the timed region. CE time includes
one batch over the judged candidates, including tokenization. Timings are
GPU-synchronized and serial. No network, catalog search/FAISS, concurrency,
queueing, startup, or model-load costs are included. The routing decision and
feature-extraction overhead beyond the existing bi margin timing may also be
excluded. P95 is from the measured subset, not all 3,133 validation queries;
quality is from all 3,133. `estimated_serial_qps` is *not* online serving QPS.

Because the policy is chosen on validation and the latency sample may be small,
all Phase 6 decisions remain exploratory until a frozen-policy evaluation on
unseen test queries and a repeat latency benchmark on a disjoint sample.
