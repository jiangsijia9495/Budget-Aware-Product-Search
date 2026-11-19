# Phase 2: Fine-tuned Bi-Encoder v1

This phase starts from the completed validation baselines recorded on 2026-11-18:

- 17,755 train queries
- 3,133 validation queries
- Pretrained dense nDCG@10: 0.709680
- BM25 nDCG@10: 0.691749
- Source-order nDCG@10: 0.579276

The test split remains untouched. Model selection is performed only on the validation split.

## What changed in this patch

1. Random-negative construction now truly follows `I -> C` priority.
2. Substitute (`S`) products are not treated as explicit negatives in v1.
3. A pair-audit command blocks training if it finds same-product pairs, empty text,
   duplicate triplets, or a negative with gain greater than or equal to the positive.
4. Training records wall-clock time, peak GPU memory, package versions, seed, loss,
   and the Git commit.
5. A paired query-level bootstrap compares the fine-tuned model against the exact
   pretrained dense baseline on the same 3,133 validation queries.

## Step 0: Update the editable install

Run from the repository root:

```bash
pip install -e ".[ml,dev]"
pytest -q
```

Expected unit-test result for this patch:

```text
18 passed
```

## Step 1: Smoke test

This trains on only 1,024 triplets and verifies the complete training stack.

```bash
bash scripts/run_phase2_smoke.sh 2>&1 | tee outputs/phase2_smoke.log
```

Success conditions:

```text
models/biencoder_v1_smoke/final/
models/biencoder_v1_smoke/final/training_summary.json
```

Do not interpret the smoke model's retrieval quality. Its only purpose is to catch
CUDA, bf16, tokenizer, package, and trainer failures before the full run.

## Step 2: Full v1 run

The script automatically:

1. builds one query-balanced random-negative triplet per eligible query;
2. audits the triplets;
3. fine-tunes `all-MiniLM-L6-v2` for one epoch;
4. scores the unchanged validation candidate pool;
5. performs a paired bootstrap comparison against the pretrained dense model.

```bash
BASELINE_PER_QUERY=outputs/dense_pretrained_val.per_query.parquet \
BATCH_SIZE=128 \
PRECISION=bf16 \
EPOCHS=1 \
bash scripts/run_phase2_biencoder_v1.sh \
  2>&1 | tee outputs/phase2_biencoder_v1.log
```

If your baseline filename is `outputs/dense_val.per_query.parquet`, either omit
`BASELINE_PER_QUERY` and let the script detect it, or set that path explicitly.

### RTX 3090 fallback

Try the physical batch size 128 first. If CUDA runs out of memory:

```bash
BASELINE_PER_QUERY=outputs/dense_pretrained_val.per_query.parquet \
BATCH_SIZE=96 \
bash scripts/run_phase2_biencoder_v1.sh \
  2>&1 | tee outputs/phase2_biencoder_v1_b96.log
```

If 96 still fails, use 64. Do not compensate with gradient accumulation for this
experiment: it changes optimizer accumulation but does not enlarge the per-step
in-batch negative pool used by standard MNRL.

## Step 3: Inspect outputs

```bash
cat data/processed/biencoder_pairs_v1.summary.json
cat data/processed/biencoder_pairs_v1.audit.json
cat models/biencoder_v1_random/final/training_summary.json
cat outputs/biencoder_v1_random_val.metrics.json
cat outputs/biencoder_v1_random_val.vs_pretrained.json
```

Pair audit must report:

```json
{
  "same_product_rows": 0,
  "same_text_rows": 0,
  "negative_gain_ge_positive_gain_rows": 0,
  "empty_anchor_rows": 0,
  "empty_positive_rows": 0,
  "empty_negative_rows": 0,
  "duplicate_triplet_rows": 0,
  "passed": true
}
```

The number of generated rows may be below 17,755 because a query must have both an
eligible positive (`E`, or `S` only when no `E` exists) and an explicit `I` or `C`
negative. Record the coverage ratio:

```text
training_pair_queries / 17,755
```

## Decision rule

Use `outputs/biencoder_v1_random_val.vs_pretrained.json` rather than comparing two
independent confidence intervals.

### Green: proceed to hard-negative mining

- `ndcg@10.mean_delta > 0`, and
- `ndcg@10.paired_ci95_low > 0`.

A practically useful first-stage gain is approximately +0.005 absolute nDCG@10 or
larger, but statistical evidence and secondary metrics should be reported as well.

### Yellow: informative but not yet a win

- mean delta is near zero, or
- the paired 95% interval crosses zero.

Keep the run as the random-negative ablation and proceed to hard-negative mining.
Do not tune on the test split.

### Red: diagnose before hard negatives

- nDCG@10 falls by more than 0.005, or
- exact recall and MRR both fall materially.

Inspect:

1. triplet coverage and label distribution;
2. training loss for NaN or instability;
3. bf16 support and tokenizer truncation;
4. whether a stale embedding cache was accidentally reused;
5. a lower learning rate (`1e-5`) as the first controlled retry.

## Files to preserve for the project report

```text
data/processed/biencoder_pairs_v1.summary.json
data/processed/biencoder_pairs_v1.audit.json
models/biencoder_v1_random/final/training_summary.json
outputs/phase2_biencoder_v1.log
outputs/biencoder_v1_random_val.metrics.json
outputs/biencoder_v1_random_val.per_query.parquet
outputs/biencoder_v1_random_val.vs_pretrained.json
```

These files establish the random-negative v1 ablation. Do not overwrite them when
starting the hard-negative v2 run.
