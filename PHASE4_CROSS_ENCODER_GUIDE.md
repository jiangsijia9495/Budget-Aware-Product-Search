# Phase 4: Cross-Encoder reranking

## Controlled experiment

- Fixed validation set: 3,133 queries.
- Champion baseline: `biencoder_v1_random`.
- Cross-Encoder input: `(query, product_text)`.
- Training target: four ESCI classes in the fixed order `E, S, C, I`.
- Ranking score:

```text
P(E) + 0.1 * P(S) + 0.01 * P(C)
```

- Primary comparison: paired query bootstrap on nDCG@10.

## Install

From the project root:

```bash
unzip -o cross_encoder_phase4_patch.zip -d .
pip install -e ".[ml,dev]"
pytest -q tests/test_cross_encoder_metrics.py
```

## Smoke test

```bash
bash scripts/run_phase4_cross_encoder_smoke.sh \
  2>&1 | tee outputs/cross_encoder_smoke.log
```

The smoke test is only a compatibility check; do not use its metric as an experiment result.

## Full experiment

```bash
bash scripts/run_phase4_cross_encoder.sh \
  2>&1 | tee outputs/cross_encoder_phase4_full.log
```

For an RTX 3090, the default training batch size is 64. If training reports CUDA OOM:

```bash
TRAIN_BATCH_SIZE=32 \
  bash scripts/run_phase4_cross_encoder.sh \
  2>&1 | tee outputs/cross_encoder_phase4_b32.log
```

Do not change the learning rate or number of epochs in the same retry.

## Decision rule

Keep the fine-tuned Cross-Encoder as a reranker when the comparison against v1 has:

```text
mean_delta > 0
paired_ci95_low > 0
```

Also verify that Exact Recall@5 does not materially regress. Do not run the test split yet.
