# Phase 8: Freeze validation-selected routing policy (Step 14)

This patch does **not** rerun training, retune a threshold or evaluate official test.
It reads the exact Phase 7 winner (`entropy_per_cost`), retrieves its **training-calibrated** threshold from the Phase 6 sweep, freezes the Phase 6 cost interpolation parameters, verifies the validation result via replay, and fingerprints the two champion model checkpoints.

## Install

```bash
unzip -o phase8-freeze-policy.zip -d .
pytest -q tests/test_phase8_freeze_policy.py
```

## Freeze exactly once

```bash
bash scripts/run_phase8_freeze_policy.sh \
  2>&1 | tee outputs/phase8_freeze.log
```

Expected frozen file:

```text
models/router_entropy_per_cost_frozen/policy.json
```

The script verifies validation nDCG@10, actual rerank rate and GPU-stage P95 against the selected Phase 6 sweep row. It will **fail closed** on mismatches, missing models, invalid cost samples, or changed code. It refuses to overwrite the policy without explicit `--force`.

The checkpoint fingerprints bind the policy to the exact current model file contents. For long-term reproducibility, archive the frozen policy **and both actual checkpoints**, not only this JSON file.

## Verify after GitHub sync

```bash
python -m budget_aware_search.routing.freeze_policy verify \
  --policy models/router_entropy_per_cost_frozen/policy.json
```

## Step 15 interface (do not run until final TEST evaluation is planned)

After your frozen bi-encoder scores official test candidates, use:

```bash
python -m budget_aware_search.routing.freeze_policy apply \
  --policy models/router_entropy_per_cost_frozen/policy.json \
  --bi-scores outputs/biencoder_v1_random_test.parquet \
  --output outputs/frozen_router_test_decisions.parquet
```

The apply command never loads CE scores or relevance labels and never reselects a threshold. It yields a `query_id`-level decision table with `rerank` and `routing_priority`. For the final system, Cross-Encoder scoring is needed **only** for routed queries. Latency must be measured independently on test; the validation 20ms P95 is **not a guaranteed test/serving SLA**.

## Reproducibility / limitations

- `threshold` is from TRAIN-score calibration, selected as a validation operating point.
- Entropy uses the *same code* and temperature (0.1) as Phase 5/6.
- The CE cost model is a 5-candidate-bin median interpolation trained on the Phase 6 **validation GPU timing subset**. This contains no relevance labels; the cost model is frozen as numeric interpolation arrays.
- Model and router code SHA256 values are checked at apply time.
- Candidate-pool scoring and measured-stage replay do not include FAISS retrieval, network, scheduling or concurrent serving effects.
- The 20ms stage P95 on the validation sample is an exploratory budget; it can change for test traffic. No test-derived adjustment is permitted.
- This is an **offline query router**, not an enforcing real-time 20ms latency guard. Its target rerank fraction (0.45) is not a guaranteed fraction on unseen test queries.
