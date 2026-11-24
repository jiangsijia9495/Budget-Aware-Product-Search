# Phase 15: FP16 repeated-inference consistency hotfix

This corrects the overly strict `atol=rtol=1e-4` score equality gate between
routed-only and full candidate-pool CE predictions.

Those two inference passes contain different batches and padding; with fp16,
tiny score differences are expected. The test quality must use actual routed-CE
scores for routed queries, never quietly substitute full CE scores.

## Install and run

```bash
unzip -o phase15-fp16-repeatability-hotfix.zip -d .
pytest -q tests/test_phase15_ce_repeatability.py
bash scripts/resume_phase15_finalize.sh \
  2>&1 | tee outputs/phase15_finalize_resume.log
```

The script verifies the frozen policy again and runs **only** Step 8. Do not rerun
model training or inference. No test-derived routing threshold adjustment.

## Before considering the test result final

Inspect:

```bash
cat outputs/final_test/ce_inference_repeatability.json
cat outputs/final_test/final_report.json
```

If consistency fails, `ce_inference_repeatability.json` still records the
magnitude and rank impact of the differences. For such a failure, investigate
frozen checkpoint, `max-length=256`, input text normalization, model version,
precision, and old cached predictions. Do not increase the tolerances to get a
passing test without diagnosing the root cause.

The 3 numeric tolerances are fixed before inspecting the user's test predictions:
mean_abs <= 0.0005, p99_abs <= 0.005, max_abs <= 0.02. These are practical,
conservative checks for 0-1 expected-gain outputs, not a guarantee of complete
bitwise reproducibility. Log top-1 and top-10-set differences separately.
