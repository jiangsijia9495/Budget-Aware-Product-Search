# Experiment Log

## Identification

- Date: 2026-11-18
- Git commit: Not recorded
- Experiment name: Phase 1 validation baselines
- Owner: Not recorded

## Hypothesis

Pretrained dense retrieval should outperform BM25 and the source-order baseline on the ESCI validation set, measured primarily by mean nDCG@10. BM25 should also outperform source-order.

## Data

- Dataset release/source: Amazon Shopping Queries Dataset / ESCI; product and query files in `data/raw/`
- Locale/version: `us`; `use_small_version=true`
- Train/validation/test query counts: 17,755 / 3,133 / 8,956
- Candidate counts: mean 19.944 candidates per validation query; 62,486 validation rows
- Any filtering: Official train/validation/test split generated with seed 42 and validation fraction 0.15; no additional candidate filtering reported
- Leakage checks performed: Query-level overlap checks are implemented between splits; dataset generation completed without reporting overlap

## Model

- Model name and exact revision: `sentence-transformers/all-MiniLM-L6-v2`, cache revision `d46f7701`; source-order and BM25 baselines also included
- License: Dense model is distributed under the Apache 2.0 license; source-order/BM25 have no separate model license
- Max sequence length: 1,600 characters for generated product text
- Product text fields: `product_title`, `product_brand`, `product_color`, `product_bullet_point`, `product_description`
- Loss: Not applicable for pretrained dense inference or baselines
- Negative sampling: Not applicable
- Precision: Dense embeddings generated on CUDA; BM25 uses a full-precision sparse scikit-learn implementation

## Training

- GPU: RTX 3090
- Epochs/steps: Not applicable; pretrained dense model is used without task-specific fine-tuning
- Batch size: Dense inference batch size 512
- Gradient accumulation: Not applicable
- Learning rate: Not applicable
- Warmup: Not applicable
- Wall-clock time: Not recorded
- Peak GPU memory: Not recorded

## Evaluation

- Score definition: Source-order uses candidate order, BM25 uses weighted sparse lexical score, and dense retrieval uses cosine similarity between normalized embeddings
- ESCI gain mapping: E=1.0, S=0.1, C=0.01, I=0.0
- Primary metric: Mean nDCG@10
- Secondary metrics: Pairwise accuracy, MRR exact, exact recall@5/@10/@20, exact success@5/@10/@20, nDCG@5/@20
- Bootstrap samples: 1,000 query bootstrap samples
- Statistical comparison: Dense nDCG@10 was 0.709680 (95% CI 0.702532–0.717026), BM25 was 0.691749 (95% CI 0.684125–0.698975), and source-order was 0.579276 (95% CI 0.571354–0.586756). The dense and BM25 intervals do not overlap

## Systems

- Hardware/CPU: RTX 3090 GPU; CPU details not recorded
- Threads: Not recorded
- Batch size: Dense inference 512; BM25 index fitting/ranking settings `k1=1.5`, `b=0.75`, `min_df=1`
- Tokenization included?: Yes, dense uses the model tokenizer and BM25 uses the CountVectorizer; source-order uses existing candidate order
- p50/p95/p99 latency: Not recorded
- QPS: Not recorded
- Memory/index size: BM25 index at `models/bm25/`; dense embeddings/cache at `models/dense_cache/`

## Results

| Model | nDCG@10 | 95% CI | Exact R@5 | P95 ms | Notes |
|---|---:|---:|---:|---:|---|
| Source-order | 0.579276 | 0.571354–0.586756 | 0.288860 | — | Weakest baseline |
| BM25 | 0.691749 | 0.684125–0.698975 | 0.404737 | — | +0.112474 nDCG@10 versus source-order |
| Dense pretrained | 0.709680 | 0.702532–0.717026 | 0.420317 | — | +0.017930 nDCG@10 versus BM25; batch size 512 |

## Failure analysis

- Query slice 1: No query slicing was performed in this run
- Query slice 2: No query slicing was performed in this run
- Query slice 3: No query slicing was performed in this run
- Representative failures: No representative failure inspection was performed

## Decision

- Keep / reject / revise: Keep the pretrained dense retriever as the strongest Phase 1 baseline and continue with training only if project targets require further gains
- Evidence: On 3,133 validation queries, dense pretrained retrieval achieved nDCG@10 0.709680, exceeding BM25 0.691749 by 0.017930 and source-order 0.579276 by 0.130404. Dense also had the highest pairwise accuracy (0.634248), MRR exact (0.764801), exact recall@5 (0.420317), and exact success@10 (97.83%)
- Next experiment: Train or distill a biencoder/cross-encoder and compare against these baselines using the identical 3,133-query validation split, gain mapping, 1,000-sample bootstrap, and the same candidate set
