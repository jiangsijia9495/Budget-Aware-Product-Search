# Budget-Aware Product Search

一个面向 Google Search / Ads / Shopping / Discover / Recommendation team match 的完整工程项目：

- lexical retrieval：source order、BM25；
- dense representation：预训练和 fine-tuned bi-encoder；
- ranking：四分类 Cross-Encoder，按 ESCI expected gain 排序；
- hard-negative mining；
- teacher-student distillation；
- adaptive reranking：在 latency budget 下决定哪些 query 调用昂贵模型；
- multimodal extension：使用 SQID 的 CLIP text/image features；
- full-catalog retrieval：FAISS；
- C++17 latency benchmark：FAISS + ONNX Runtime。

最终项目应回答一个明确问题：

> 在限定 P95 latency / average compute budget 的条件下，怎样组合 lexical retrieval、dense retrieval、Cross-Encoder reranking、distillation 和 uncertainty-aware routing，以获得更好的 product-search quality–efficiency Pareto frontier？

---

## 1. 评估协议：先把问题做对

### Track A：ESCI candidate-pool reranking

ESCI Task 1 为每个 query 提供最多约 40 个候选及 E/S/C/I relevance judgment。这个 track 的目标是重新排序已有候选，主指标为：

```text
nDCG@10 with gains:
E = 1.00
S = 0.10
C = 0.01
I = 0.00
```

这是最严谨、最适合模型比较的主 track。

### Track B：full-catalog retrieval

将 query 在全部商品 embedding 中做 FAISS top-K 检索。ESCI 只标注了原始候选集合，因此 catalog 中未标注商品是 **unknown**，不能直接当 irrelevant。

本项目只报告：

- known Exact recall lower bound；
- judged fraction@K；
- lower-bound nDCG；
- condensed nDCG；
- latency、QPS、memory、index size。
- 

### Track C：SQID multimodal custom split

SQID 的主要 image features 对应 ESCI 官方 test subset。因此，若用这些 label 训练 fusion model，必须在该 subset 内重新做 query-disjoint train/validation/test split，并明确写：

> These custom-split metrics are not directly comparable with the official ESCI test benchmark.

---

## 2. 工程结构

```text
budget-aware-product-search/
├── configs/base.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── sqid/
├── models/
├── outputs/
├── reports/
├── scripts/
├── serving_cpp/
│   ├── CMakeLists.txt
│   ├── include/
│   └── src/
├── src/budget_aware_search/
│   ├── baselines/
│   ├── multimodal/
│   ├── routing/
│   ├── serving/
│   └── training/
└── tests/
```

---

## 3. 环境安装

Linux、Python 3、NVIDIA RTX 4090。

```bash
cd /budget-aware-product-search
conda create -n budget python=3.11 pip -y
conda activate budget
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ".[ml,dev]"
```

检查：

```bash
python scripts/check_environment.py
pytest -q
```

所有 unit tests 应通过。

---

## 4. 下载 ESCI

```bash
bash scripts/download_esci.sh
```

成功后应出现：

```text
data/raw/shopping_queries_dataset_examples.parquet
data/raw/shopping_queries_dataset_products.parquet
```

若官方仓库的下载方式发生变化，手动下载这两个文件并放到上述路径即可。

---

# Milestone 1：可运行 baseline

## 5. 先跑 300-query debug

```bash
bash scripts/run_phase1_debug.sh
```

它会依次完成：

prepare query-disjoint data
        ↓
source-order baseline
        ↓
BM25 baseline
        ↓
pretrained dense baseline
        ↓
uniform nDCG evaluation

查看：

```bash
cat data/processed/dataset_summary.json
cat outputs/source_order_val.metrics.json
cat outputs/bm25_val.metrics.json
cat outputs/dense_val.metrics.json
```

### 正确性检查

- train、val、test 的 `query_id` 不重叠；
- perfect ordering 的测试 nDCG 应为 1；
- BM25 至少应能为 lexical query 产生非零分数；
- dense score 不应全部相同。

---

## 6. 生成完整数据

确认 debug 成功后，删除 debug outputs，生成完整 split：

```bash
python -m budget_aware_search.data \
  --config configs/base.yaml \
  --output-dir data/processed
```

查看统计：

```bash
cat data/processed/dataset_summary.json
```

**从此以后只在 validation 上调参。官方 test 只在模型和 threshold 冻结后运行。**

---

## 7. Source-order baseline

```bash
python -m budget_aware_search.baselines.source_order \
  --input data/processed/val.parquet \
  --output outputs/source_order_val.parquet \
  --bootstrap-samples 1000
```

它衡量 Amazon 原候选顺序本身有多强，避免拿 neural model 与 random baseline 比较而夸大结果。

---

## 8. BM25 baseline

```bash
python -m budget_aware_search.baselines.bm25 \
  --processed-dir data/processed \
  --split val \
  --index-dir models/bm25 \
  --output outputs/bm25_val.parquet \
  --refit \
  --bootstrap-samples 1000
```

记录：

- vocabulary size；
- index build time；
- nDCG@5/10/20；
- Exact Recall@5/10；
- query-level bootstrap CI。

---

## 9. Pretrained dense baseline

快速模型：

```bash
python -m budget_aware_search.baselines.dense \
  --processed-dir data/processed \
  --split val \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --device cuda \
  --batch-size 512 \
  --cache-dir models/dense_cache_pretrained \
  --reencode-catalog \
  --output outputs/dense_pretrained_val.parquet \
  --bootstrap-samples 1000 \
```

可再增加一个更强但更慢的 baseline，例如 BGE 或 MPNet。模型名称和 license 必须记录在 experiment log 中。

---

# Milestone 2：Fine-tuned bi-encoder

## 10. 构造训练 triples

```bash
python -m budget_aware_search.training.build_pairs \
  --input data/processed/train.parquet \
  --output data/processed/biencoder_pairs_v1.parquet \
  --max-positives-per-query 1 \
  --negatives-per-positive 1 \
  --negative-label-priority I,C \
  --seed 42
```

audit：


```bash
python -m budget_aware_search.training.audit_pairs \
  --pairs data/processed/biencoder_pairs_v1.parquet
```

初始正负样本策略：

- positive：优先 E；没有 E 时允许 S；
- negative：优先 I，再 C；
- 每个 query 默认只选一个 positive，避免训练 batch 中大量重复 query。

---

## 11. Fine-tune bi-encoder

```bash
python -m budget_aware_search.training.train_biencoder \
  --pairs data/processed/biencoder_pairs_v1.parquet \
  --base-model sentence-transformers/all-MiniLM-L6-v2 \
  --output-dir models/biencoder_v1_random \
  --epochs 1 \
  --batch-size 128 \
  --learning-rate 2e-5 \
  --max-seq-length 256 \
  --precision bf16 \
  --loss mnrl \
  --seed 42 \
  2>&1 | tee outputs/biencoder_v1_random.train.log
```

4090 显存不足时：

```bash
--batch-size 64 --gradient-accumulation-steps 2
```

验证：

```bash
python -m budget_aware_search.baselines.dense \
  --processed-dir data/processed \
  --split val \
  --model models/biencoder_v1_random/final \
  --device cuda \
  --batch-size 512 \
  --cache-dir models/dense_cache \
  --reencode-catalog \
  --output outputs/biencoder_v1_random_val.parquet \
  --bootstrap-samples 1000 \
  2>&1 | tee outputs/biencoder_v1_random_val.log
```

只有 validation 优于 pretrained baseline 才进入下一步。

做 paired bootstrap comparison:

```bash
python -m budget_aware_search.evaluation.compare_rankings \
  --baseline outputs/dense_pretrained_val.per_query.parquet \
  --candidate outputs/biencoder_v1_random_val.per_query.parquet \
  --output outputs/biencoder_v1_random_val.vs_pretrained.json \
  --bootstrap-samples 5000
```

---

## 12. Hard-negative mining

先在 train candidate pool 上打分：

```bash
python -m budget_aware_search.baselines.dense \
  --processed-dir data/processed \
  --split train \
  --model models/biencoder_v1/final \
  --device cuda \
  --cache-dir models/dense_cache \
  --output outputs/biencoder_v1_train.parquet
```

挖 highest-scoring false candidates：

```bash
python -m budget_aware_search.training.mine_hard_negatives \
  --scored-train outputs/biencoder_v1_train.parquet \
  --output data/processed/biencoder_hard_pairs.parquet \
  --negatives-per-positive 2
```

再训练 v2：

```bash
python -m budget_aware_search.training.train_biencoder \
  --pairs data/processed/biencoder_hard_pairs.parquet \
  --base-model models/biencoder_v1/final \
  --output-dir models/biencoder_v2_hard \
  --epochs 1 \
  --batch-size 128 \
  --precision bf16
```

必须做 ablation：

```text
pretrained
vs random negatives
vs hard negatives
```

---

# Milestone 3：Cross-Encoder reranker

## 13. 训练四分类 reranker

模型预测 I/C/S/E 四类概率，然后按 expected gain 排序：

```text
score = P(E) * 1.00 + P(S) * 0.10 + P(C) * 0.01
```

训练：

```bash
python -m budget_aware_search.training.train_cross_encoder \
  --train data/processed/train.parquet \
  --validation data/processed/val.parquet \
  --base-model microsoft/MiniLM-L12-H384-uncased \
  --output-dir models/cross_encoder_v1 \
  --epochs 1 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --precision bf16 \
  --max-length 256
```

显存不足：

```bash
--batch-size 16 --gradient-accumulation-steps 2
```

验证打分：

```bash
python -m budget_aware_search.training.score_cross_encoder \
  --input data/processed/val.parquet \
  --model models/cross_encoder_v1/final \
  --device cuda \
  --batch-size 256 \
  --output outputs/cross_encoder_v1_val.parquet \
  --bootstrap-samples 1000
```

Cross-Encoder 应比 bi-encoder 更准确但更慢。若没有提升，优先检查：

- label mapping；
- class imbalance；
- product text truncation；
- learning rate；
- query leakage；
- output 是否使用 expected gain 而非 class id。

---

# Milestone 4：Budget-aware adaptive reranking

## 14. 在 validation 上拟合 router

base 使用 bi-encoder，reranker 使用 Cross-Encoder：

```bash
python -m budget_aware_search.routing.adaptive fit \
  --base outputs/biencoder_v2_val.parquet \
  --reranker outputs/cross_encoder_v1_val.parquet \
  --policy-output models/router_margin.json \
  --curve-output reports/router_val_curve.csv \
  --base-latency-ms 1.0 \
  --reranker-latency-ms 8.0 \
  --latency-budget-ms 5.0
```

最初可以用估计 latency 调试，但最终报告必须替换成真实 benchmark。

router 使用 base top-1/top-2 margin：

```text
low margin -> uncertain -> call reranker
high margin -> accept inexpensive base ranking
```

画图：

```text
x-axis: reranking rate or estimated P95 latency
y-axis: nDCG@10
```

这张 Pareto curve 是项目最重要的结果图。

---

## 15. 冻结 policy 后应用到 test

先产生 test base/reranker predictions，然后：

```bash
python -m budget_aware_search.routing.adaptive apply \
  --base outputs/biencoder_v2_test.parquet \
  --reranker outputs/cross_encoder_v1_test.parquet \
  --policy models/router_margin.json \
  --output outputs/adaptive_test.parquet \
  --bootstrap-samples 2000
```

不要用 test 重新选择 threshold。

---

# Milestone 5：Knowledge distillation

## 16. 用 Cross-Encoder teacher score 蒸馏 bi-encoder

先为 train 生成 teacher scores：

```bash
python -m budget_aware_search.training.score_cross_encoder \
  --input data/processed/train.parquet \
  --model models/cross_encoder_v1/final \
  --device cuda \
  --batch-size 256 \
  --output outputs/cross_teacher_train.parquet
```

蒸馏：

```bash
python -m budget_aware_search.training.distill_biencoder \
  --teacher-scores outputs/cross_teacher_train.parquet \
  --base-model models/biencoder_v2_hard/final \
  --output-dir models/biencoder_distilled \
  --epochs 1 \
  --batch-size 128 \
  --precision bf16
```

比较：

```text
bi-encoder v1
bi-encoder + hard negatives
bi-encoder + teacher distillation
Cross-Encoder teacher
adaptive cascade
```

项目价值不在于“做了 distillation”，而在于量化：

```text
retained % of teacher nDCG
latency reduction
memory reduction
reranking-rate reduction
```

---

# Milestone 6：SQID multimodal extension

## 17. 下载 SQID

```bash
bash scripts/download_sqid.sh
```

主要文件：

```text
data/sqid/data/product_features.parquet
data/sqid/data/query_features.parquet
```

## 18. 自定义 query-disjoint fusion experiment

先对 ESCI official test 产生一个 frozen text model score，然后：

```bash
python -m budget_aware_search.multimodal.fuse_sqid \
  --base-predictions outputs/cross_encoder_v1_test.parquet \
  --product-features data/sqid/data/product_features.parquet \
  --query-features data/sqid/data/query_features.parquet \
  --output-dir outputs/sqid_fusion \
  --seed 42
```

比较：

- frozen text ranker；
- CLIP query–product text similarity；
- CLIP query–product image similarity；
- learned fusion。

重点 error slices：

- color；
- material；
- shape/style；
- branded exact queries；
- queries where text and image disagree；
- products with missing or default images。

---

# Milestone 7：Full-catalog FAISS retrieval

## 19. Python FAISS diagnostic

Dense baseline 已保存 catalog embeddings。指定对应 cache 文件：

```bash
python -m budget_aware_search.baselines.faiss_retrieval \
  --judgments data/processed/val.parquet \
  --catalog-embeddings models/dense_cache/<MODEL_CACHE>/catalog_embeddings.npy \
  --catalog-ids models/dense_cache/<MODEL_CACHE>/catalog_product_ids.npy \
  --model models/biencoder_distilled/final \
  --device cuda \
  --index hnsw \
  --hnsw-m 32 \
  --ef-search 128 \
  --top-k 100 \
  --output-dir outputs/full_catalog_hnsw
```

不要将 unjudged products 当成 irrelevant。README 和最终报告必须保留 incomplete-judgment warning。

---

# Milestone 8：C++17 serving benchmark

## 20. 导出 normalized embedding matrices

```bash
python -m budget_aware_search.serving.export_embeddings \
  --catalog data/processed/catalog.parquet \
  --queries data/processed/val.parquet \
  --model models/biencoder_distilled/final \
  --device cuda \
  --output-dir serving_artifacts/embeddings
```

## 21. 编译 FAISS C++ benchmark

安装 FAISS C++ development package，随后：

```bash
cmake -S serving_cpp -B serving_cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ROOT=/path/to/faiss
cmake --build serving_cpp/build -j
```

运行 exact search：

```bash
./serving_cpp/build/faiss_benchmark \
  --catalog serving_artifacts/embeddings/catalog.f32mat \
  --queries serving_artifacts/embeddings/queries.f32mat \
  --index flat \
  --top-k 100 \
  --threads 1
```

运行 HNSW：

```bash
./serving_cpp/build/faiss_benchmark \
  --catalog serving_artifacts/embeddings/catalog.f32mat \
  --queries serving_artifacts/embeddings/queries.f32mat \
  --index hnsw \
  --hnsw-m 32 \
  --ef-search 128 \
  --top-k 100 \
  --threads 1
```

报告：

- index build time；
- p50/p95/p99 per-query latency；
- sequential QPS；
- exact-vs-HNSW recall@K；
- memory/index size；
- varying `efSearch` 的 quality-latency curve。

---

## 22. ONNX Cross-Encoder benchmark

将训练好的 Cross-Encoder 导出为 text-classification ONNX：

```bash
optimum-cli export onnx \
  --model models/cross_encoder_v1/final \
  --task text-classification \
  serving_artifacts/cross_encoder_onnx
```

准备固定长度 token inputs：

```bash
python -m budget_aware_search.serving.prepare_onnx_inputs \
  --input data/processed/val.parquet \
  --tokenizer models/cross_encoder_v1/final \
  --output-dir serving_artifacts/onnx_inputs \
  --max-length 256 \
  --max-rows 10000
```

用 ONNX Runtime C++ package 重新配置：

```bash
cmake -S serving_cpp -B serving_cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ROOT=/path/to/faiss \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build serving_cpp/build -j
```

运行：

```bash
./serving_cpp/build/onnx_benchmark \
  --model serving_artifacts/cross_encoder_onnx/model.onnx \
  --input-dir serving_artifacts/onnx_inputs \
  --threads 1 \
  --max-rows 10000
```

注意：当前 C++ benchmark 使用 Python 预先 tokenized 的 inputs，以隔离模型执行 latency。最终报告中应把：

```text
tokenization latency
model latency
FAISS latency
end-to-end latency
```

分开报告，不要把 model-only latency 冒充 end-to-end serving latency。

---

# 23. 必做 ablation

至少完成：

1. title-only vs structured product text；
2. source order vs BM25 vs pretrained dense；
3. random negatives vs hard negatives；
4. bi-encoder vs Cross-Encoder；
5. no distillation vs distillation；
6. always rerank vs adaptive rerank；
7. text-only vs text+image；
8. FlatIP vs HNSW；
9. FP32 vs ONNX INT8（CPU）；
10. query slice and failure analysis。

---

# 24. 结果表模板

```text
Model                         nDCG@10   Exact R@5   P95 ms   % reranked
Source order                  ...       ...         ...      0%
BM25                          ...       ...         ...      0%
Pretrained bi-encoder         ...       ...         ...      0%
Fine-tuned bi-encoder         ...       ...         ...      0%
+ hard negatives              ...       ...         ...      0%
+ distillation                ...       ...         ...      0%
Cross-Encoder                 ...       ...         ...      100%
Adaptive cascade              ...       ...         ...      ...%
Multimodal fusion             ...       ...         ...      ...%
```

所有主结果都应给 query-level bootstrap 95% CI。

---

# 25. 最终简历 bullet 模板

只有填入真实结果后才能使用：

> Built a two-stage product-search system on **[N]** real query–product judgments, fine-tuning a dense bi-encoder with hard negatives and a 4-class Cross-Encoder reranker; improved candidate-pool nDCG@10 from **[A]** to **[B]** over BM25 and pretrained embedding baselines.

> Designed a confidence-based adaptive reranking policy that invoked the expensive model for **[R]%** of queries while retaining **[Q]%** of always-rerank quality, reducing measured p95 inference latency by **[L]%**.

> Exported model and embedding artifacts for a C++17 FAISS/ONNX Runtime benchmark, evaluating exact and HNSW retrieval across p50/p95/p99 latency, QPS, memory, ANN recall, and INT8 quality–efficiency trade-offs.

---

# 26. 第一天只做这些

```bash
source .venv/bin/activate
python scripts/check_environment.py
pytest -q
bash scripts/download_esci.sh
bash scripts/run_phase1_debug.sh
```

第一天不要训练 Cross-Encoder，不要下载 SQID，不要写 C++。先确保数据协议、split、metric 和两个 baseline 全部正确。
