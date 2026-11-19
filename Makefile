PYTHON ?= python3
CONFIG ?= configs/base.yaml

.PHONY: setup test prepare bm25-val dense-val pairs train-biencoder train-cross score-cross clean

setup:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip && pip install -e ".[ml,dev]"

test:
	$(PYTHON) -m pytest

prepare:
	$(PYTHON) -m budget_aware_search.data --config $(CONFIG)

bm25-val:
	$(PYTHON) -m budget_aware_search.baselines.bm25 --processed-dir data/processed --split val --index-dir models/bm25 --output outputs/bm25_val.parquet

dense-val:
	$(PYTHON) -m budget_aware_search.baselines.dense --processed-dir data/processed --split val --model sentence-transformers/all-MiniLM-L6-v2 --device cuda --output outputs/dense_val.parquet

pairs:
	$(PYTHON) -m budget_aware_search.training.build_pairs --input data/processed/train.parquet --output data/processed/biencoder_pairs.parquet

train-biencoder:
	$(PYTHON) -m budget_aware_search.training.train_biencoder --pairs data/processed/biencoder_pairs.parquet --output-dir models/biencoder

train-cross:
	$(PYTHON) -m budget_aware_search.training.train_cross_encoder --train data/processed/train.parquet --validation data/processed/val.parquet --output-dir models/cross_encoder

score-cross:
	$(PYTHON) -m budget_aware_search.training.score_cross_encoder --input data/processed/val.parquet --model models/cross_encoder/final --output outputs/cross_val.parquet

clean:
	rm -rf data/processed/* outputs/* models/*
	touch data/processed/.gitkeep outputs/.gitkeep models/.gitkeep
