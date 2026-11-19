#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/data/sqid"
mkdir -p "${OUTPUT_DIR}"

if ! command -v hf >/dev/null 2>&1; then
  echo "The Hugging Face CLI is required. Install project ML dependencies first." >&2
  exit 1
fi

hf download crossingminds/shopping-queries-image-dataset \
  --repo-type dataset \
  --include "data/product_features.parquet" \
  --include "data/query_features.parquet" \
  --include "data/product_image_urls.parquet" \
  --local-dir "${OUTPUT_DIR}"

echo "SQID files downloaded under ${OUTPUT_DIR}/data"
