#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${ROOT_DIR}/vendor/esci-data"
RAW_DIR="${ROOT_DIR}/data/raw"

# command -v git >/dev/null 2>&1 || { echo "git is required" >&2; exit 1; }
# command -v git-lfs >/dev/null 2>&1 || {
#   echo "git-lfs is required. On Ubuntu: sudo apt-get install git-lfs" >&2
#   exit 1
# }

# git lfs install
# mkdir -p "${ROOT_DIR}/vendor" "${RAW_DIR}"
# if [[ ! -d "${VENDOR_DIR}/.git" ]]; then
#   git clone https://github.com/amazon-science/esci-data.git "${VENDOR_DIR}"
# fi
# (
#   cd "${VENDOR_DIR}"
#   git pull --ff-only
#   git lfs pull
# )

EXAMPLES_PATH="$(find "${VENDOR_DIR}" -name shopping_queries_dataset_examples.parquet -print -quit)"
PRODUCTS_PATH="$(find "${VENDOR_DIR}" -name shopping_queries_dataset_products.parquet -print -quit)"
SOURCES_PATH="$(find "${VENDOR_DIR}" -name shopping_queries_dataset_sources.csv -print -quit || true)"

if [[ -z "${EXAMPLES_PATH}" || -z "${PRODUCTS_PATH}" ]]; then
  echo "Could not find ESCI parquet files after cloning ${VENDOR_DIR}." >&2
  echo "Inspect the official repository README for any updated download instructions." >&2
  exit 1
fi

cp -f "${EXAMPLES_PATH}" "${RAW_DIR}/shopping_queries_dataset_examples.parquet"
cp -f "${PRODUCTS_PATH}" "${RAW_DIR}/shopping_queries_dataset_products.parquet"
if [[ -n "${SOURCES_PATH}" ]]; then
  cp -f "${SOURCES_PATH}" "${RAW_DIR}/shopping_queries_dataset_sources.csv"
fi

echo "ESCI files copied to ${RAW_DIR}"
