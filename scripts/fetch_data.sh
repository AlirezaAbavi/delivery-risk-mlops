#!/usr/bin/env bash
# Download the real Olist Brazilian E-Commerce dataset from Kaggle.
#
# This is OPTIONAL. The repo ships a synthetic sample dataset (sample_data/) so the whole
# stack runs offline with no credentials. Use this only if you want to train/evaluate on
# the full ~100k-order dataset (`make bootstrap DATA=full`).
#
# Requires the Kaggle CLI and API credentials:
#   pip install kaggle
#   place kaggle.json at ~/.kaggle/kaggle.json  (chmod 600)
#   (or export KAGGLE_USERNAME and KAGGLE_KEY)
#
# The files land in olist_data/olist_data/ (gitignored), matching RAW_DATA_DIR for
# `make bootstrap DATA=full`.
set -euo pipefail

DEST="olist_data/olist_data"
DATASET="olistbr/brazilian-ecommerce"

have_creds() {
  [ -f "${HOME}/.kaggle/kaggle.json" ] || { [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; }
}

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Install it with:  pip install kaggle"
  echo "Skipping full-data download — the committed sample_data/ still works with:  make bootstrap"
  exit 0
fi

if ! have_creds; then
  echo "No Kaggle credentials found (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY)."
  echo "See https://www.kaggle.com/docs/api for how to create an API token."
  echo "Skipping full-data download — the committed sample_data/ still works with:  make bootstrap"
  exit 0
fi

mkdir -p "${DEST}"
echo ">> downloading ${DATASET} into ${DEST}/ ..."
kaggle datasets download -d "${DATASET}" -p "${DEST}" --unzip
echo ">> done. Run the full pipeline with:  make bootstrap DATA=full"
