#!/usr/bin/env bash
set -euo pipefail

ZIP_PATH="${1:-data/zips/BraTS2024-BraTS-GLI-TrainingData.zip}"
OUT_DIR="${2:-data/raw/brats2024_gli}"
LOG_DIR="data/cache/logs"

mkdir -p "$OUT_DIR" "$LOG_DIR"

if [ ! -f "$ZIP_PATH" ]; then
  echo "[ERR] zip not found: $ZIP_PATH" >&2
  exit 1
fi

echo "### 1) sha256"
sha256sum "$ZIP_PATH" | tee "${ZIP_PATH}.sha256"

echo
echo "### 2) unzip integrity test"
unzip -t "$ZIP_PATH" | tee "${LOG_DIR}/unzip_test_$(date +%Y%m%d_%H%M%S).log"

echo
echo "### 3) unzip to ${OUT_DIR} (no overwrite)"
unzip -n "$ZIP_PATH" -d "$OUT_DIR" | tee "${LOG_DIR}/unzip_extract_$(date +%Y%m%d_%H%M%S).log"

echo
echo "### 4) show top-level after unzip"
find "$OUT_DIR" -maxdepth 2 -type d | head -n 50

echo
echo "[OK] Unzip finished. Raw root: $OUT_DIR"
