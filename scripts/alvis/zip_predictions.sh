#!/bin/bash
# Zip every run's KITTI predictions/GT/AP-results (test_kitti/) and
# run_manifest.json into ONE archive file.
#
# Why: Alvis enforces a per-project FILE COUNT quota, not (only) a byte-size
# quota. The corruption sweep writes one KITTI .txt file per frame per
# condition per road/time/weather split (~4x redundancy -- predictions land
# under preds/, and again under each matching condition subdir), which
# multiplies out to over a million individual files even though the total
# byte size is modest (a few GB). Collapsing them into a single zip reduces
# the file count from ~1.1M to 1, independent of how much disk space is
# actually used.
#
# Usage (run ON Alvis, e.g. via ssh):
#   bash scripts/alvis/zip_predictions.sh [output.zip]
#
# Default output: outputs/all_predictions.zip
# Source scope: outputs/alvis_seq*/**/test_kitti/** and
#               outputs/alvis_seq*/**/run_manifest.json
#
# Does NOT delete the source files -- run scripts/alvis/delete_zipped_predictions.sh
# (or the equivalent manual rm) only after verifying the zip's file count
# matches the source count.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
OUT_DIR="$REPO_ROOT/outputs"
ZIP_PATH="${1:-$OUT_DIR/all_predictions.zip}"
FILELIST="$OUT_DIR/.predictions_filelist.txt"

cd "$OUT_DIR"

echo "Building file list (test_kitti/** + run_manifest.json under alvis_seq*/)..."
find alvis_seq* -type f \( -path '*test_kitti*' -o -name 'run_manifest.json' \) 2>/dev/null > "$FILELIST"
N_FILES=$(wc -l < "$FILELIST")
echo "Found $N_FILES files."

if [ "$N_FILES" -eq 0 ]; then
  echo "No files found -- nothing to zip." >&2
  rm -f "$FILELIST"
  exit 1
fi

if [ -f "$ZIP_PATH" ]; then
  echo "Removing existing $ZIP_PATH before rebuilding."
  rm -f "$ZIP_PATH"
fi

echo "Zipping into $ZIP_PATH (store mode, -0, no compression -- the actual" \
     "problem is file COUNT not byte size, so skip the CPU-bound deflate work)..."
zip -q -0 "$ZIP_PATH" -@ < "$FILELIST"

N_IN_ZIP=$(unzip -l "$ZIP_PATH" | tail -1 | awk '{print $2}')
echo "Zip written: $ZIP_PATH"
echo "Files in zip: $N_IN_ZIP (source list had $N_FILES)"
if [ "$N_IN_ZIP" != "$N_FILES" ]; then
  echo "WARNING: file count mismatch -- do not delete source files until this is resolved." >&2
  exit 1
fi

ls -la "$ZIP_PATH"
echo "Verified: file count matches. Source files NOT deleted -- delete manually" \
     "once you've confirmed the zip is usable (e.g. pulled locally, or its" \
     "presence alone is enough to reduce the file-count quota if source is removed)."
rm -f "$FILELIST"
