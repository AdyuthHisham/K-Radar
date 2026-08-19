#!/bin/bash
#SBATCH -A NAISS2026-4-494
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 01:00:00
#SBATCH -J run_metrics
#SBATCH -o run_metrics_%j.log
#
# Runs compute_robustness.py (RR/mRR + CRI) across all 10 swept sequences
# against their existing complete_results.txt files -- no container/GPU
# needed (pure polars/stdlib), but every alvis-partition node requires a GPU
# allocation, so this requests one A40 anyway (same rationale as
# zip_predictions.sbatch / build_gallery.sbatch).
#
# Does NOT touch reconstruct_object_ids.py or its output -- independent of
# the trk_id backend, runs against complete_results.txt (official KITTI-eval
# AP) which already exists for every swept sequence.

set -e

AM=/cephyr/NOBACKUP/groups/c3se2025-3-18/users/adyuth/AMSCUP/repos/K-Radar
cd "$AM"

module load polars/1.28.1-gfbf-2024a

OUT=$AM/outputs/robustness_reports
mkdir -p "$OUT"

SEQS="2 7 8 9 15 16 18 41 48 55"

for SEQ in $SEQS; do
  SEQ_DIR="$AM/outputs/alvis_seq${SEQ}"
  ORIG_DIR="$SEQ_DIR/original"
  if [ ! -d "$ORIG_DIR" ]; then
    echo "seq $SEQ: no original/ dir, skipping" >&2
    continue
  fi

  CORRUPTED_ARGS=""
  for d in "$SEQ_DIR"/*/; do
    name=$(basename "$d")
    if [ "$name" == "original" ]; then continue; fi
    CORRUPTED_ARGS="$CORRUPTED_ARGS --corrupted-dir $d"
  done

  echo "=== seq $SEQ: $(echo $CORRUPTED_ARGS | wc -w) corrupted-dir args ==="
  python3 scripts/alvis/compute_robustness.py \
    --seq "$SEQ" \
    --original-dir "$ORIG_DIR" \
    $CORRUPTED_ARGS \
    --out "$OUT/seq${SEQ}_robustness.json" \
    2>&1 | tee "$OUT/seq${SEQ}_robustness.log"
done

echo "--- reports written ---"
ls -la "$OUT"
