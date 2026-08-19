#!/bin/bash
#SBATCH -A NAISS2026-4-494
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 03:00:00
#SBATCH -J run_ap_robustness
#SBATCH -o run_ap_robustness_%j.log
#
# RR/mRR + CRI computed directly from outputs/all_preds.txt / all_gts.txt
# via ingest.py + ap_metrics.py -- see run_ap_robustness.py's docstring for
# why complete_results.txt-based compute_robustness.py can't be used here
# (no 'original' entry survives the merge-to-flat-text step).
#
# Pure polars/numpy, no torch/container needed -- but every alvis-partition
# node requires a GPU allocation, so this requests one A40 anyway.
#
# Uses the coordinate-fallback matcher in ingest.py (force_backend=coordinate
# implicitly, since all_*_with_trkid.txt don't exist yet) -- this is the path
# that timed out past 5 minutes on a 274K-row local sample; real data here is
# ~438K pred lines, so this generously requests 3h walltime. If it's still
# running near the limit, that's a real finding (the matcher needs
# vectorizing, not sbatch tuning) -- report it, don't just raise the walltime
# and resubmit blindly.

set -e

AM=/cephyr/NOBACKUP/groups/c3se2025-3-18/users/adyuth/AMSCUP/repos/K-Radar
cd "$AM"

module load polars/1.28.1-gfbf-2024a

OUT=$AM/outputs/robustness_reports
mkdir -p "$OUT"

python3 scripts/alvis/run_ap_robustness.py \
  --outputs-dir outputs \
  --out "$OUT/ap_robustness.json"
