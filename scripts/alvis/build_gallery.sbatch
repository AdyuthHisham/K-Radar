#!/bin/bash
#SBATCH -A NAISS2026-4-494
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 00:45:00
#SBATCH -J build_gallery
#SBATCH -o build_gallery_%j.log
#
# Gallery-rendering job. Doesn't touch the model or need the GPU itself (just
# renders BEV/camera images from already-written predictions and sensor
# dumps) -- but Alvis has no CPU-only nodes, every node in the alvis
# partition requires a GPU allocation, so this requests one A40 anyway.
# Runs GALLERY_CMD inside the same apptainer container used for inference,
# since that's where numpy/torch/opencv/PIL already live -- no separate env
# needed.
#
# Required env var (set via sbatch --export):
#   GALLERY_SCRIPT   path to a small shell script containing the exact
#                    python3 command to run. NOT passed as an inline command
#                    string via --export -- a multi-argument command with
#                    spaces/braces mangles across the local-shell -> ssh ->
#                    remote-shell -> sbatch --export quoting chain (confirmed
#                    empirically: --out got silently truncated off the end).
#                    Write the command to a file instead; see
#                    scripts/alvis/submit_gallery.sh.

set -e

: "${GALLERY_SCRIPT:?GALLERY_SCRIPT must be set (see scripts/alvis/submit_gallery.sh)}"

AM=/cephyr/NOBACKUP/groups/c3se2025-3-18/users/adyuth/AMSCUP/repos/K-Radar
SIF=/mimer/NOBACKUP/groups/naiss2025-22-321/users/adyuth/containers/kradarV0.sif
KRADAR_DATA=/cephyr/NOBACKUP/groups/c3se2025-3-18/datasets/kradar

cd "$AM"
apptainer exec --bind "$AM:$AM" --bind "$KRADAR_DATA:$AM/data_bind" "$SIF" bash -c "
  cd $AM
  export PYTHONPATH=$AM:\$PYTHONPATH
  bash $GALLERY_SCRIPT
"

echo "Gallery job finished."
