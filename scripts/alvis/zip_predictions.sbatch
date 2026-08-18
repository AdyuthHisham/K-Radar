#!/bin/bash
#SBATCH -A NAISS2026-4-494
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 01:00:00
#SBATCH -J zip_predictions
#SBATCH -o zip_predictions_%j.log
#
# Runs scripts/alvis/zip_predictions.sh as a proper batch job. Doesn't touch
# the GPU or the K-Radar container -- this is plain find/zip file I/O -- but
# Alvis has no CPU-only nodes, every node in the alvis partition requires a
# GPU allocation, so this requests one A40 anyway (same rationale as
# build_gallery.sbatch).

set -e

AM=/cephyr/NOBACKUP/groups/c3se2025-3-18/users/adyuth/AMSCUP/repos/K-Radar
cd "$AM"
bash scripts/alvis/zip_predictions.sh
