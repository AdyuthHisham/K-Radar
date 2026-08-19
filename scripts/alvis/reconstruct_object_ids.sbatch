#!/bin/bash
#SBATCH -A NAISS2026-4-494
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 04:00:00
#SBATCH -J reconstruct_object_ids
#SBATCH -o reconstruct_object_ids_%j.log
#
# Runs scripts/alvis/reconstruct_object_ids.py inside the K-Radar container.
# Needs torch/open3d/cv2 (KRadarFusion_v1_0's deps) -- can't run on the login
# node or in a plain uv venv, same reasoning as asf_run.sbatch. Doesn't touch
# the GPU itself, but the alvis partition rejects GPU-less jobs, so this
# requests one A40 anyway (same rationale as zip_predictions.sbatch).
#
# Reads outputs/all_preds.txt / outputs/all_gts.txt (already merged by
# merge_predictions_text.py) and instantiates KRadarFusion_v1_0(cfg,
# split='test') read-only per swept sequence to recover each object's
# persistent trk_id -- see reconstruct_object_ids.py's own docstring for the
# two-stage method (GT rejoin by float-tolerance match, then greedy BEV-IoU
# pred<->GT matching reusing utils/kitti_eval/eval.py:bev_box_overlap).
#
# Usage:
#   Dry run on one sequence first (fast, cheap correctness check before
#   committing 4 hours of walltime to all 10):
#     sbatch --export=SEQS=32 scripts/alvis/reconstruct_object_ids.sbatch
#   Full run, all 10 swept sequences:
#     sbatch scripts/alvis/reconstruct_object_ids.sbatch

set -e

SEQS="${SEQS:-2,7,8,9,15,16,18,41,48,55}"

AM=/cephyr/NOBACKUP/groups/c3se2025-3-18/users/adyuth/AMSCUP/repos/K-Radar
SIF=/mimer/NOBACKUP/groups/naiss2025-22-321/users/adyuth/containers/kradarV0.sif

KRADAR_DATA=/cephyr/NOBACKUP/groups/c3se2025-3-18/datasets/kradar
PRE=/cephyr/NOBACKUP/groups/c3se2025-3-18/datasets/kradar_pre/processed

OUT=$AM/outputs
mkdir -p "$OUT"
export TMPDIR=$OUT/tmp_reconstruct
mkdir -p "$TMPDIR"

# Same overlay pattern as asf_run.sbatch: bind only the fixed/new files over
# the baked container image, keep the baked v2_0 labels. reconstruct_object_ids.py
# itself is new -- must be bound in since it isn't baked into the image.
BINDS=""
BINDS="$BINDS,$AM/scripts/alvis/reconstruct_object_ids.py:/opt/K-Radar/scripts/alvis/reconstruct_object_ids.py"
BINDS="$BINDS,$AM/datasets/kradar_fusion_v1_0.py:/opt/K-Radar/datasets/kradar_fusion_v1_0.py"
BINDS="$BINDS,$AM/configs:/opt/K-Radar/configs"
BINDS="$BINDS,$AM/utils/kitti_eval/eval.py:/opt/K-Radar/utils/kitti_eval/eval.py"
BINDS="$BINDS,$AM/utils/kitti_eval/eval_revised.py:/opt/K-Radar/utils/kitti_eval/eval_revised.py"
BINDS="$BINDS,$AM/utils/kitti_eval/rotate_iou_cpu.py:/opt/K-Radar/utils/kitti_eval/rotate_iou_cpu.py"
BINDS="$BINDS,$KRADAR_DATA:/opt/K-Radar/kradar_dataset"
BINDS="$BINDS,$PRE/processed_lpc_calib_roi:/opt/K-Radar/pre_processed/processed_lpc_calib_roi"
BINDS="$BINDS,$PRE/rdr_sparse_data:/opt/K-Radar/pre_processed/rdr_sparse_data"
BINDS="$BINDS,$PRE/kradar_imgs:/opt/K-Radar/pre_processed/kradar_imgs"
BINDS="$BINDS,$OUT:/opt/K-Radar/outputs"
BINDS="${BINDS#,}"

echo "=== reconstruct_object_ids: sequences=$SEQS ==="

cd "$AM"
apptainer exec --nv --bind "$BINDS" "$SIF" bash -c "
  cd /opt/K-Radar
  export PYTHONPATH=/opt/K-Radar:\$PYTHONPATH
  python scripts/alvis/reconstruct_object_ids.py \
    --outputs-dir outputs \
    --configs-dir configs \
    --gts-out outputs/all_gts_with_trkid.txt \
    --preds-out outputs/all_preds_with_trkid.txt \
    --fn-out outputs/all_fn.txt \
    --ambiguous-out outputs/all_gts_ambiguous.txt \
    --frame-index-map-out outputs/frame_index_map.txt \
    --seqs '$SEQS' \
    --workers 8
"

echo '--- output files ---'
ls -la "$OUT"/all_*_with_trkid.txt "$OUT"/all_fn.txt "$OUT"/all_gts_ambiguous.txt "$OUT"/frame_index_map.txt 2>/dev/null
