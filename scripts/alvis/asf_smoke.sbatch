#!/bin/bash
#SBATCH -A NAISS2026-4-494
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 01:00:00
#SBATCH -J asf_smoke
#SBATCH -o asf_smoke_%j.log
#
# NOTE: #SBATCH directives are parsed before OUT_NAME is known, so -J/-o are
# generic here. submit_smoke.sh passes --job-name/-o with the real condition
# name via the sbatch CLI, which override these.

# ASF (A2Fusion, tri-modal) clean-vs-noisy smoke test on Alvis.
#
# One job = one condition, selected entirely by the env vars below (no other
# edits needed to add a fourth condition). Scope: 6 frames total — sequences
# 1 and 2, the 3 densest frames each — via configs/ASF_v2_0_smoke_alvis.yml's
# portion/max_per_seq/frame_select knobs, so all conditions run the identical
# frame set and predictions are directly comparable.
#
# Required env vars (set by submit_smoke.sh, or export before `sbatch --export=...`):
#   OUT_NAME          result subdir name, e.g. clean / noisy_shape / noisy_blackout
# Optional:
#   NOISE_CFG         noise YAML under configs/noise/, e.g. shape_preserving.yml
#                      (omit/empty for the clean run)
#   BLACKOUT_POLICY   empty (default) | zero_feature | last_frame_hold
#                      only meaningful when NOISE_CFG sets a blackout effect
#
# Carries the three Alvis fixes proven on the full 17.5k-frame ASF run (see
# scripts/main_cond_0_args.py / utils/kitti_eval/eval.py history): no eager
# GPU-NMS import at startup, CPU rotated-IoU fallback, KITTI split clamped for
# small N.

set -e

: "${OUT_NAME:?OUT_NAME must be set (see scripts/alvis/submit_smoke.sh)}"
NOISE_CFG="${NOISE_CFG:-}"
BLACKOUT_POLICY="${BLACKOUT_POLICY:-empty}"

AM=/cephyr/NOBACKUP/groups/c3se2025-3-18/users/adyuth/AMSCUP/repos/K-Radar
SIF=/mimer/NOBACKUP/groups/naiss2025-22-321/users/adyuth/containers/kradarV0.sif

KRADAR_DATA=/cephyr/NOBACKUP/groups/c3se2025-3-18/datasets/kradar
PRE=/cephyr/NOBACKUP/groups/c3se2025-3-18/datasets/kradar_pre/processed
TRAINED=/cephyr/NOBACKUP/groups/c3se2025-3-18/datasets/kradar_pre/trained

CKPT=/mimer/NOBACKUP/groups/naiss2025-22-321/users/adyuth/output/kradar/exp_260506_135737_A2FUSION_rlc_l1d256_l2p2t32d256g_l2g_scl_mha16/models/model_10.pt
CONFIG=./configs/ASF_v2_0_smoke_alvis.yml

OUT=$AM/outputs/alvis_smoke/$OUT_NAME
mkdir -p "$OUT"
export TMPDIR=$OUT/tmp
mkdir -p "$TMPDIR"

# Overlay only the fixed/new files over the baked repo (keeps the baked v2_0 labels).
BINDS=""
BINDS="$BINDS,$AM/scripts/main_cond_0_args.py:/opt/K-Radar/scripts/main_cond_0_args.py"
BINDS="$BINDS,$AM/datasets/kradar_fusion_v1_0.py:/opt/K-Radar/datasets/kradar_fusion_v1_0.py"
BINDS="$BINDS,$AM/datasets/effects:/opt/K-Radar/datasets/effects"
BINDS="$BINDS,$AM/configs/ASF_v2_0_smoke_alvis.yml:/opt/K-Radar/configs/ASF_v2_0_smoke_alvis.yml"
BINDS="$BINDS,$AM/configs/noise:/opt/K-Radar/configs/noise"
BINDS="$BINDS,$AM/utils/kitti_eval/eval.py:/opt/K-Radar/utils/kitti_eval/eval.py"
BINDS="$BINDS,$AM/utils/kitti_eval/eval_revised.py:/opt/K-Radar/utils/kitti_eval/eval_revised.py"
BINDS="$BINDS,$AM/utils/kitti_eval/rotate_iou_cpu.py:/opt/K-Radar/utils/kitti_eval/rotate_iou_cpu.py"
BINDS="$BINDS,$KRADAR_DATA:/opt/K-Radar/kradar_dataset"
BINDS="$BINDS,$PRE/processed_lpc_calib_roi:/opt/K-Radar/pre_processed/processed_lpc_calib_roi"
BINDS="$BINDS,$PRE/rdr_sparse_data:/opt/K-Radar/pre_processed/rdr_sparse_data"
BINDS="$BINDS,$PRE/kradar_imgs:/opt/K-Radar/pre_processed/kradar_imgs"
BINDS="$BINDS,$TRAINED/CAMERA_MODEL_10.pt:/opt/K-Radar/pretrained/CAMERA_MODEL_10.pt"
BINDS="$BINDS,$TRAINED/SECOND_10.pt:/opt/K-Radar/pretrained/SECOND_10.pt"
BINDS="$BINDS,$TRAINED/RTNH_wide_10.pt:/opt/K-Radar/pretrained/RTNH_wide_10.pt"
BINDS="$BINDS,$CKPT:/opt/K-Radar/run_ckpt.pt"
BINDS="$BINDS,$OUT:/opt/K-Radar/logs"
BINDS="${BINDS#,}"

NOISE_ARGS=""
if [ -n "$NOISE_CFG" ]; then
  NOISE_ARGS="--noise-config /opt/K-Radar/configs/noise/$NOISE_CFG --blackout-policy $BLACKOUT_POLICY"
fi

echo "=== condition: $OUT_NAME  noise-config: ${NOISE_CFG:-<none, clean run>}  blackout-policy: $BLACKOUT_POLICY ==="

cd "$AM"
# The inference + dump finish first; the process can still abort with a
# "free(): invalid pointer" during teardown (a tensorboard/torch cleanup
# segfault on exit), which happens after the preds are on disk. Do not let
# that fail the job; judge success on the preds.
set +e
apptainer exec --nv --bind "$BINDS" "$SIF" bash -c "
  cd /opt/K-Radar
  export PYTHONPATH=/opt/K-Radar:\$PYTHONPATH
  export TMPDIR=/opt/K-Radar/logs/tmp
  mkdir -p \$TMPDIR
  export KRADAR_EVAL_GPU_IOU=1
  python scripts/main_cond_0_args.py \
    --config $CONFIG \
    --model /opt/K-Radar/run_ckpt.pt \
    --conf_thr 0.3 \
    $NOISE_ARGS
"
set -e

echo '--- prediction files written ---'
NPRED=$(find "$OUT" -path '*all/preds/*.txt' | wc -l)
echo "--- count: $NPRED ---"
echo '--- AP results (complete_results.txt, not meaningful on 6 frames — see compare_smoke.py) ---'
find "$OUT" -name 'complete_results.txt' -exec echo {} \; -exec cat {} \;

if [ "$NPRED" -gt 0 ]; then
  echo "ASF smoke run OK ($OUT_NAME): $NPRED prediction files written."
  exit 0
else
  echo "ASF smoke run FAILED ($OUT_NAME): no predictions written."
  exit 1
fi
