#!/bin/bash
#SBATCH -A NAISS2026-4-494
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 01:30:00
#SBATCH -J asf_run
#SBATCH -o asf_run_%j.log
#
# NOTE: #SBATCH directives are parsed before SEQ/OUT_NAME are known, so -J/-o
# are generic here. submit_run.sh passes --job-name/-o with the real
# sequence+condition name via the sbatch CLI, which override these.

# ASF (A2Fusion, tri-modal) original-vs-corrupted run on a full K-Radar
# sequence on Alvis. Sequence-generic: SEQ selects both the dataset config
# (configs/ASF_v2_0_seq${SEQ}_alvis.yml) and the output tree
# (outputs/alvis_seq${SEQ}/$OUT_NAME). Generalized from asf_seq46.sbatch,
# which stays in place, untouched, as the record of the sequence-46 pilot.
#
# One job = one (sequence, condition) pair, selected entirely by the env
# vars below.
#
# Required env vars (set by submit_run.sh, or export before `sbatch --export=...`):
#   SEQ               sequence number, e.g. 46 or 1 -- must have a matching
#                      configs/ASF_v2_0_seq${SEQ}_alvis.yml
#   OUT_NAME          result subdir name, e.g. original / camera_gaussian_noise_high
# Optional:
#   NOISE_CFG         noise YAML under configs/noise/, e.g. single/camera_gaussian_noise_high.yml
#                      (omit/empty for the original/baseline run)
#   BLACKOUT_POLICY   empty (default) | zero_feature | last_frame_hold
#                      only meaningful when NOISE_CFG sets a blackout effect
#
# Carries the three Alvis fixes proven on the full 17.5k-frame ASF run (see
# scripts/main_cond_0_args.py / utils/kitti_eval/eval.py history): no eager
# GPU-NMS import at startup, CPU rotated-IoU fallback, KITTI split clamped for
# small N.

set -e

: "${SEQ:?SEQ must be set (see scripts/alvis/submit_run.sh)}"
: "${OUT_NAME:?OUT_NAME must be set (see scripts/alvis/submit_run.sh)}"
NOISE_CFG="${NOISE_CFG:-}"
BLACKOUT_POLICY="${BLACKOUT_POLICY:-empty}"

AM=/cephyr/NOBACKUP/groups/c3se2025-3-18/users/adyuth/AMSCUP/repos/K-Radar
SIF=/mimer/NOBACKUP/groups/naiss2025-22-321/users/adyuth/containers/kradarV0.sif

KRADAR_DATA=/cephyr/NOBACKUP/groups/c3se2025-3-18/datasets/kradar
PRE=/cephyr/NOBACKUP/groups/c3se2025-3-18/datasets/kradar_pre/processed
TRAINED=/cephyr/NOBACKUP/groups/c3se2025-3-18/datasets/kradar_pre/trained

CKPT=/mimer/NOBACKUP/groups/naiss2025-22-321/users/adyuth/output/kradar/exp_260506_135737_A2FUSION_rlc_l1d256_l2p2t32d256g_l2g_scl_mha16/models/model_10.pt
CONFIG=./configs/ASF_v2_0_seq${SEQ}_alvis.yml

if [ ! -f "$AM/$CONFIG" ]; then
  echo "FATAL: $AM/$CONFIG does not exist -- no dataset config for sequence $SEQ" >&2
  exit 1
fi

OUT=$AM/outputs/alvis_seq${SEQ}/$OUT_NAME
mkdir -p "$OUT"
export TMPDIR=$OUT/tmp
mkdir -p "$TMPDIR"

# Overlay only the fixed/new files over the baked repo (keeps the baked v2_0 labels).
BINDS=""
BINDS="$BINDS,$AM/scripts/main_cond_0_args.py:/opt/K-Radar/scripts/main_cond_0_args.py"
BINDS="$BINDS,$AM/datasets/kradar_fusion_v1_0.py:/opt/K-Radar/datasets/kradar_fusion_v1_0.py"
BINDS="$BINDS,$AM/datasets/effects:/opt/K-Radar/datasets/effects"
BINDS="$BINDS,$AM/$CONFIG:/opt/K-Radar/$CONFIG"
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
  # Sensor dumps land next to the predictions so each condition keeps its own
  # corrupted images/point clouds (with original copies) for visual inspection.
  NOISE_ARGS="--noise-config /opt/K-Radar/configs/noise/$NOISE_CFG"
  NOISE_ARGS="$NOISE_ARGS --blackout-policy $BLACKOUT_POLICY"
  NOISE_ARGS="$NOISE_ARGS --dump-sensors /opt/K-Radar/logs/sensor_dump"
fi

echo "=== sequence: $SEQ  condition: $OUT_NAME  noise-config: ${NOISE_CFG:-<none, original run>}  blackout-policy: $BLACKOUT_POLICY ==="

# Run manifest: written before inference so it exists even if the job aborts.
# Records what this run actually is, independent of stdout log parsing.
GIT_SHA=$(cd "$AM" && git rev-parse HEAD)
NOISE_CFG_CONTENT="null"
if [ -n "$NOISE_CFG" ] && [ -f "$AM/configs/noise/$NOISE_CFG" ]; then
  NOISE_CFG_CONTENT=$(python3 -c "
import json, sys
try:
    import yaml
    print(json.dumps(yaml.safe_load(open('$AM/configs/noise/$NOISE_CFG'))))
except Exception as e:
    print(json.dumps({'error': str(e)}))
" 2>/dev/null || echo "null")
fi
cat > "$OUT/run_manifest.json" <<MANIFEST
{
  "out_name": "$OUT_NAME",
  "sequence": "$SEQ",
  "config": "$CONFIG",
  "noise_config_path": "${NOISE_CFG:-null}",
  "noise_config_content": $NOISE_CFG_CONTENT,
  "blackout_policy": "$BLACKOUT_POLICY",
  "checkpoint": "$CKPT",
  "git_sha": "$GIT_SHA",
  "slurm_job_id": "${SLURM_JOB_ID:-unknown}",
  "submitted_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
MANIFEST
echo "--- run manifest written: $OUT/run_manifest.json ---"
cat "$OUT/run_manifest.json"

cd "$AM"
# The inference + dump finish first; the process can still abort with a
# "free(): invalid pointer" during teardown (a tensorboard/torch cleanup
# segfault on exit), which happens after the preds are on disk. Do not let
# that fail the job; judge success on the preds.
#
# Some conditions (radar/lidar frame_deletion at high severity, both
# loss_complete_zero variants) are EXPECTED to abort the model itself in the
# sparse-conv backbone on zero-point input -- that is a real result for those
# conditions, not a driver bug. The exit-code logic below already treats "no
# predictions written" as a normal (non-zero-exit) outcome rather than
# crashing the script.
set +e
apptainer exec --nv --bind "$BINDS" "$SIF" bash -c "
  cd /opt/K-Radar
  export PYTHONPATH=/opt/K-Radar:\$PYTHONPATH
  export TMPDIR=/opt/K-Radar/logs/tmp
  mkdir -p \$TMPDIR
  # Do NOT set KRADAR_EVAL_GPU_IOU=1: that opts into a numba @cuda.jit
  # rotated-IoU kernel in nms_gpu.py which SIGSEGVs on driver mismatch
  # (see utils/kitti_eval/eval_revised.py:13-19). Leave unset -> CPU fallback.
  python scripts/main_cond_0_args.py \
    --config $CONFIG \
    --model /opt/K-Radar/run_ckpt.pt \
    --conf_thr 0.3 \
    $NOISE_ARGS
"
set -e

echo '--- prediction files written ---'
NPRED=$(find "$OUT" -path '*all/preds/*.txt' | wc -l)
echo "--- count: $NPRED (no baked-in expectation -- sequence $SEQ's post-filter frame count is read from this number, not assumed) ---"
echo "--- sensor dump: $(find "$OUT/sensor_dump" -type f 2>/dev/null | wc -l) file(s) ---"
echo '--- AP results ---'
find "$OUT" -name 'complete_results.txt' -exec echo {} \; -exec cat {} \;

if [ "$NPRED" -gt 0 ]; then
  echo "ASF run OK (seq $SEQ, $OUT_NAME): $NPRED prediction files written."
  exit 0
else
  echo "ASF run FAILED (seq $SEQ, $OUT_NAME): no predictions written."
  exit 1
fi
