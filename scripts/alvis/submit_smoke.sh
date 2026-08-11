#!/bin/bash
# Submit ASF corruption-study jobs on Alvis.
#
# Each job runs ONE corruption (or none) over the same 6-frame scope, so
# results are directly comparable across conditions.
#
# Usage:
#   bash scripts/alvis/submit_smoke.sh clean
#   bash scripts/alvis/submit_smoke.sh <out_name> <noise_cfg_path> [blackout_policy]
#   bash scripts/alvis/submit_smoke.sh sweep        # all 12 single-effect runs
#
# <noise_cfg_path> is relative to configs/noise/, e.g. single/lidar_loss_complete.yml
# blackout_policy defaults to 'empty' (a genuinely dead input, no model-side
# compensation).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$HERE/asf_smoke.sbatch"

submit_one() {
  local out_name="$1" noise_cfg="$2" blackout_policy="${3:-empty}"
  echo "Submitting $out_name (noise-config=${noise_cfg:-<none>}, blackout-policy=$blackout_policy)"
  sbatch \
    --job-name="asf_${out_name}" \
    --output="asf_${out_name}_%j.log" \
    --export="ALL,OUT_NAME=${out_name},NOISE_CFG=${noise_cfg},BLACKOUT_POLICY=${blackout_policy}" \
    "$SBATCH_SCRIPT"
}

case "${1:-}" in
  clean)
    submit_one clean "" empty
    ;;
  sweep)
    # The full canonical taxonomy, one effect per job. The four blackout runs
    # (radar/lidar frame_deletion and loss_complete) are expected to abort in
    # the sparse-conv backbone on a zero-point input -- that abort is the
    # result for those conditions, not a failure to fix.
    for m in radar lidar camera; do
      for e in frame_deletion gaussian_noise noise_induced_shifts loss_partial loss_complete; do
        cfg="single/${m}_${e}.yml"
        [ -f "configs/noise/$cfg" ] || continue
        submit_one "${m}_${e}" "$cfg" empty
      done
    done
    ;;
  "")
    echo "Usage: $0 {clean|sweep|<out_name> <noise_cfg> [policy]}" >&2
    exit 1
    ;;
  *)
    submit_one "$1" "${2:?noise config required (relative to configs/noise/)}" "${3:-empty}"
    ;;
esac
