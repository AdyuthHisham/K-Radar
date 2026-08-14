#!/bin/bash
# Submit ASF sequence-46 runs on Alvis.
#
# Scope (per current plan — only 2 conditions, not the full 22-corruption
# sweep): 1) clean baseline, 2) camera gaussian_noise at high severity
# (sigma=40, configs/noise/single/camera_gaussian_noise.yml). Run 'clean'
# first and verify its predictions before submitting 'camera_gaussian_high' —
# this script will not submit the corrupted run until you pass --confirmed.
#
# Usage:
#   bash scripts/alvis/submit_seq46.sh clean
#   bash scripts/alvis/submit_seq46.sh camera_gaussian_high --confirmed
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$HERE/asf_seq46.sbatch"

submit_one() {
  local out_name="$1" noise_cfg="$2" blackout_policy="${3:-empty}"
  echo "Submitting $out_name (noise-config=${noise_cfg:-<none>}, blackout-policy=$blackout_policy)"
  sbatch \
    --job-name="asf46_${out_name}" \
    --output="asf46_${out_name}_%j.log" \
    --export="ALL,OUT_NAME=${out_name},NOISE_CFG=${noise_cfg},BLACKOUT_POLICY=${blackout_policy}" \
    "$SBATCH_SCRIPT"
}

case "${1:-}" in
  clean)
    submit_one clean "" empty
    ;;
  camera_gaussian_high)
    if [ "${2:-}" != "--confirmed" ]; then
      echo "Refusing to submit the corrupted run before the clean run is verified." >&2
      echo "Re-run as: $0 camera_gaussian_high --confirmed" >&2
      exit 1
    fi
    submit_one camera_gaussian_high "single/camera_gaussian_noise.yml" empty
    ;;
  *)
    echo "Usage: $0 {clean|camera_gaussian_high --confirmed}" >&2
    exit 1
    ;;
esac
