#!/bin/bash
# Submit the ASF clean-vs-noisy smoke jobs on Alvis.
#
# Usage:
#   bash scripts/alvis/submit_smoke.sh clean
#   bash scripts/alvis/submit_smoke.sh noisy_shape
#   bash scripts/alvis/submit_smoke.sh noisy_blackout [empty|zero_feature|last_frame_hold]
#   bash scripts/alvis/submit_smoke.sh all              # submits clean only;
#                                                        # run the other two after
#                                                        # clean's preds are confirmed
#
# Run from the K-Radar repo root on Alvis (where scripts/alvis/asf_smoke.sbatch
# is relative to `cd $AM` inside the job).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$HERE/asf_smoke.sbatch"

submit_one() {
  local out_name="$1" noise_cfg="$2" blackout_policy="$3"
  echo "Submitting $out_name (noise-config=${noise_cfg:-<none>}, blackout-policy=$blackout_policy)"
  sbatch \
    --job-name="asf_smoke_${out_name}" \
    --output="asf_smoke_${out_name}_%j.log" \
    --export="ALL,OUT_NAME=${out_name},NOISE_CFG=${noise_cfg},BLACKOUT_POLICY=${blackout_policy}" \
    "$SBATCH_SCRIPT"
}

case "${1:-}" in
  clean)
    submit_one clean "" empty
    ;;
  noisy_shape)
    submit_one noisy_shape shape_preserving.yml empty
    ;;
  noisy_blackout)
    submit_one noisy_blackout blackout.yml "${2:-empty}"
    ;;
  all)
    echo "Submitting clean only. Confirm it produces preds (see plan Verification"
    echo "step 2-3), then run: submit_smoke.sh noisy_shape && submit_smoke.sh noisy_blackout"
    submit_one clean "" empty
    ;;
  *)
    echo "Usage: $0 {clean|noisy_shape|noisy_blackout [empty|zero_feature|last_frame_hold]|all}" >&2
    exit 1
    ;;
esac
