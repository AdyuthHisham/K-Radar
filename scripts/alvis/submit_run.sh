#!/bin/bash
# Submit ASF original/corrupted runs on Alvis, for any sequence with a
# matching configs/ASF_v2_0_seq${SEQ}_alvis.yml.
#
# Generalized from submit_seq46.sh (which stays in place, untouched, as the
# record of the sequence-46 pilot's 2-run scope). This version covers the
# full sweep: 1 original baseline + 20 corrupted single-effect configs under
# configs/noise/single/ (9 effects x 2 severities + 2 unconditional
# loss_complete_zero variants -- see gen_single_effect_configs.py).
#
# Run 'original' first and verify its predictions before running anything
# else -- both a single named condition and 'sweep' refuse to submit without
# --confirmed, since burning 20 corrupted-job GPU-hours on a broken baseline
# is expensive to discover after the fact.
#
# Some sweep conditions are EXPECTED to abort the model outright (radar/lidar
# frame_deletion at high severity, both loss_complete_zero variants -- the
# sparse-conv backbone chokes on zero-point input). That is a real result for
# those conditions, not a driver bug; asf_run.sbatch already treats "no
# predictions written" as a normal non-crashing outcome.
#
# Usage:
#   bash scripts/alvis/submit_run.sh <SEQ> original
#   bash scripts/alvis/submit_run.sh <SEQ> <condition_name> --confirmed
#   bash scripts/alvis/submit_run.sh <SEQ> sweep --confirmed
#
# <condition_name> is a config/noise/single/<condition_name>.yml basename,
# e.g. camera_gaussian_noise_high, radar_loss_complete_zero.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$HERE/asf_run.sbatch"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
NOISE_DIR="$REPO_ROOT/configs/noise/single"

SEQ="${1:-}"
CONDITION="${2:-}"
CONFIRMED_FLAG="${3:-}"

if [ -z "$SEQ" ]; then
  echo "Usage: $0 <SEQ> {original|<condition_name> --confirmed|sweep --confirmed}" >&2
  exit 1
fi

CONFIG_FILE="$REPO_ROOT/configs/ASF_v2_0_seq${SEQ}_alvis.yml"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "No dataset config for sequence $SEQ: $CONFIG_FILE does not exist." >&2
  exit 1
fi

submit_one() {
  local seq="$1" out_name="$2" noise_cfg="$3" blackout_policy="${4:-empty}"
  echo "Submitting seq${seq}/${out_name} (noise-config=${noise_cfg:-<none>}, blackout-policy=$blackout_policy)"
  sbatch \
    --job-name="asf${seq}_${out_name}" \
    --output="asf${seq}_${out_name}_%j.log" \
    --export="ALL,SEQ=${seq},OUT_NAME=${out_name},NOISE_CFG=${noise_cfg},BLACKOUT_POLICY=${blackout_policy}" \
    "$SBATCH_SCRIPT"
}

require_confirmed() {
  local usage_example="$1"
  if [ "$CONFIRMED_FLAG" != "--confirmed" ]; then
    echo "Refusing to submit corrupted run(s) before the original baseline is verified." >&2
    echo "Re-run as: $usage_example" >&2
    exit 1
  fi
}

case "$CONDITION" in
  original)
    submit_one "$SEQ" original "" empty
    ;;
  sweep)
    require_confirmed "$0 $SEQ sweep --confirmed"
    if [ ! -d "$NOISE_DIR" ] || [ -z "$(ls -A "$NOISE_DIR"/*.yml 2>/dev/null)" ]; then
      echo "No noise configs found under $NOISE_DIR -- run gen_single_effect_configs.py first." >&2
      exit 1
    fi
    n=0
    for cfg_path in "$NOISE_DIR"/*.yml; do
      cond_name="$(basename "$cfg_path" .yml)"
      submit_one "$SEQ" "$cond_name" "single/$(basename "$cfg_path")" empty
      n=$((n + 1))
    done
    echo "Submitted $n corrupted-condition jobs for sequence $SEQ."
    ;;
  "")
    echo "Usage: $0 <SEQ> {original|<condition_name> --confirmed|sweep --confirmed}" >&2
    exit 1
    ;;
  *)
    cfg_path="$NOISE_DIR/${CONDITION}.yml"
    if [ ! -f "$cfg_path" ]; then
      echo "No noise config for condition '$CONDITION': $cfg_path does not exist." >&2
      exit 1
    fi
    require_confirmed "$0 $SEQ $CONDITION --confirmed"
    submit_one "$SEQ" "$CONDITION" "single/$(basename "$cfg_path")" empty
    ;;
esac
