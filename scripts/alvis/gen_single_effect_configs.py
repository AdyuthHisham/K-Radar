#!/usr/bin/env python3
"""Generate low/medium/high-severity noise-config YAML triples, one per
(modality, effect).

The robustness sweep runs each corruption in isolation at three severity
levels -- one effect, one modality, one severity, everything else original --
so any change in detections is attributable to that single corruption at that
intensity. Emits:
  - 30 low/medium/high triples (10 effects x 3 severities): radar
    frame_deletion, gaussian_noise, loss_partial; lidar frame_deletion,
    gaussian_noise, loss_partial; camera frame_deletion, gaussian_noise,
    loss_partial. (radar noise_induced_shifts intentionally excluded --
    DISABLED, see datasets/effects/noise_injection.py TODO.)
  - 2 unconditional configs (no severity split, no params): radar and lidar
    loss_complete_zero -- distinct from `loss_complete` (see below).

2026-08-17 literature-audit update: radar gaussian_noise added (new effect,
isotropic, SNR-independent -- distinct from noise_induced_shifts); lidar
gaussian_noise collapsed from a sigma_xy/sigma_z anisotropic split to one
isotropic sigma; camera gaussian_noise sigma raised to 20/50/80 (was
1.0/40.0/50.0); loss_partial's high tier lowered to 0.7 (was 0.8) across
all three modalities. frame_deletion and radar/lidar loss_complete/
loss_complete_zero are unchanged by this update.

2026-08-18: lidar gaussian_noise's severities changed to match radar's
(0.1/1.0/2.0, was 0.04/0.08/0.12 -- CommonCorruption's gaussian_rad levels
1/3/5) per user decision. Per docs/lit_survey/noise_addition/parameter_audit.md,
the CommonCorruption values were themselves an unanchored evenly-spaced ladder
(no physical/sensor-spec derivation given in that paper) calibrated for
KITTI/Velodyne, not K-Radar's Ouster os2-64/os1-128 LiDARs, and applied noise
to spherical-radial r rather than isotropic Cartesian xyz as implemented
here -- matching lidar to radar's range trades one unanchored ladder for a
different one (radar's own 0.1/1.0/2.0 is only paper-anchored at its two
endpoints, per Kumar et al. 2025), not a strictly better-justified choice,
just a consistent one across both point-cloud modalities.

`loss_complete` (the None-setting variant) is deliberately NOT emitted, same
as before. Per docs/vAIlt/04_Design/Noise_Taxonomy.md:19 and this generator's
prior empirical finding (2026-08-11 sweep), it is redundant with
`frame_deletion(p=1.0)` for radar/lidar -- both call _set_none unconditionally
when fired, and aborted identically in that sweep. `loss_complete_zero` (added
2026-08-14) is NOT redundant with frame_deletion: it zero-fills the sensor
array in place instead of setting it to None, so the model sees a present,
well-formed all-zero tensor rather than a missing key resolved later by
blackout_policy. Camera's own `loss_complete` is already zero-tensor based
(no `_zero` camera variant needed).

Severity values come from docs/vAIlt/06_Results/noise-visual-inspection-
severity-levels-report.md's full low/medium/high table, NOT
outputs/noise_visual_inspection/effect_parameters.yaml's single realized
values. Confirmed 2026-08-14/17: camera gaussian_noise's medium=40/high=50
match the doc exactly (an earlier 2-severity sweep this session had used
sigma=40 as "high", overriding the doc's own high=50, per explicit
instruction at the time -- now superseded: with a third tier added, 40
becomes "medium" and 50 becomes "high", consistent with the doc throughout,
confirmed 2026-08-17). Every other effect's low/high values already matched
the doc; medium is new for all of them.

`frame_deletion` uses `mode: random, p: <low|medium|high>` (not the earlier
`mode: deterministic, interval: 2`) to match the severity table's
probability-based parameterization.

Usage:
    python scripts/alvis/gen_single_effect_configs.py
"""
from __future__ import annotations

import os

OUT_DIR = os.path.join("configs", "noise", "single")

# (effect_name, {"low": params, "medium": params, "high": params}, note) per modality.
SEVERITY_EFFECTS: dict[str, list[tuple[str, dict, str]]] = {
    "radar": [
        ("frame_deletion",
         {"low": {"mode": "random", "p": 0.1}, "medium": {"mode": "random", "p": 0.5},
          "high": {"mode": "random", "p": 0.9}},
         "radar frame randomly deleted with probability p; sets rdr_sparse to None -> blackout policy applies"),
        # "noise_induced_shifts" intentionally excluded as of 2026-08-17 --
        # DISABLED in datasets/effects/noise_injection.py pending an SNR-dial
        # reparameterization decision (see TODO at radar_noise_induced_shifts()'s
        # definition there for current state / options / limitation). Do not
        # re-add this tuple without resolving that first.
        ("gaussian_noise",
         {"low": {"sigma": 0.1}, "medium": {"sigma": 1.0}, "high": {"sigma": 2.0}},
         "new 2026-08-17: additive N(0, sigma) on radar xyz, isotropic, "
         "independent of point power/RCS/range -- distinct from noise_induced_shifts"),
        ("loss_partial",
         {"low": {"fraction": 0.1}, "medium": {"fraction": 0.3}, "high": {"fraction": 0.7}},
         "fraction of radar detections zeroed in place; row count preserved"),
    ],
    "lidar": [
        ("frame_deletion",
         {"low": {"mode": "random", "p": 0.1}, "medium": {"mode": "random", "p": 0.5},
          "high": {"mode": "random", "p": 0.9}},
         "lidar frame randomly deleted with probability p; sets ldr64 to None -> blackout policy applies"),
        ("gaussian_noise",
         {"low": {"sigma": 0.1}, "medium": {"sigma": 1.0}, "high": {"sigma": 2.0}},
         "additive N(0, sigma) on lidar xyz, isotropic (xy/z split dropped 2026-08-17); shape "
         "preserved. Severities matched to radar_gaussian_noise's per user decision "
         "2026-08-18 (was 0.04/0.08/0.12, CommonCorruption's gaussian_rad levels 1/3/5 -- see "
         "docs/lit_survey/noise_addition/parameter_audit.md for that source's caveats, "
         "including that it perturbs spherical-radial r, not isotropic Cartesian xyz as "
         "implemented here, and was calibrated for KITTI/Velodyne, not K-Radar's Ouster "
         "os2-64/os1-128 LiDARs)."),
        ("loss_partial",
         {"low": {"fraction": 0.1}, "medium": {"fraction": 0.3}, "high": {"fraction": 0.7}},
         "fraction of lidar points zeroed in place; row count preserved"),
    ],
    "camera": [
        ("frame_deletion",
         {"low": {"mode": "random", "p": 0.1}, "medium": {"mode": "random", "p": 0.5},
          "high": {"mode": "random", "p": 0.9}},
         "camera frame randomly deleted with probability p; writes a black-image tensor, never None"),
        ("gaussian_noise",
         {"low": {"sigma": 20.0}, "medium": {"sigma": 50.0}, "high": {"sigma": 80.0}},
         "additive N(0, sigma) in 0-255 image units"),
        ("loss_partial",
         {"low": {"fraction": 0.1}, "medium": {"fraction": 0.3}, "high": {"fraction": 0.7}},
         "fraction of the image (contiguous rectangle) blacked out"),
    ],
}

# (modality, effect, note) -- unconditional, no severity split, no params.
UNCONDITIONAL_EFFECTS: list[tuple[str, str, str]] = [
    ("radar", "loss_complete_zero",
     "radar sensor alive but every reading zero-filled in place (distinct from "
     "loss_complete, which sets the key to None)"),
    ("lidar", "loss_complete_zero",
     "lidar sensor alive but every reading zero-filled in place (distinct from "
     "loss_complete, which sets the key to None)"),
]

HEADER = """\
# Single-corruption run: {modality} / {effect}{severity_suffix}
#
# {note}
#
# Exactly one effect on exactly one modality -- every other sensor is original
# -- so any change in detections versus the original baseline is attributable
# to this corruption alone. Generated by scripts/alvis/gen_single_effect_configs.py;
# edit that file rather than this one.
seed: 42
"""


def _fmt_params(params: dict) -> str:
    if not params:
        return "    params: {}\n"
    lines = ["    params:\n"]
    for k, v in params.items():
        lines.append(f"      {k}: {v}\n")
    return "".join(lines)


def _write_config(path: str, modality: str, effect: str, params: dict, note: str,
                   severity: str | None) -> None:
    severity_suffix = f" ({severity})" if severity else ""
    body = HEADER.format(modality=modality, effect=effect, note=note,
                          severity_suffix=severity_suffix)
    for m in ("radar", "lidar", "camera"):
        if m != modality:
            body += f"{m}: []\n"
            continue
        body += f"{m}:\n  - name: {effect}\n    p: 1.0\n"
        body += _fmt_params(params)
    with open(path, "w") as f:
        f.write(body)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    written = []

    for modality, effects in SEVERITY_EFFECTS.items():
        for effect, severities, note in effects:
            for severity, params in severities.items():
                path = os.path.join(OUT_DIR, f"{modality}_{effect}_{severity}.yml")
                _write_config(path, modality, effect, params, note, severity)
                written.append(path)

    for modality, effect, note in UNCONDITIONAL_EFFECTS:
        path = os.path.join(OUT_DIR, f"{modality}_{effect}.yml")
        _write_config(path, modality, effect, {}, note, None)
        written.append(path)

    print(f"Wrote {len(written)} configs to {OUT_DIR}/")
    for p in sorted(written):
        print(f"  {p}")


if __name__ == "__main__":
    main()
