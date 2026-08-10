#!/usr/bin/env python3
"""
3-severity-level (low/medium/high) visual inspection, v3.

Extends v2 (`visualize_effects_v2.py`) from one fixed parameter value per
effect to three severity levels per effect, for every severity-capable
effect in the 12-effect taxonomy. `loss_complete` is excluded from the
severity axis (confirmed in `noise-injection-param-sweep-report.md` R4/L4/C4:
always full clear regardless of params) and rendered once per frame instead.

Reuses v2's rendering functions verbatim by importing them (render_bev,
render_bev_zeroout, data loaders, tensor<->bgr helpers) rather than
reimplementing — this task is about parameter coverage, not new rendering
logic. Adds only: a severity parameter table, a per-severity render loop,
and a 3-column (low|medium|high) index.html grid.

Severity values reuse exactly the tested points from the param sweep
(`noise-injection-param-sweep-report.md`) — see SEVERITY_PARAMS below and
the accompanying report for per-effect justification. Does NOT modify
noise_injection.py, visualize_effects.py, visualize_effects_v2.py, or
diagnose_loss_partial_zeroout.py.

Run: .smoke_venv/bin/python datasets/effects/visualize_effects_v3_severity.py
"""

from __future__ import annotations

import sys
import hashlib
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize_effects_v2 import (
    load_rdr_sparse, load_lidar_pcd, load_camera_img,
    make_dict_item, apply_effect,
    render_bev, render_bev_zeroout,
    tensor_to_bgr,
    FRAMES, MASTER_SEED,
)
import torch

REPO_DIR = Path("/home/adhish/Productivity/AMSCUP/repos/K-Radar")
OUTPUT_DIR = REPO_DIR / "outputs" / "noise_visual_inspection"
VIZ_SCRIPT_PATH = REPO_DIR / "datasets" / "effects" / "visualize_effects_v3_severity.py"

SEVERITIES = ["low", "medium", "high"]

# Values reused verbatim from noise-injection-param-sweep-report.md's tested
# points (see accompanying severity report for the per-effect trace-back).
SEVERITY_PARAMS = {
    "radar": {
        "loss_partial": {"low": {"fraction": 0.1}, "medium": {"fraction": 0.3}, "high": {"fraction": 0.8}},
        "frame_deletion": {
            "low": {"mode": "random", "p": 0.1},
            "medium": {"mode": "random", "p": 0.5},
            "high": {"mode": "random", "p": 0.9},
        },
        "noise_induced_shifts": {"low": {"shift_std": 0.01, "distribution": "gaussian"},
                                  "medium": {"shift_std": 1.0, "distribution": "gaussian"},
                                  "high": {"shift_std": 5.0, "distribution": "gaussian"}},
    },
    "lidar": {
        "loss_partial": {"low": {"fraction": 0.1}, "medium": {"fraction": 0.3}, "high": {"fraction": 0.8}},
        "frame_deletion": {
            "low": {"mode": "random", "p": 0.1},
            "medium": {"mode": "random", "p": 0.5},
            "high": {"mode": "random", "p": 0.9},
        },
        "gaussian_noise": {"low": {"sigma_xy": 0.01, "sigma_z": 0.01},
                            "medium": {"sigma_xy": 0.5, "sigma_z": 0.2},
                            "high": {"sigma_xy": 1.0, "sigma_z": 1.0}},
    },
    "camera": {
        "loss_partial": {"low": {"fraction": 0.1}, "medium": {"fraction": 0.3}, "high": {"fraction": 0.8}},
        "frame_deletion": {
            "low": {"mode": "random", "p": 0.1},
            "medium": {"mode": "random", "p": 0.5},
            "high": {"mode": "random", "p": 0.9},
        },
        "gaussian_noise": {"low": {"sigma": 1.0}, "medium": {"sigma": 40.0}, "high": {"sigma": 50.0}},
    },
}

# Statistically-expected outcome per severity, used to steer the seed search
# in _draw_frame_deletion (None = no preferred side, report whatever lands).
FRAME_DELETION_WANT_FIRED = {"low": False, "medium": None, "high": True}

MOD_EFFECTS = {
    "radar": ["frame_deletion", "noise_induced_shifts", "loss_partial"],
    "lidar": ["frame_deletion", "gaussian_noise", "loss_partial"],
    "camera": ["frame_deletion", "gaussian_noise", "loss_partial"],
}


def _seed_offset(modality: str, effect_name: str, severity: str, label: str, attempt: int = 0) -> int:
    h = hashlib.sha256(f"{modality}:{effect_name}:{severity}:{label}:{attempt}".encode()).digest()
    return int.from_bytes(h[:4], "little") % (2 ** 16)


def _frame_deletion_fired(modality: str, corrupted: dict) -> bool:
    """Detect whether frame_deletion actually fired on this draw.

    radar/lidar: fired sets the array key to None. camera: fired zero-fills
    the tensor with noise_injection.py's `_zero_camera_tensor` — NOT a
    literal all-zero-std tensor: it's 3 *different* per-channel constants
    ((0-mean)/std for R/G/B), so its global std is a fixed ~0.1327, not ~0.
    (Verified empirically — a naive "std < 1e-3" threshold never matches,
    which is what caused camera/high to search all 25 seed attempts and
    still land on "not fired" for every frame in the first version of this
    script.) Detect fired by comparing against the exact known zero-tensor
    per-channel means instead of a std threshold.
    """
    if modality == "radar":
        return corrupted.get("rdr_sparse", None) is None
    if modality == "lidar":
        return corrupted.get("ldr64", None) is None
    tensor = corrupted.get("front0", None)
    if tensor is None:
        return True
    # noise_injection._zero_camera_tensor's fixed per-channel constants for
    # IMAGENET_MEAN/STD (see bgr_to_normalized_tensor) — black (0,0,0) mapped
    # to normalized space: (0 - mean) / std.
    zero_means = torch.tensor([-2.1179, -2.0357, -1.8044])
    per_channel_mean = tensor.mean(dim=(1, 2))
    per_channel_std = tensor.std(dim=(1, 2))
    return bool(torch.allclose(per_channel_mean, zero_means, atol=1e-3) and
                torch.all(per_channel_std < 1e-4))


# frame_deletion is a genuinely binary per-frame effect (present/absent) —
# there is no fractional severity to it. A single random draw at a given p
# is inherently a coin flip and CAN come up "wrong" for a given seed (e.g.
# p=0.9 not firing is a real, documented 10% outcome, not a bug in p itself)
# — see v2's frame_deletion_random spot-check, which explicitly kept both
# outcomes to demonstrate the coin-flip is real. But for a fixed 3-image
# low/medium/high comparison, a flipped outcome makes the sequence LOOK
# non-monotonic even though the underlying probabilities are correctly
# ordered. So: search a small number of alternate seeds per (modality,
# label) to land the LOW (p=0.1) and HIGH (p=0.9) extremes on their
# statistically-expected outcome (not fired / fired respectively) — MEDIUM
# (p=0.5) is left as the first draw with no search, since 50/50 has no
# "expected" side to search for. Actual realized outcome per frame is
# reported in the severity report's frame_deletion table either way.
def _draw_frame_deletion(modality, label, params, clean_item, want_fired=None, max_tries=25):
    for attempt in range(max_tries):
        seed_off = _seed_offset(modality, "frame_deletion", str(params.get("p", params.get("mode"))), label, attempt)
        rng = np.random.default_rng(MASTER_SEED + seed_off)
        torch.manual_seed(MASTER_SEED + seed_off)
        corrupted = apply_effect(clean_item, modality, "frame_deletion", params, rng)
        fired = _frame_deletion_fired(modality, corrupted)
        if want_fired is None or fired == want_fired:
            return corrupted, seed_off, fired, attempt
    return corrupted, seed_off, fired, attempt


def main():
    print("[START] Visual inspection generator v3 (severity levels)")

    frame_deletion_log = []  # (modality, label, severity, p, fired, n_attempts)

    for label in [f[4] for f in FRAMES]:
        for mod in ["radar", "lidar", "camera"]:
            (OUTPUT_DIR / f"frame_{label}" / mod).mkdir(parents=True, exist_ok=True)

    severity_yaml_path = OUTPUT_DIR / "severity_parameters.yaml"
    with open(str(severity_yaml_path), "w") as f:
        f.write("# Severity-level parameters — Noise Visual Inspection v3\n")
        f.write(f"seed: {MASTER_SEED}\n\n")
        for mod, effects in SEVERITY_PARAMS.items():
            f.write(f"{mod}:\n")
            for eff_name, sev_map in effects.items():
                f.write(f"  {eff_name}:\n")
                for sev, params in sev_map.items():
                    f.write(f"    {sev}: {params}\n")
                f.write("\n")

    for seq, rdr_idx, ldr_idx, camf_idx, label in FRAMES:
        print(f"\n{'='*60}")
        print(f"[LOAD] {label}  seq={seq}  rdr={rdr_idx}  lidar={ldr_idx}  cam={camf_idx}")

        rdr = load_rdr_sparse(seq, rdr_idx)
        lidar = load_lidar_pcd(seq, ldr_idx)
        cam = load_camera_img(seq, camf_idx)
        print(f"  radar: {len(rdr)} pts  lidar: {len(lidar)} pts  cam: {cam.shape}")

        def _lims(arr, pad=3):
            return [float(arr[:, 0].min()) - pad, float(arr[:, 0].max()) + pad,
                    float(arr[:, 1].min()) - pad, float(arr[:, 1].max()) + pad]

        rdr_lims = _lims(rdr, pad=3)
        lidar_lims = _lims(lidar, pad=3)

        idx = {"rdr": 0, "ldr": 0, "cam": 0}
        clean_item = make_dict_item(rdr, lidar, cam, idx)

        for modality in ["radar", "lidar", "camera"]:
            mod_dir = OUTPUT_DIR / f"frame_{label}" / modality
            ax_limits = rdr_lims if modality == "radar" else (lidar_lims if modality == "lidar" else None)

            # ── loss_complete: single reference render, no severity axis ──
            lc_params = {}
            seed_off = _seed_offset(modality, "loss_complete", "n_a", label)
            rng = np.random.default_rng(MASTER_SEED + seed_off)
            torch.manual_seed(MASTER_SEED + seed_off)
            corrupted = apply_effect(clean_item, modality, "loss_complete", lc_params, rng)
            lc_path = mod_dir / f"{modality}_loss_complete.png"
            if modality == "camera":
                tensor = corrupted.get("front0", None)
                bgr = tensor_to_bgr(tensor) if tensor is not None else np.zeros((256, 704, 3), dtype=np.uint8)
                cv2.putText(bgr, f"loss_complete (N/A - always full clear) | {label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.imwrite(str(lc_path), bgr)
            else:
                key = "rdr_sparse" if modality == "radar" else "ldr64"
                data = corrupted.get(key, None)
                pts_data = data[:, :2] if data is not None and len(data) > 0 else None
                render_bev(pts_data, None, ax_limits,
                           f"loss_complete (N/A - always full clear) | {label}", lc_path)
            print(f"  [OK] {modality}/loss_complete (single render, no severity)")

            # ── severity-capable effects ──
            for eff_name in MOD_EFFECTS[modality]:
                for severity in SEVERITIES:
                    params = SEVERITY_PARAMS[modality][eff_name][severity]

                    if eff_name == "frame_deletion":
                        want_fired = FRAME_DELETION_WANT_FIRED[severity]
                        corrupted, seed_off, fired, n_attempts = _draw_frame_deletion(
                            modality, label, params, clean_item, want_fired=want_fired)
                        frame_deletion_log.append((modality, label, severity, params.get("p"),
                                                    fired, n_attempts))
                    else:
                        seed_off = _seed_offset(modality, eff_name, severity, label)
                        rng = np.random.default_rng(MASTER_SEED + seed_off)
                        torch.manual_seed(MASTER_SEED + seed_off)
                        corrupted = apply_effect(clean_item, modality, eff_name, params, rng)

                    out_path = mod_dir / f"{modality}_{eff_name}_{severity}.png"
                    is_new_style = (eff_name == "loss_partial" and modality in ("radar", "lidar"))

                    if modality in ("radar", "lidar"):
                        key = "rdr_sparse" if modality == "radar" else "ldr64"
                        data = corrupted.get(key, None)
                        pts_data = data[:, :2] if data is not None and len(data) > 0 else None
                        n_pts = len(data) if data is not None else 0
                        param_str = ", ".join(f"{k}={v}" for k, v in params.items() if k not in ("mode", "index_list"))
                        title = f"{eff_name} [{severity}] ({param_str}) | {label}" if param_str else f"{eff_name} [{severity}] | {label}"
                        title += f"\n{n_pts} pts"

                        if is_new_style:
                            is_zeroed = np.all(data == 0, axis=1) if data is not None else np.zeros(0, dtype=bool)
                            render_bev_zeroout(pts_data, is_zeroed, ax_limits,
                                                f"{modality.upper()} loss_partial [{severity}] FIXED ({param_str}) | {label}",
                                                out_path)
                        else:
                            vals = (data[:, 3] if modality == "radar" and data is not None
                                    and len(data) > 0 and data.shape[1] >= 4 else None)
                            render_bev(pts_data, vals, ax_limits, title, out_path)
                    else:  # camera
                        tensor = corrupted.get("front0", None)
                        bgr = tensor_to_bgr(tensor) if tensor is not None else np.zeros((256, 704, 3), dtype=np.uint8)
                        param_str = ", ".join(f"{k}={v}" for k, v in params.items() if k not in ("mode", "index_list"))
                        title_text = f"{eff_name} [{severity}] ({param_str}) | {label}"
                        cv2.putText(bgr, title_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (0, 255, 0), 2)
                        cv2.imwrite(str(out_path), bgr)

                    print(f"  [OK] {modality}/{eff_name}/{severity}")

    # ── frame_deletion realized-outcome log (random mode is probabilistic;
    #    document exactly what each displayed image shows, and how many
    #    alternate-seed attempts it took to land the wanted extreme) ──
    fd_log_path = OUTPUT_DIR / "frame_deletion_severity_log.md"
    with open(str(fd_log_path), "w") as f:
        f.write("# frame_deletion severity realized outcomes\n\n")
        f.write("`frame_deletion` is a binary per-frame effect (present/absent); severity here is\n")
        f.write("the random-mode `p` (probability of firing this specific frame), not a\n")
        f.write("continuous corruption strength. A single Bernoulli draw at a given `p` can\n")
        f.write("land on the less-likely outcome (e.g. p=0.9 not firing is a real ~10% event).\n")
        f.write("`low`/`high` search up to 25 alternate seeds to land on the statistically\n")
        f.write("expected outcome (not fired / fired); `medium` (p=0.5) takes the first draw\n")
        f.write("with no preference, since 50/50 has no expected side.\n\n")
        f.write("| Modality | Frame | Severity | p | Fired? | Seed attempts |\n")
        f.write("|---|---|---|---|---|---|\n")
        for mod, label, sev, p, fired, n_attempts in frame_deletion_log:
            f.write(f"| {mod} | {label} | {sev} | {p} | {fired} | {n_attempts + 1} |\n")
    print(f"[DONE] frame_deletion log: {fd_log_path}")

    # ──────────────────────────────────────────────
    #   HTML index (3-column severity grid)
    # ──────────────────────────────────────────────
    index_path = OUTPUT_DIR / "index.html"
    with open(str(index_path), "w") as f:
        f.write('''<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Noise Injection — Visual Inspection (v3, severity levels)</title>
<style>
body { font-family: -apple-system,BlinkMacSystemFont,sans-serif; max-width:1700px; margin:0 auto;
       padding:20px; background:#111; color:#ddd; }
h1 { color:#e94560; }
h2 { color:#e94560; margin:30px 0 10px; border-bottom:2px solid #333; padding-bottom:5px; }
h3 { color:#aaa; margin:15px 0 5px; font-size:14px; text-transform:uppercase; letter-spacing:1px; }
.frame-box { background:#1a1a2e; border-radius:8px; padding:20px; margin:20px 0; }
.effect-row { margin:14px 0; }
.effect-label { font-size:13px; color:#ccc; margin-bottom:6px; }
.sev-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }
.sev-grid img { width:100%; border:1px solid #333; border-radius:4px; }
.caption { text-align:center; font-size:11px; color:#888; margin-bottom:4px; }
.single-grid { display:grid; grid-template-columns:repeat(1,300px); gap:8px; }
.single-grid img { width:100%; border:1px solid #333; border-radius:4px; }
.badge { display:inline-block; font-size:10px; padding:1px 6px; border-radius:3px; margin-left:6px; }
.badge-fixed { background:#e94560; color:#fff; }
.badge-na { background:#555; color:#fff; }
.notice { background:#332; border-left:4px solid #e94560; padding:10px 14px; margin:10px 0 20px; font-size:13px; }
</style>
</head>
<body>
<h1>Noise Injection — Visual Inspection (v3, severity levels)</h1>
<p>12-effect taxonomy (Rev 2) applied to 3 real K-Radar frames, at low/medium/high severity
where a severity axis exists. Seed: ''' + str(MASTER_SEED) + '''</p>
<div class="notice">
<b>v3 adds severity levels to v2.</b> Each severity-capable effect now shows three renders
side-by-side (low | medium | high) using values reused directly from
<code>noise-injection-param-sweep-report.md</code>'s tested points. <code>loss_complete</code>
has no severity axis (always full clear regardless of params, confirmed in the param sweep) and
is shown as a single reference image per modality. radar/lidar <code>loss_partial</code> still use
the zero-out-aware <code>render_bev_zeroout()</code> (magenta marker + zoomed origin inset + counts)
at every severity. See
<a href="../../../docs/vAIlt/noise-visual-inspection-severity-levels-report.md" style="color:#e94560">noise-visual-inspection-severity-levels-report.md</a>.
</div>
''')

        for seq, rdr_idx, ldr_idx, camf_idx, label in FRAMES:
            f.write('<div class="frame-box">\n')
            f.write(f'<h2>Frame: {label}</h2>\n')
            f.write(f'<p style="color:#888">seq={seq}, rdr={rdr_idx}, lidar={ldr_idx}, cam={camf_idx}</p>\n')
            for modality in ["radar", "lidar", "camera"]:
                f.write(f'<h3>{modality.upper()}</h3>\n')

                clean_rel = f"frame_{label}/{modality}/{modality}_clean.png"
                if (OUTPUT_DIR / clean_rel).exists():
                    f.write('<div class="effect-row"><div class="effect-label">Clean (reference)</div>'
                            f'<div class="single-grid"><div><div class="caption">Clean</div>'
                            f'<a href="{clean_rel}"><img src="{clean_rel}"></a></div></div></div>\n')

                for eff in MOD_EFFECTS[modality]:
                    fixed_badge = ' <span class="badge badge-fixed">FIXED RENDER</span>' if (modality in ("radar", "lidar") and eff == "loss_partial") else ""
                    f.write(f'<div class="effect-row"><div class="effect-label">{eff.replace("_"," ").title()}{fixed_badge}</div>\n')
                    f.write('<div class="sev-grid">\n')
                    for sev in SEVERITIES:
                        img_name = f"{modality}_{eff}_{sev}.png"
                        rel = f"frame_{label}/{modality}/{img_name}"
                        f.write(f'<div><div class="caption">{sev.upper()}</div>'
                                f'<a href="{rel}"><img src="{rel}" alt="{sev}"></a></div>\n')
                    f.write("</div></div>\n")

                lc_rel = f"frame_{label}/{modality}/{modality}_loss_complete.png"
                f.write('<div class="effect-row"><div class="effect-label">Loss Complete'
                        '<span class="badge badge-na">N/A - no severity axis</span></div>'
                        f'<div class="single-grid"><div><div class="caption">Always full clear</div>'
                        f'<a href="{lc_rel}"><img src="{lc_rel}"></a></div></div></div>\n')
            f.write("</div>\n")

        f.write(f'<hr><p>Severity params: <a href="severity_parameters.yaml" style="color:#e94560">severity_parameters.yaml</a></p>\n')
        f.write(f'<p>Base params (v2): <a href="effect_parameters.yaml" style="color:#e94560">effect_parameters.yaml</a></p>\n')
        f.write(f"<p>Script: {VIZ_SCRIPT_PATH}</p>\n")
        f.write("</body>\n</html>\n")

    print(f"\n[DONE] Index: {index_path}")
    print(f"[DONE] Severity params: {severity_yaml_path}")
    print("ALL DONE.")


if __name__ == "__main__":
    main()
