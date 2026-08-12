#!/usr/bin/env python3
"""Build an HTML gallery of the ASF corruption study's sensor data.

Renders clean-vs-corrupted BEV images for every condition and frame in the
viewable bundle (see export_viewable.py), copies the camera frames, and writes
an index.html in the same style as outputs/noise_visual_inspection/index.html.

Reuses render_bev / render_bev_zeroout from
datasets/effects/visualize_effects_v2.py so the renders match the existing
inspection gallery. loss_partial and *_zeros use the zero-out-aware renderer,
because those effects collapse points to the origin rather than deleting them —
a plain BEV would just show a dense dot at (0,0) with no explanation.

If --kitti-dir points at a directory of <condition>/{preds,gts}/<frame>.txt
(KITTI-format, as written by the eval pipeline), every radar/LiDAR BEV image
also gets the model's detected objects overlaid: solid boxes for that panel's
own predictions (the clean panel uses the clean run's predictions; the
corrupted panel uses that condition's own predictions on the corrupted input),
dashed boxes for ground truth. See kitti_boxes.py for the coordinate mapping.

Usage:
    python scripts/alvis/build_gallery.py --bundle ~/kradar_viewable \
        --kitti-dir outputs/alvis_smoke/kitti_data
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "datasets", "effects"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_effects_v2 import render_bev, render_bev_zeroout  # noqa: E402
from kitti_boxes import parse_kitti_boxes, draw_boxes  # noqa: E402

RADAR_IMG = (900, 700)     # render_bev defaults
RADAR_ZO_IMG = (1000, 760)  # render_bev_zeroout defaults

# ROI from configs/ASF_v2_0_smoke_alvis.yml: xyz [0,-16,-2, 72,16,7.6].
# Padded slightly so points just outside the ROI are still visible.
RADAR_LIMS = (-5.0, 80.0, -25.0, 25.0)
LIDAR_LIMS = (-5.0, 80.0, -25.0, 25.0)

# Conditions whose point clouds contain origin-collapsed (zeroed) rows.
ZEROOUT_EFFECTS = ("loss_partial", "zeros")

CONDITION_NOTES = {
    "radar_noise_induced_shifts": ("Radar — Noise-Induced Shifts",
        "Every detection displaced by N(0, 2.0 m). Shape preserved.", "16 / 19 detections (−3)"),
    "radar_loss_partial": ("Radar — Loss Partial (0.5)",
        "Half the detections zeroed in place; row count preserved.", "19 / 19 detections (+0)"),
    "radar_zeros": ("Radar — All-Zero Frame",
        "Sensor alive but every reading zero (loss_partial at fraction 1.0).", "20 / 19 detections (+1)"),
    "radar_frame_deletion": ("Radar — Frame Deletion",
        "Radar delivers nothing on even frames. ABORTS the model in the "
        "sparse-conv backbone (zero voxels), so only frame 0 exists.", "job aborted, 0 predictions"),
    "lidar_gaussian_noise": ("LiDAR — Gaussian Noise",
        "N(0, 0.5 m) on x/y, N(0, 0.2 m) on z. Exceeds the 0.4 m voxel size.", "7 / 19 detections (−12)"),
    "lidar_loss_partial": ("LiDAR — Loss Partial (0.5)",
        "Half the points zeroed in place; row count preserved.", "16 / 19 detections (−3)"),
    "lidar_zeros": ("LiDAR — All-Zero Frame",
        "Sensor alive but every reading zero.", "9 / 19 detections (−10)"),
    "lidar_frame_deletion": ("LiDAR — Frame Deletion",
        "LiDAR delivers nothing on even frames. ABORTS the model in the "
        "sparse-conv backbone, so only frame 0 exists.", "job aborted, 0 predictions"),
    "camera_gaussian_noise": ("Camera — Gaussian Noise (σ=40)",
        "Additive noise in 0–255 image units.", "18 / 19 detections (−1)"),
    "camera_loss_partial": ("Camera — Loss Partial (0.3)",
        "A contiguous 30% rectangle blacked out.", "18 / 19 detections (−1)"),
    "camera_frame_deletion": ("Camera — Frame Deletion",
        "Whole frame blacked out on even frames (3 of 6 here).", "19 / 19 detections (+0)"),
}

MODALITY_OF = {"radar": "rdr_sparse", "lidar": "ldr64", "camera": "front0"}


def read_pcd(path: str) -> np.ndarray:
    raw = open(path, "rb").read()
    marker = b"DATA binary\n"
    off = raw.index(marker) + len(marker)
    header = raw[:off].decode("ascii")
    n = int([l for l in header.splitlines() if l.startswith("POINTS")][0].split()[1])
    return np.frombuffer(raw[off:], dtype=np.float32).reshape(n, 4)


def frame_index(stem: str) -> str:
    """'000000_seq1' -> '000000', matching the KITTI pred/gt filenames."""
    return stem.split("_")[0]


def render_pair(clean_pcd, corrupt_pcd, lims, title, out_clean, out_corrupt, zeroout,
                kitti_dir: str | None, cond: str, stem: str) -> None:
    clean = read_pcd(clean_pcd)
    render_bev(clean[:, :2], clean[:, 3], lims, f"CLEAN | {title}", out_clean)

    corrupt = read_pcd(corrupt_pcd)
    if zeroout:
        is_zeroed = np.all(corrupt[:, :3] == 0, axis=1)
        render_bev_zeroout(corrupt[:, :2], is_zeroed, lims, f"CORRUPTED | {title}", out_corrupt)
    else:
        render_bev(corrupt[:, :2], corrupt[:, 3], lims, f"CORRUPTED | {title}", out_corrupt)

    if kitti_dir is None:
        return
    fi = frame_index(stem)
    img_w, img_h = RADAR_ZO_IMG if zeroout else RADAR_IMG

    gt = parse_kitti_boxes(os.path.join(kitti_dir, "clean", "gts", f"{fi}.txt"))

    clean_preds = parse_kitti_boxes(os.path.join(kitti_dir, "clean", "preds", f"{fi}.txt"))
    draw_boxes(out_clean, clean_preds, gt, lims, img_w, img_h)

    corrupt_preds_path = os.path.join(kitti_dir, cond, "preds", f"{fi}.txt")
    if os.path.exists(corrupt_preds_path):
        corrupt_preds = parse_kitti_boxes(corrupt_preds_path)
        draw_boxes(out_corrupt, corrupt_preds, gt, lims, img_w, img_h)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default=os.path.expanduser("~/kradar_viewable"))
    ap.add_argument("--dest", default=None, help="default: <bundle>/gallery")
    ap.add_argument("--kitti-dir", default=None,
                    help="<dir>/<condition>/{preds,gts}/<frame>.txt -- overlays detected "
                         "and ground-truth BEV boxes on the radar/lidar renders when set")
    args = ap.parse_args()

    bundle = os.path.abspath(os.path.expanduser(args.bundle))
    dest = args.dest or os.path.join(bundle, "gallery")
    img_dir = os.path.join(dest, "img")
    os.makedirs(img_dir, exist_ok=True)
    kitti_dir = os.path.abspath(os.path.expanduser(args.kitti_dir)) if args.kitti_dir else None
    if kitti_dir and not os.path.isdir(os.path.join(kitti_dir, "clean")):
        print(f"WARNING: {kitti_dir}/clean not found -- boxes will not be drawn")
        kitti_dir = None

    conditions = sorted(
        d for d in os.listdir(bundle)
        if os.path.isdir(os.path.join(bundle, d)) and d not in ("clean", "gallery")
    )

    sections = []
    for cond in conditions:
        modality = cond.split("_")[0]
        key = MODALITY_OF.get(modality)
        title, desc, score = CONDITION_NOTES.get(cond, (cond, "", ""))
        zeroout = any(t in cond for t in ZEROOUT_EFFECTS)
        rows = []

        if key == "front0":
            for src in sorted(glob.glob(os.path.join(bundle, cond, "*_front0.png"))):
                stem = os.path.basename(src).replace("_front0.png", "")
                clean_src = os.path.join(bundle, "clean", f"{stem}_front0.png")
                c_out, k_out = f"img/{cond}_{stem}_corrupt.png", f"img/{cond}_{stem}_clean.png"
                shutil.copy2(src, os.path.join(dest, c_out))
                if os.path.exists(clean_src):
                    shutil.copy2(clean_src, os.path.join(dest, k_out))
                rows.append((stem, k_out, c_out))
        else:
            for src in sorted(glob.glob(os.path.join(bundle, cond, f"*_{key}.pcd"))):
                stem = os.path.basename(src).replace(f"_{key}.pcd", "")
                clean_src = os.path.join(bundle, "clean", f"{stem}_{key}.pcd")
                if not os.path.exists(clean_src):
                    continue
                k_out, c_out = f"img/{cond}_{stem}_clean.png", f"img/{cond}_{stem}_corrupt.png"
                lims = RADAR_LIMS if modality == "radar" else LIDAR_LIMS
                render_pair(clean_src, src, lims, f"{cond} | {stem}",
                            os.path.join(dest, k_out), os.path.join(dest, c_out), zeroout,
                            kitti_dir, cond, stem)
                rows.append((stem, k_out, c_out))

        sections.append((cond, title, desc, score, rows))
        print(f"{cond:30s} {len(rows)} frame(s)")

    _write_html(os.path.join(dest, "index.html"), sections)
    print(f"\nGallery: {os.path.join(dest, 'index.html')}")


def _write_html(path: str, sections) -> None:
    parts = ["""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>ASF Corruption Study — Clean vs Noisy Sensor Data</title>
<style>
body { font-family: -apple-system,BlinkMacSystemFont,sans-serif; max-width:1700px; margin:0 auto;
       padding:20px; background:#111; color:#ddd; }
h1 { color:#e94560; }
h2 { color:#e94560; margin:30px 0 10px; border-bottom:2px solid #333; padding-bottom:5px; }
h3 { color:#aaa; margin:15px 0 5px; font-size:14px; text-transform:uppercase; letter-spacing:1px; }
.frame-box { background:#1a1a2e; border-radius:8px; padding:20px; margin:20px 0; }
.pair-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:8px; margin-bottom:14px; }
.pair-grid img { width:100%; border:1px solid #333; border-radius:4px; }
.caption { text-align:center; font-size:11px; color:#888; margin-bottom:4px; }
.desc { color:#bbb; font-size:13px; margin:6px 0 14px; }
.badge { display:inline-block; font-size:10px; padding:2px 7px; border-radius:3px; margin-left:8px; }
.badge-ok { background:#2d6a4f; color:#fff; }
.badge-crash { background:#e94560; color:#fff; }
.notice { background:#332; border-left:4px solid #e94560; padding:10px 14px; margin:10px 0 20px; font-size:13px; }
table { border-collapse:collapse; margin:10px 0 20px; font-size:13px; }
th,td { border:1px solid #333; padding:5px 10px; text-align:left; }
th { background:#1a1a2e; color:#e94560; }
code { background:#222; padding:1px 5px; border-radius:3px; }
a { color:#e94560; }
</style>
</head>
<body>
<h1>ASF Corruption Study — Clean vs Noisy Sensor Data</h1>
<p>Per-sensor corruption sweep on the K-Radar 6-frame sample (sequences 1 &amp; 2, the 3 densest
frames each). Each condition applies <b>exactly one effect to exactly one modality</b>; every
other sensor is clean. Seed: 42 &middot; conf_thr: 0.3</p>
<div class="notice">
<b>These are the actual tensors handed to the model</b>, dumped at inference time — not
re-simulated afterwards. Every <code>CLEAN</code> / <code>CORRUPTED</code> pair is the same frame
before and after injection. Point clouds are rendered in BEV; the ROI is
<code>x 0–72 m, y ±16 m</code>.
<br><br>
<b>Reading <code>loss_partial</code> and <code>_zeros</code>:</b> these zero points <i>in place</i>
rather than deleting them, so the affected points collapse to the origin. They use the
zero-out-aware renderer (magenta marker + origin inset + counts), matching the existing
<a href="../../noise_visual_inspection/index.html">noise visual inspection</a> gallery.
<br><br>
<b>Detection counts</b> are from <code>corruption_summary.csv</code>; see
<code>outputs/alvis_smoke/RESULTS.md</code> for the full analysis and caveats.
<br><br>
<b>Boxes on the radar/LiDAR views:</b> <span style="color:#3cdc3c">solid green</span> = Sedan
detection, <span style="color:#ffa03c">solid orange</span> = Bus/Truck detection (label shows
confidence score), <span style="color:#00e6e6">dashed cyan</span> = ground truth. The CLEAN panel
always shows the <b>clean run's own predictions</b>; the CORRUPTED panel shows <b>that
condition's own predictions</b> on the corrupted input — so a missing solid box in the
corrupted panel is a real missed detection, not a rendering gap. Ground truth is identical
across all conditions (same frame, same objects) and is shown for reference on both panels.
Two conditions (radar/lidar frame_deletion) crashed the model before writing predictions, so
their surviving frame shows sensor data only, no boxes.
<br><br>
<b>Camera boxes are not shown</b> — the pipeline's KITTI output only carries a placeholder 2D
bbox (<code>50 50 150 150</code> for every detection), not a real image-plane projection, so
there is nothing accurate to overlay on the camera frames.
</div>
<h2>Summary</h2>
<table>
<tr><th>Condition</th><th>Detections (clean = 19)</th></tr>"""]

    for cond, title, desc, score, rows in sections:
        parts.append(f"<tr><td>{title}</td><td>{score}</td></tr>")
    parts.append("</table>")

    for cond, title, desc, score, rows in sections:
        crashed = "abort" in score
        badge = ('<span class="badge badge-crash">MODEL ABORTED</span>' if crashed
                 else '<span class="badge badge-ok">RAN</span>')
        parts.append(f'<div class="frame-box">\n<h2>{title}{badge}</h2>')
        parts.append(f'<p class="desc">{desc}<br><b>Result:</b> {score}</p>')
        if not rows:
            parts.append('<p class="desc">No frames exported.</p>')
        for stem, clean_img, corrupt_img in rows:
            parts.append(f'<h3>{stem}</h3>')
            parts.append('<div class="pair-grid">')
            parts.append(f'<div><div class="caption">CLEAN</div>'
                         f'<a href="{clean_img}"><img src="{clean_img}" alt="clean"></a></div>')
            parts.append(f'<div><div class="caption">CORRUPTED</div>'
                         f'<a href="{corrupt_img}"><img src="{corrupt_img}" alt="corrupted"></a></div>')
            parts.append('</div>')
        parts.append('</div>')

    parts.append("</body>\n</html>\n")
    with open(path, "w") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
