#!/usr/bin/env python3
"""Build a self-contained local HTML gallery for one (sequence, corrupted
condition) pair: radar/LiDAR BEV (original + corrupted panels, boxed with
predictions/GT) and camera (original + corrupted, real projected 3D boxes),
for a random sample of frames.

Promoted from ad hoc scratch rendering done during the sequence-46
camera_gaussian_high pilot this session into a reusable, parameterized
script. Deliberately does NOT reuse build_gallery.py's bundle-layout
assumptions (hardcoded "clean" dirname, "<modality>_<effect>" condition-name
parsing, baked-in CONDITION_NOTES) -- those are incompatible with the flat
--dump-sensors file layout and multi-sequence/multi-condition use this needs.
build_gallery.py stays as-is for the original smoke study.

Reuses, unchanged: render_bev/render_bev_zeroout (datasets/effects/
visualize_effects_v2.py), parse_kitti_boxes/draw_boxes/draw_camera_boxes
(scripts/alvis/kitti_boxes.py).

Terminology: "original" (not "clean") throughout captions/labels, per
established convention -- except where quoting a literal on-disk identifier
(e.g. a directory actually named "clean").

Usage:
    python scripts/alvis/build_seq_gallery.py \\
        --seq 46 \\
        --original-run-dir outputs/alvis_seq46/original \\
        --corrupted-run-dir outputs/alvis_seq46/camera_gaussian_noise_high \\
        --dump-dir outputs/alvis_seq46/camera_gaussian_noise_high/sensor_dump \\
        --n-frames 15 --seed 42 \\
        --out outputs/alvis_seq46/camera_gaussian_noise_high/gallery.html
"""
from __future__ import annotations

import argparse
import ast
import base64
import glob
import os
import random
import re
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "datasets", "effects"))
sys.path.insert(0, os.path.join(_REPO, "scripts", "alvis"))

import numpy as np
import pickle
from visualize_effects_v2 import render_bev  # noqa: E402
from kitti_boxes import parse_kitti_boxes, draw_boxes, draw_camera_boxes  # noqa: E402

RADAR_LIMS = (-5.0, 80.0, -25.0, 25.0)
LIDAR_LIMS = (-5.0, 80.0, -25.0, 25.0)
RADAR_IMG = (900, 700)
LIDAR_IMG = (900, 700)
CAM_IMG = (704, 256)
CAM_RESIZE = 0.7
CAM_CROP = (96, 170, 800, 426)  # matches cam_process in the ASF_v2_0_seq*_alvis.yml configs


def find_kitti_dirs(run_dir: str) -> tuple[str, str] | None:
    """Locate the most recent (preds_dir, gts_dir) pair under a run directory."""
    matches = glob.glob(os.path.join(run_dir, "exp_*", "test_kitti", "*", "*", "all", "preds"))
    if not matches:
        return None
    preds_dir = max(matches, key=os.path.getmtime)
    gts_dir = os.path.join(os.path.dirname(preds_dir), "gts")
    return preds_dir, gts_dir


def parse_meta(path: str) -> dict:
    meta = {"changed": {}, "sensor_idx": {}}
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r"^sensor_idx:\s*(\{.*\})$", line)
            if m:
                meta["sensor_idx"] = ast.literal_eval(m.group(1))
                continue
            m = re.match(r"^(\w+):.*changed=(True|False)", line)
            if m:
                meta["changed"][m.group(1)] = m.group(2) == "True"
    return meta


def load_calib(calib_path: str) -> dict:
    with open(calib_path, "rb") as f:
        return pickle.load(f)["front0"]


def b64_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def render_frame(fid: str, seq: str, dump_dir: str, out_dir: str, original_preds: list, gt: list,
                  corrupted_preds: list, r2i: np.ndarray, aug: np.ndarray,
                  raw_cam_dir: str | None) -> dict:
    # Dump filenames are "{idx:06d}_seq{seq}_{key}..." (eval_wrapper.py:_dump),
    # not just "{idx:06d}_{key}...".
    stem = f"{fid}_seq{seq}"
    meta = parse_meta(os.path.join(dump_dir, f"{stem}_meta.txt"))

    # radar: original + corrupted panels (identical unless radar was the corrupted modality)
    rdr_o = np.load(os.path.join(dump_dir, f"{stem}_rdr_sparse_clean.npy"))
    rdr_c = np.load(os.path.join(dump_dir, f"{stem}_rdr_sparse.npy"))
    radar_o_path = os.path.join(out_dir, f"{fid}_radar_original.png")
    radar_c_path = os.path.join(out_dir, f"{fid}_radar_corrupt.png")
    render_bev(rdr_o[:, :2], rdr_o[:, 3], RADAR_LIMS, f"radar ORIGINAL | {fid}",
               radar_o_path, img_w=RADAR_IMG[0], img_h=RADAR_IMG[1])
    draw_boxes(radar_o_path, original_preds, gt, RADAR_LIMS, RADAR_IMG[0], RADAR_IMG[1])
    render_bev(rdr_c[:, :2], rdr_c[:, 3], RADAR_LIMS, f"radar CORRUPT | {fid}",
               radar_c_path, img_w=RADAR_IMG[0], img_h=RADAR_IMG[1])
    draw_boxes(radar_c_path, original_preds, gt, RADAR_LIMS, RADAR_IMG[0], RADAR_IMG[1])

    # lidar: original + corrupted panels
    ldr_o = np.load(os.path.join(dump_dir, f"{stem}_ldr64_clean.npy"))
    ldr_c = np.load(os.path.join(dump_dir, f"{stem}_ldr64.npy"))
    lidar_o_path = os.path.join(out_dir, f"{fid}_lidar_original.png")
    lidar_c_path = os.path.join(out_dir, f"{fid}_lidar_corrupt.png")
    render_bev(ldr_o[:, :2], ldr_o[:, 3], LIDAR_LIMS, f"lidar ORIGINAL | {fid}",
               lidar_o_path, img_w=LIDAR_IMG[0], img_h=LIDAR_IMG[1])
    draw_boxes(lidar_o_path, original_preds, gt, LIDAR_LIMS, LIDAR_IMG[0], LIDAR_IMG[1])
    render_bev(ldr_c[:, :2], ldr_c[:, 3], LIDAR_LIMS, f"lidar CORRUPT | {fid}",
               lidar_c_path, img_w=LIDAR_IMG[0], img_h=LIDAR_IMG[1])
    draw_boxes(lidar_c_path, original_preds, gt, LIDAR_LIMS, LIDAR_IMG[0], LIDAR_IMG[1])

    # camera: original + corrupted, real projected 3D boxes (each panel its own predictions)
    import shutil
    cam_o_path = os.path.join(out_dir, f"{fid}_camera_original.png")
    cam_c_path = os.path.join(out_dir, f"{fid}_camera_corrupt.png")
    front0_clean = os.path.join(dump_dir, f"{stem}_front0_clean.png")
    front0_corrupt = os.path.join(dump_dir, f"{stem}_front0.png")
    shutil.copy2(front0_clean if os.path.exists(front0_clean) else front0_corrupt, cam_o_path)
    shutil.copy2(front0_corrupt, cam_c_path)
    draw_camera_boxes(cam_o_path, original_preds, gt, r2i, aug, CAM_IMG[0], CAM_IMG[1])
    draw_camera_boxes(cam_c_path, corrupted_preds, gt, r2i, aug, CAM_IMG[0], CAM_IMG[1])

    result = {
        "radar_original": b64_file(radar_o_path), "radar_corrupt": b64_file(radar_c_path),
        "lidar_original": b64_file(lidar_o_path), "lidar_corrupt": b64_file(lidar_c_path),
        "cam_original": b64_file(cam_o_path), "cam_corrupt": b64_file(cam_c_path),
        "changed": meta["changed"],
        "gt_count": len(gt), "original_pred_count": len(original_preds),
        "corrupted_pred_count": len(corrupted_preds),
    }

    # front1 stereo reference (unused by ASF -- see camera-corruption gallery
    # note this session established: cam: {front1: False} in every ASF_v2_0_seq*
    # config, zero front1 files ever appear in a --dump-sensors dump).
    if raw_cam_dir and "camf" in meta["sensor_idx"]:
        camf = meta["sensor_idx"]["camf"]
        raw_path = os.path.join(raw_cam_dir, f"cam-front_{camf}.png")
        if os.path.exists(raw_path):
            from PIL import Image
            im = Image.open(raw_path).convert("RGB")
            w, h = im.size
            half = w // 2
            front1 = im.crop((half, 0, w, h))
            front1 = front1.resize((int(half * CAM_RESIZE), int(h * CAM_RESIZE)))
            front1 = front1.crop(CAM_CROP)
            front1_path = os.path.join(out_dir, f"{fid}_camera_front1.png")
            front1.save(front1_path)
            result["cam_front1"] = b64_file(front1_path)

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--original-run-dir", required=True)
    ap.add_argument("--corrupted-run-dir", required=True)
    ap.add_argument("--dump-dir", required=True,
                     help="sensor_dump directory from the corrupted run (--dump-sensors)")
    ap.add_argument("--calib", default=None,
                     help="default: resources/cam_calib/T_params_seq/<seq>")
    ap.add_argument("--raw-cam-dir", default=None,
                     help="default: data/<seq>/cam-front (for the front1 reference view)")
    ap.add_argument("--n-frames", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    calib_path = args.calib or os.path.join(_REPO, "resources", "cam_calib", "T_params_seq", str(args.seq))
    raw_cam_dir = args.raw_cam_dir or os.path.join(_REPO, "data", str(args.seq), "cam-front")
    if not os.path.isdir(raw_cam_dir):
        raw_cam_dir = None

    calib = load_calib(calib_path)
    r2i, aug = calib["radar2image"], calib["img_aug_matrix"]

    original_dirs = find_kitti_dirs(args.original_run_dir)
    corrupted_dirs = find_kitti_dirs(args.corrupted_run_dir)
    if original_dirs is None:
        raise SystemExit(f"No KITTI preds/gts found under {args.original_run_dir}")
    if corrupted_dirs is None:
        raise SystemExit(f"No KITTI preds/gts found under {args.corrupted_run_dir} "
                          "(model may have aborted for this condition -- no gallery possible)")
    original_preds_dir, gts_dir = original_dirs
    corrupted_preds_dir, _ = corrupted_dirs

    available = sorted({os.path.splitext(os.path.basename(p))[0].split("_")[0]
                         for p in glob.glob(os.path.join(args.dump_dir, "*_meta.txt"))})
    if not available:
        raise SystemExit(f"No *_meta.txt files found under {args.dump_dir}")

    rng = random.Random(args.seed)
    n = min(args.n_frames, len(available))
    frame_ids = sorted(rng.sample(available, n))

    out_dir = os.path.dirname(os.path.abspath(args.out))
    render_dir = os.path.join(out_dir, "_gallery_render_tmp")
    os.makedirs(render_dir, exist_ok=True)

    frames_data = []
    for fid in frame_ids:
        gt = parse_kitti_boxes(os.path.join(gts_dir, f"{fid}.txt"))
        original_preds = parse_kitti_boxes(os.path.join(original_preds_dir, f"{fid}.txt"))
        corrupted_preds = parse_kitti_boxes(os.path.join(corrupted_preds_dir, f"{fid}.txt"))
        data = render_frame(fid, args.seq, args.dump_dir, render_dir, original_preds, gt,
                             corrupted_preds, r2i, aug, raw_cam_dir)
        frames_data.append((fid, data))
        print(f"{fid}: gt={data['gt_count']} original_pred={data['original_pred_count']} "
              f"corrupted_pred={data['corrupted_pred_count']}")

    condition_name = os.path.basename(os.path.normpath(args.corrupted_run_dir))
    html = _build_html(args.seq, condition_name, frames_data)
    with open(args.out, "w") as f:
        f.write(html)

    import shutil as _shutil
    _shutil.rmtree(render_dir, ignore_errors=True)

    print(f"\nGallery written: {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


def _pair_row(label: str, d: dict, key_o: str, key_c: str, cap_o: str, cap_c: str) -> str:
    return f'''
  <div class="modality-row">
    <div class="mod-label">{label}</div>
    <div class="pair">
      <figure><img src="data:image/png;base64,{d[key_o]}" alt="{label} original"><figcaption>{cap_o}</figcaption></figure>
      <figure><img src="data:image/png;base64,{d[key_c]}" alt="{label} corrupted"><figcaption>{cap_c}</figcaption></figure>
    </div>
  </div>'''


def _frame_card(fid: str, d: dict) -> str:
    delta = d["corrupted_pred_count"] - d["original_pred_count"]
    delta_cls = "delta-up" if delta > 0 else ("delta-down" if delta < 0 else "delta-flat")
    delta_str = f'<span class="delta {delta_cls}">{"+" if delta > 0 else ""}{delta}</span>'

    radar_changed = d["changed"].get("rdr_sparse", False)
    lidar_changed = d["changed"].get("ldr64", False)
    radar_cap = "corrupted" if radar_changed else "original (identical input)"
    lidar_cap = "corrupted" if lidar_changed else "original (identical input)"

    radar_row = _pair_row("RADAR", d, "radar_original", "radar_corrupt", "original", radar_cap)
    lidar_row = _pair_row("LIDAR", d, "lidar_original", "lidar_corrupt", "original", lidar_cap)
    cam_row = _pair_row("CAMERA &middot; FRONT0 (model input)", d, "cam_original", "cam_corrupt",
                         "original", "corrupted")

    front1_row = ""
    if "cam_front1" in d:
        front1_row = f'''
  <div class="modality-row">
    <div class="mod-label">CAMERA &middot; FRONT1 <span class="unused-tag">not used by ASF &mdash; reference only</span></div>
    <img class="single" src="data:image/png;base64,{d['cam_front1']}" alt="front1 stereo reference">
  </div>'''

    return f'''
<article class="frame">
  <div class="frame-head">
    <span class="frame-id">FRAME {fid}</span>
    <span class="frame-stats">GT <b>{d['gt_count']}</b> &nbsp;&middot;&nbsp; ORIGINAL <b>{d['original_pred_count']}</b> &nbsp;&middot;&nbsp; CORRUPT <b>{d['corrupted_pred_count']}</b> {delta_str}</span>
  </div>
  {radar_row}
  {lidar_row}
  {cam_row}
  {front1_row}
</article>'''


def _build_html(seq: str, condition_name: str, frames_data: list[tuple[str, dict]]) -> str:
    cards = "\n".join(_frame_card(fid, d) for fid, d in frames_data)
    return f'''<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Sequence {seq} — {condition_name}</title>
<style>
:root {{
  --bg: #0c0d11; --panel: #161822; --panel-2: #1c1f2c; --border: #2a2d3d;
  --text: #dcdce2; --text-dim: #8b8ea3; --accent: #e94560; --warn: #e8a33d;
  --mono: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--text); font-family: var(--sans); margin: 0; padding: 28px; line-height: 1.55; }}
h1 {{ font-family: var(--mono); font-size: 22px; margin: 0 0 6px; }}
.dek {{ color: var(--text-dim); font-size: 13px; margin: 0 0 24px; }}
.frames-grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; max-width: 1000px; margin: 0 auto; }}
.frame {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
.frame-head {{ display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; border-bottom: 1px solid var(--border); font-family: var(--mono); font-size: 11px; }}
.frame-id {{ color: var(--text-dim); letter-spacing: 0.05em; }}
.frame-stats {{ color: var(--text-dim); font-variant-numeric: tabular-nums; }}
.frame-stats b {{ color: var(--text); font-weight: 600; }}
.delta {{ padding: 1px 6px; border-radius: 3px; margin-left: 4px; font-weight: 600; }}
.delta-up {{ background: rgba(45,157,111,0.18); color: #2d9d6f; }}
.delta-down {{ background: rgba(233,69,96,0.18); color: var(--accent); }}
.delta-flat {{ background: var(--panel-2); color: var(--text-dim); }}
.modality-row {{ padding: 12px 14px; border-bottom: 1px solid var(--border); }}
.modality-row:last-child {{ border-bottom: none; }}
.mod-label {{ font-family: var(--mono); font-size: 10px; letter-spacing: 0.1em; color: var(--text-dim); margin-bottom: 8px; }}
.pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--border); }}
.pair figure {{ margin: 0; background: var(--panel); }}
.pair img {{ width: 100%; display: block; }}
.pair figcaption {{ font-family: var(--mono); font-size: 10px; text-align: center; padding: 6px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.06em; }}
img.single {{ width: 100%; max-width: 704px; display: block; border-radius: 4px; border: 1px dashed var(--border); opacity: 0.85; }}
.unused-tag {{ font-family: var(--sans); text-transform: none; letter-spacing: normal; font-size: 10px; color: var(--warn); background: rgba(232,163,61,0.12); padding: 1px 6px; border-radius: 3px; margin-left: 6px; }}
@media (max-width: 640px) {{ .pair {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>Sequence {seq} &middot; {condition_name}</h1>
<p class="dek">{len(frames_data)} randomly sampled frames, original vs. corrupted panels for radar/LiDAR/camera, boxed with real predictions/ground truth.</p>
<div class="frames-grid">
{cards}
</div>
</body>
</html>
'''


if __name__ == "__main__":
    main()
