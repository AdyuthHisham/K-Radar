#!/usr/bin/env python3
"""Build ONE self-contained gallery.html per sequence, covering every
inferenced frame across every corruption condition in that sequence's run
tree -- radar/LiDAR BEV and camera, boxed with real predictions/GT.

Structure (like the original smoke study's outputs/alvis_smoke/gallery/
index.html): a summary table, then one section per condition, each showing
every available frame as an ORIGINAL | CORRUPTED pair.

Deduplication: for a given frame, the ORIGINAL panel (radar/LiDAR/camera) is
rendered ONCE per modality and reused across every condition section -- not
re-rendered per condition. Any single corrupted condition's sensor_dump
carries a "*_clean.npy"/"*_front0_clean.png" pre-injection backup that is
byte-identical to every other condition's backup for the same frame (all
copied from the same underlying dataset item before injection runs); the
`original` run itself has no --dump-sensors dump (NOISE_CFG is empty there),
so one corrupted condition's backups are used as the original-panel source.
This cuts per-sequence image count roughly 3x versus rendering original
panels fresh in every condition section, with no loss of content.

Usage:
    python scripts/alvis/build_sequence_full_gallery.py \\
        --seq 8 \\
        --run-root outputs/alvis_seq8 \\
        --out outputs/alvis_seq8/gallery.html
"""
from __future__ import annotations

import argparse
import ast
import base64
import glob
import os
import re
import shutil
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
CAM_CROP = (96, 170, 800, 426)


def find_kitti_dirs(run_dir: str) -> tuple[str, str] | None:
    matches = glob.glob(os.path.join(run_dir, "exp_*", "test_kitti", "*", "*", "all", "preds"))
    if not matches:
        return None
    preds_dir = max(matches, key=os.path.getmtime)
    gts_dir = os.path.join(os.path.dirname(preds_dir), "gts")
    return preds_dir, gts_dir


def find_sensor_dump(run_dir: str) -> str | None:
    path = os.path.join(run_dir, "sensor_dump")
    return path if os.path.isdir(path) else None


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


def b64_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def render_original_frame(fid: str, stem: str, dump_dir: str, out_dir: str,
                           preds: list, gt: list, r2i: np.ndarray, aug: np.ndarray,
                           raw_cam_dir: str | None) -> dict:
    """Render the shared ORIGINAL panels for one frame, sourced from a
    corrupted condition's pre-injection "_clean" backups (see module
    docstring)."""
    rdr = np.load(os.path.join(dump_dir, f"{stem}_rdr_sparse_clean.npy"))
    ldr = np.load(os.path.join(dump_dir, f"{stem}_ldr64_clean.npy"))

    radar_path = os.path.join(out_dir, f"{fid}_radar_original.png")
    render_bev(rdr[:, :2], rdr[:, 3], RADAR_LIMS, f"radar ORIGINAL | {fid}",
               radar_path, img_w=RADAR_IMG[0], img_h=RADAR_IMG[1])
    draw_boxes(radar_path, preds, gt, RADAR_LIMS, RADAR_IMG[0], RADAR_IMG[1])

    lidar_path = os.path.join(out_dir, f"{fid}_lidar_original.png")
    render_bev(ldr[:, :2], ldr[:, 3], LIDAR_LIMS, f"lidar ORIGINAL | {fid}",
               lidar_path, img_w=LIDAR_IMG[0], img_h=LIDAR_IMG[1])
    draw_boxes(lidar_path, preds, gt, LIDAR_LIMS, LIDAR_IMG[0], LIDAR_IMG[1])

    cam_path = os.path.join(out_dir, f"{fid}_camera_original.png")
    front0_clean = os.path.join(dump_dir, f"{stem}_front0_clean.png")
    front0_corrupt = os.path.join(dump_dir, f"{stem}_front0.png")
    shutil.copy2(front0_clean if os.path.exists(front0_clean) else front0_corrupt, cam_path)
    draw_camera_boxes(cam_path, preds, gt, r2i, aug, CAM_IMG[0], CAM_IMG[1])

    result = {
        "radar": b64_file(radar_path), "lidar": b64_file(lidar_path), "camera": b64_file(cam_path),
        "gt_count": len(gt), "pred_count": len(preds),
    }

    meta = parse_meta(os.path.join(dump_dir, f"{stem}_meta.txt"))
    if raw_cam_dir and "camf" in meta["sensor_idx"]:
        camf = meta["sensor_idx"]["camf"]
        raw_path = os.path.join(raw_cam_dir, f"cam-front_{camf}.png")
        if os.path.exists(raw_path):
            from PIL import Image
            im = Image.open(raw_path).convert("RGB")
            w, h = im.size
            half = w // 2
            front1 = im.crop((half, 0, w, h)).resize((int(half * CAM_RESIZE), int(h * CAM_RESIZE)))
            front1 = front1.crop(CAM_CROP)
            front1_path = os.path.join(out_dir, f"{fid}_camera_front1.png")
            front1.save(front1_path)
            result["cam_front1"] = b64_file(front1_path)

    return result


def render_corrupted_frame(fid: str, stem: str, dump_dir: str, out_dir: str, condition: str,
                            preds: list, gt: list, meta: dict, r2i: np.ndarray,
                            aug: np.ndarray) -> dict:
    """Render only the CORRUPTED-branch panels for one (frame, condition)
    pair. Which modality actually changed is read from meta.txt's
    changed=True/False per key, not assumed from the condition name."""
    result: dict = {"changed": meta["changed"]}

    rdr_path = os.path.join(dump_dir, f"{stem}_rdr_sparse.npy")
    if os.path.exists(rdr_path):
        rdr = np.load(rdr_path)
        out_path = os.path.join(out_dir, f"{condition}_{fid}_radar.png")
        render_bev(rdr[:, :2], rdr[:, 3], RADAR_LIMS, f"radar {condition} | {fid}",
                   out_path, img_w=RADAR_IMG[0], img_h=RADAR_IMG[1])
        draw_boxes(out_path, preds, gt, RADAR_LIMS, RADAR_IMG[0], RADAR_IMG[1])
        result["radar"] = b64_file(out_path)

    ldr_path = os.path.join(dump_dir, f"{stem}_ldr64.npy")
    if os.path.exists(ldr_path):
        ldr = np.load(ldr_path)
        out_path = os.path.join(out_dir, f"{condition}_{fid}_lidar.png")
        render_bev(ldr[:, :2], ldr[:, 3], LIDAR_LIMS, f"lidar {condition} | {fid}",
                   out_path, img_w=LIDAR_IMG[0], img_h=LIDAR_IMG[1])
        draw_boxes(out_path, preds, gt, LIDAR_LIMS, LIDAR_IMG[0], LIDAR_IMG[1])
        result["lidar"] = b64_file(out_path)

    front0_path = os.path.join(dump_dir, f"{stem}_front0.png")
    if os.path.exists(front0_path):
        out_path = os.path.join(out_dir, f"{condition}_{fid}_camera.png")
        shutil.copy2(front0_path, out_path)
        draw_camera_boxes(out_path, preds, gt, r2i, aug, CAM_IMG[0], CAM_IMG[1])
        result["camera"] = b64_file(out_path)

    result["pred_count"] = len(preds)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--run-root", required=True,
                     help="e.g. outputs/alvis_seq8, containing original/ and one dir per condition")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--raw-cam-dir", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    calib_path = args.calib or os.path.join(_REPO, "resources", "cam_calib", "T_params_seq", str(args.seq))
    raw_cam_dir = args.raw_cam_dir or os.path.join(_REPO, "data", str(args.seq), "cam-front")
    if not os.path.isdir(raw_cam_dir):
        raw_cam_dir = None

    with open(calib_path, "rb") as f:
        calib = pickle.load(f)["front0"]
    r2i, aug = calib["radar2image"], calib["img_aug_matrix"]

    original_dir = os.path.join(args.run_root, "original")
    original_kitti = find_kitti_dirs(original_dir)
    if original_kitti is None:
        raise SystemExit(f"No original-run KITTI results found under {original_dir}")
    original_preds_dir, gts_dir = original_kitti

    frame_ids = sorted(os.path.splitext(f)[0] for f in os.listdir(original_preds_dir)
                        if f.endswith(".txt"))
    if not frame_ids:
        raise SystemExit(f"No frames found in {original_preds_dir}")

    condition_dirs = sorted(
        d for d in os.listdir(args.run_root)
        if os.path.isdir(os.path.join(args.run_root, d)) and d != "original"
        and not d.startswith("_")
    )

    out_dir = os.path.dirname(os.path.abspath(args.out))
    render_dir = os.path.join(out_dir, f"_gallery_render_tmp_seq{args.seq}")
    os.makedirs(render_dir, exist_ok=True)

    # Find one condition with a usable sensor_dump to source the shared
    # ORIGINAL panels' pre-injection backups from (the `original` run itself
    # never dumps sensor data -- NOISE_CFG is empty there).
    original_source_dir = None
    for cond in condition_dirs:
        dump = find_sensor_dump(os.path.join(args.run_root, cond))
        if dump and all(os.path.exists(os.path.join(dump, f"{frame_ids[0]}_seq{args.seq}_{k}"))
                         for k in ("rdr_sparse_clean.npy", "ldr64_clean.npy", "meta.txt")):
            original_source_dir = dump
            break
    if original_source_dir is None:
        raise SystemExit("No condition with a usable sensor_dump found -- cannot source "
                          "original-panel backups (every corrupted condition may have aborted).")

    print(f"Sourcing shared ORIGINAL panels from: {original_source_dir}")
    originals: dict[str, dict] = {}
    for fid in frame_ids:
        stem = f"{fid}_seq{args.seq}"
        gt = parse_kitti_boxes(os.path.join(gts_dir, f"{fid}.txt"))
        preds = parse_kitti_boxes(os.path.join(original_preds_dir, f"{fid}.txt"))
        originals[fid] = render_original_frame(fid, stem, original_source_dir, render_dir,
                                                 preds, gt, r2i, aug, raw_cam_dir)
        print(f"original {fid}: gt={originals[fid]['gt_count']} pred={originals[fid]['pred_count']}")

    conditions_data = []
    for cond in condition_dirs:
        cond_dir = os.path.join(args.run_root, cond)
        dump = find_sensor_dump(cond_dir)
        kitti = find_kitti_dirs(cond_dir)
        if dump is None or kitti is None:
            conditions_data.append({"name": cond, "status": "no_predictions", "frames": {}})
            print(f"{cond}: no_predictions (model likely aborted)")
            continue
        preds_dir, _ = kitti
        frames = {}
        for fid in frame_ids:
            stem = f"{fid}_seq{args.seq}"
            meta_path = os.path.join(dump, f"{stem}_meta.txt")
            if not os.path.exists(meta_path):
                continue
            meta = parse_meta(meta_path)
            gt = parse_kitti_boxes(os.path.join(gts_dir, f"{fid}.txt"))
            preds = parse_kitti_boxes(os.path.join(preds_dir, f"{fid}.txt"))
            frames[fid] = render_corrupted_frame(fid, stem, dump, render_dir, cond,
                                                  preds, gt, meta, r2i, aug)
        conditions_data.append({"name": cond, "status": "ok" if frames else "no_predictions",
                                 "frames": frames})
        print(f"{cond}: {len(frames)} frame(s) rendered")

    html = _build_html(args.seq, frame_ids, originals, conditions_data)
    with open(args.out, "w") as f:
        f.write(html)

    shutil.rmtree(render_dir, ignore_errors=True)
    print(f"\nGallery written: {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


def _build_html(seq: str, frame_ids: list[str], originals: dict[str, dict],
                 conditions_data: list[dict]) -> str:
    parts = [f'''<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Sequence {seq} — Full Corruption Gallery</title>
<style>
:root {{
  --bg: #0c0d11; --panel: #161822; --panel-2: #1c1f2c; --border: #2a2d3d;
  --text: #dcdce2; --text-dim: #8b8ea3; --accent: #e94560; --ok: #2d9d6f; --warn: #e8a33d;
  --mono: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--text); font-family: var(--sans); margin: 0; padding: 28px; line-height: 1.55; max-width: 1400px; margin: 0 auto; }}
h1 {{ font-family: var(--mono); font-size: 24px; margin: 0 0 6px; }}
h2 {{ font-family: var(--mono); font-size: 15px; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 8px; margin: 40px 0 16px; }}
.dek {{ color: var(--text-dim); font-size: 13px; margin: 0 0 20px; }}
.badge {{ display: inline-block; font-size: 10px; padding: 2px 7px; border-radius: 3px; margin-left: 8px; font-family: var(--mono); }}
.badge-ok {{ background: rgba(45,157,111,0.18); color: var(--ok); }}
.badge-abort {{ background: rgba(233,69,96,0.18); color: var(--accent); }}
table {{ border-collapse: collapse; width: 100%; font-family: var(--mono); font-size: 12px; margin-bottom: 24px; }}
th, td {{ padding: 6px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
th {{ color: var(--text-dim); font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; font-size: 10px; }}
.frame {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; margin-bottom: 16px; }}
.frame-head {{ padding: 8px 14px; border-bottom: 1px solid var(--border); font-family: var(--mono); font-size: 11px; color: var(--text-dim); }}
.pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--border); }}
.pair figure {{ margin: 0; background: var(--panel); }}
.pair img {{ width: 100%; display: block; }}
.pair figcaption {{ font-family: var(--mono); font-size: 10px; text-align: center; padding: 6px; color: var(--text-dim); text-transform: uppercase; }}
.orig-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }}
.orig-grid figure {{ margin: 0; background: var(--panel); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }}
.orig-grid img {{ width: 100%; display: block; }}
.orig-grid figcaption {{ font-family: var(--mono); font-size: 10px; text-align: center; padding: 6px; color: var(--text-dim); }}
</style>
</head>
<body>
<h1>Sequence {seq} &middot; Full Corruption Gallery</h1>
<p class="dek">{len(frame_ids)} inferenced frames &times; {len(conditions_data)} corrupted conditions (+ 1 original baseline). ORIGINAL panels are shared across all condition sections below (rendered once per frame, not per condition).</p>
<h2>Summary</h2>
<table>
<tr><th>Condition</th><th>Status</th><th>Frames rendered</th></tr>
''']
    for c in conditions_data:
        badge = "badge-ok" if c["status"] == "ok" else "badge-abort"
        label = "ran" if c["status"] == "ok" else "no predictions"
        parts.append(f'<tr><td>{c["name"]}</td><td><span class="badge {badge}">{label}</span></td>'
                     f'<td>{len(c["frames"])}/{len(frame_ids)}</td></tr>\n')
    parts.append("</table>\n")

    parts.append('<h2>Original baseline<span class="badge badge-ok">reference</span></h2>\n')
    parts.append('<div class="frames-original">\n')
    for fid in frame_ids:
        o = originals[fid]
        parts.append(f'<div class="frame"><div class="frame-head">FRAME {fid} &middot; '
                     f'GT {o["gt_count"]} &middot; PRED {o["pred_count"]}</div>\n'
                     f'<div class="orig-grid">\n'
                     f'<figure><img src="data:image/png;base64,{o["radar"]}"><figcaption>radar</figcaption></figure>\n'
                     f'<figure><img src="data:image/png;base64,{o["lidar"]}"><figcaption>lidar</figcaption></figure>\n'
                     f'<figure><img src="data:image/png;base64,{o["camera"]}"><figcaption>camera front0</figcaption></figure>\n')
        if "cam_front1" in o:
            parts.append(f'<figure><img src="data:image/png;base64,{o["cam_front1"]}">'
                         f'<figcaption>camera front1 (unused by ASF)</figcaption></figure>\n')
        parts.append('</div></div>\n')
    parts.append('</div>\n')

    for c in conditions_data:
        badge = ('<span class="badge badge-ok">ran</span>' if c["status"] == "ok"
                 else '<span class="badge badge-abort">no predictions</span>')
        parts.append(f'<h2>{c["name"]}{badge}</h2>\n')
        if not c["frames"]:
            parts.append('<p class="dek">No predictions for this condition '
                         '(model likely aborted on this input).</p>\n')
            continue
        for fid in frame_ids:
            if fid not in c["frames"]:
                continue
            cf = c["frames"][fid]
            o = originals[fid]
            parts.append(f'<div class="frame"><div class="frame-head">FRAME {fid} &middot; '
                         f'ORIGINAL PRED {o["pred_count"]} &middot; CORRUPTED PRED {cf["pred_count"]}</div>\n')
            for modality, label in (("radar", "RADAR"), ("lidar", "LIDAR"), ("camera", "CAMERA")):
                if modality not in cf:
                    continue
                changed = cf["changed"].get(
                    "rdr_sparse" if modality == "radar" else
                    "ldr64" if modality == "lidar" else "front0", None)
                corrupt_label = "corrupted" if changed else "original (identical input)"
                parts.append(f'<div class="pair">\n'
                             f'<figure><img src="data:image/png;base64,{o[modality]}">'
                             f'<figcaption>{label} original</figcaption></figure>\n'
                             f'<figure><img src="data:image/png;base64,{cf[modality]}">'
                             f'<figcaption>{label} {corrupt_label}</figcaption></figure>\n'
                             f'</div>\n')
            parts.append('</div>\n')

    parts.append("</body>\n</html>\n")
    return "".join(parts)


if __name__ == "__main__":
    main()
