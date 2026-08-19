#!/usr/bin/env python3
"""Render a review gallery for reconstruct_object_ids.py's Stage A/B output:
random false positives, all false negatives, and all unmatched-GT lines, as
BEV LiDAR crops with GT (dashed cyan) and predicted (solid) boxes drawn, same
visual language as outputs/alvis_smoke/gallery/index.html.

Read-only: uses KRadarFusion_v1_0(cfg, split='test') only to load LiDAR
points (get_ldr64) and to read dict_item['meta']['label'] (native-coordinate
GT boxes, no re-derivation from the rounded KITTI text needed). Reuses
render_bev (datasets/effects/visualize_effects_v2.py) and the BEV corner
geometry from scripts/alvis/kitti_boxes.py so a box lands on the same pixels
it would in the existing corruption-study gallery.

Usage (inside the same Apptainer container as recon_seq32.sbatch):
    python scripts/alvis/render_recon_review.py \
        --outputs-dir outputs/object_id_recon \
        --seq 32 \
        --configs-dir configs \
        --n-fp 5 \
        --seed 0 \
        --out-dir outputs/object_id_recon/review_gallery
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import random
import sys

import cv2
import numpy as np

_REPO = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, osp.join(_REPO, "datasets", "effects"))
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, _REPO)

from visualize_effects_v2 import render_bev, _normalize_to_pixel  # noqa: E402
from kitti_boxes import _corners_bev, _dashed_polylines  # noqa: E402
from utils.util_config import cfg, cfg_from_yaml_file  # noqa: E402
from datasets.kradar_fusion_v1_0 import KRadarFusion_v1_0  # noqa: E402

LIDAR_LIMS = (-5.0, 80.0, -25.0, 25.0)
IMG_W, IMG_H = 900, 700

_CLASS_COLOR = {"sed": (60, 220, 60), "bus": (60, 160, 255)}
_GT_COLOR = (230, 230, 0)
_HIGHLIGHT_COLOR = (0, 0, 255)  # bright red ring around the flagged object

CLS_KW_TO_NAME = {v: k for k, v in {
    "Sedan": "sed", "Bus or Truck": "bus", "Motorcycle": "mot",
    "Bicycle": "bic", "Bicycle Group": "big",
    "Pedestrian": "ped", "Pedestrian Group": "peg",
}.items()}


def load_gts(path: str) -> dict[tuple[str, str], list[dict]]:
    """seq32_gts_with_trkid.txt: tag cls_kw trk_id ambiguous h w l loc_x loc_y loc_z rot_y"""
    out: dict[tuple[str, str], list[dict]] = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            seq, frame = p[0].split("_", 1)
            row = dict(cls_kw=p[1], trk_id=p[2], ambiguous=p[3],
                       h=float(p[4]), w=float(p[5]), l=float(p[6]),
                       loc_x=float(p[7]), loc_y=float(p[8]), loc_z=float(p[9]), rot_y=float(p[10]))
            out.setdefault((seq, frame), []).append(row)
    return out


def load_preds(path: str) -> dict[tuple[str, str, str], list[dict]]:
    """seq32_preds_with_trkid.txt: tag cls_kw trk_id is_fp h w l loc_x loc_y loc_z rot_y score"""
    out: dict[tuple[str, str, str], list[dict]] = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            seq, frame, condition = p[0].split("_", 2)
            row = dict(cls_kw=p[1], trk_id=p[2], is_fp=p[3] == "1",
                       h=float(p[4]), w=float(p[5]), l=float(p[6]),
                       loc_x=float(p[7]), loc_y=float(p[8]), loc_z=float(p[9]), rot_y=float(p[10]),
                       score=float(p[11]))
            out.setdefault((seq, frame, condition), []).append(row)
    return out


def load_fn(path: str) -> list[tuple[str, str, str, int]]:
    """seq32_fn.txt: '{seq}_{frame}_{condition} gt_line_idx={idx}'"""
    out = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            seq, frame, condition = p[0].split("_", 2)
            gt_line_idx = int(p[1].split("=")[1])
            out.append((seq, frame, condition, gt_line_idx))
    return out


def load_unmatched_gt(path: str) -> list[dict]:
    """seq32_gts_unmatched.txt: tag cls_kw h w l loc_x loc_y loc_z rot_y"""
    out = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            seq, frame = p[0].split("_", 1)
            out.append(dict(seq=seq, frame=frame, cls_kw=p[1],
                             h=float(p[2]), w=float(p[3]), l=float(p[4]),
                             loc_x=float(p[5]), loc_y=float(p[6]), loc_z=float(p[7]), rot_y=float(p[8])))
    return out


def kitti_row_to_native_box(row: dict) -> dict:
    """kitti_boxes.py's documented inverse: xc=loc_z, yc=loc_x, l=col_l, w=col_w, theta=rot_y."""
    return dict(cls=row["cls_kw"], xc=row["loc_z"], yc=row["loc_x"],
                l=row["l"], w=row["w"], theta=row["rot_y"], score=row.get("score"))


def draw_gt_boxes(canvas, boxes: list[dict]) -> None:
    for box in boxes:
        corners = _corners_bev(box)
        cols, rows, _ = _normalize_to_pixel(corners, LIDAR_LIMS, IMG_W, IMG_H, 60)
        pts = np.column_stack([cols, rows]).astype(np.float32)
        _dashed_polylines(canvas, pts, _GT_COLOR, thickness=1)


def draw_pred_boxes(canvas, boxes: list[dict]) -> None:
    for box in boxes:
        corners = _corners_bev(box)
        cols, rows, _ = _normalize_to_pixel(corners, LIDAR_LIMS, IMG_W, IMG_H, 60)
        pts = np.column_stack([cols, rows]).astype(np.int32)
        color = _CLASS_COLOR.get(box["cls"], (0, 255, 0))
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
        label = box["cls"]
        if box.get("score") is not None:
            label += f" {box['score']:.2f}"
        lx, ly = int(pts[:, 0].min()), int(pts[:, 1].min()) - 4
        cv2.putText(canvas, label, (lx, max(ly, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, cv2.LINE_AA)


def highlight_box(canvas, box: dict) -> None:
    corners = _corners_bev(box)
    cols, rows, _ = _normalize_to_pixel(corners, LIDAR_LIMS, IMG_W, IMG_H, 60)
    cx, cy = int(cols.mean()), int(rows.mean())
    cv2.circle(canvas, (cx, cy), 22, _HIGHLIGHT_COLOR, 2)
    cv2.circle(canvas, (cx, cy), 26, (255, 255, 255), 1)


def render_frame(dataset, idx_datum: int, gt_rows: list[dict], pred_rows: list[dict],
                  highlight: dict, save_path: str, title: str) -> None:
    dict_item = dataset.list_dict_item[idx_datum]
    dict_item = dataset.get_ldr64(dict_item)
    pts = dict_item["ldr64"]
    pts_xy = pts[:, :2] if pts is not None and len(pts) else None

    render_bev(pts_xy, None, LIDAR_LIMS, title, save_path, img_w=IMG_W, img_h=IMG_H)

    canvas = cv2.imread(save_path)
    gt_boxes = [dict(cls=CLS_KW_TO_NAME.get(g["cls_kw"], g["cls_kw"]), xc=g["loc_z"], yc=g["loc_x"],
                      l=g["l"], w=g["w"], theta=g["rot_y"]) for g in gt_rows]
    draw_gt_boxes(canvas, gt_boxes)
    pred_boxes = [kitti_row_to_native_box(p) for p in pred_rows]
    draw_pred_boxes(canvas, pred_boxes)
    highlight_box(canvas, highlight)
    cv2.imwrite(save_path, canvas)


HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Object-ID Reconstruction Review — seq{seq}</title>
<style>
body {{ font-family: -apple-system,BlinkMacSystemFont,sans-serif; max-width:1300px; margin:0 auto;
       padding:20px; background:#111; color:#ddd; }}
h1 {{ color:#e94560; }}
h2 {{ color:#e94560; margin:30px 0 10px; border-bottom:2px solid #333; padding-bottom:5px; }}
h3 {{ color:#aaa; margin:15px 0 5px; font-size:14px; }}
.frame-box {{ background:#1a1a2e; border-radius:8px; padding:16px; margin:16px 0; display:inline-block;
             vertical-align:top; width:420px; margin-right:16px; }}
.frame-box img {{ width:100%; border:1px solid #333; border-radius:4px; }}
.desc {{ color:#bbb; font-size:12px; margin:6px 0 2px; font-family:monospace; }}
.badge {{ display:inline-block; font-size:10px; padding:2px 7px; border-radius:3px; margin-left:6px; }}
.badge-fp {{ background:#e94560; color:#fff; }}
.badge-fn {{ background:#f2a900; color:#111; }}
.badge-unmatched {{ background:#7b5cff; color:#fff; }}
.notice {{ background:#332; border-left:4px solid #e94560; padding:10px 14px; margin:10px 0 20px; font-size:13px; }}
.legend span {{ display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:4px; vertical-align:middle; }}
code {{ background:#222; padding:1px 5px; border-radius:3px; }}
</style>
</head>
<body>
<h1>Object-ID Reconstruction Review — sequence {seq}</h1>
<div class="notice">
Read-only LiDAR BEV crops (ROI x 0&ndash;80m, y &plusmn;25m) rendered from real point cloud data via
<code>scripts/alvis/reconstruct_object_ids.py</code>'s dataset instantiation.
<b>Legend:</b>
<span style="background:#e6e600"></span> dashed = ground truth &middot;
<span style="background:#3cdc3c"></span> solid green = Sedan pred &middot;
<span style="background:#3ca0ff"></span> solid blue = Bus/Truck pred &middot;
<span style="background:#ff0000; border-radius:50%"></span> red ring = the flagged object this card is about.
</div>
"""


def build_html(items: list[dict], seq: str, out_path: str) -> None:
    html = [HTML_HEAD.format(seq=seq)]
    for section_title, badge_cls, badge_text in [
        ("False Positives (random sample)", "badge-fp", "FP"),
        ("False Negatives (all)", "badge-fn", "FN"),
        ("Unmatched GT (Stage A, all)", "badge-unmatched", "UNMATCHED"),
    ]:
        section_items = [it for it in items if it["kind"] == badge_text]
        if not section_items:
            continue
        html.append(f'<h2>{section_title} <span class="badge {badge_cls}">{len(section_items)}</span></h2>\n')
        for it in section_items:
            html.append(f'<div class="frame-box">\n')
            html.append(f'<h3>{it["title"]}<span class="badge {badge_cls}">{badge_text}</span></h3>\n')
            html.append(f'<img src="img/{it["img"]}" alt="{it["title"]}">\n')
            html.append(f'<div class="desc">{it["desc"]}</div>\n')
            html.append('</div>\n')
    html.append("</body></html>\n")
    with open(out_path, "w") as f:
        f.write("".join(html))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default="outputs/object_id_recon")
    ap.add_argument("--configs-dir", default="configs")
    ap.add_argument("--seq", required=True)
    ap.add_argument("--n-fp", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or osp.join(args.outputs_dir, "review_gallery")
    img_dir = osp.join(out_dir, "img")
    os.makedirs(img_dir, exist_ok=True)

    seq = args.seq
    gts = load_gts(osp.join(args.outputs_dir, f"seq{seq}_gts_with_trkid.txt"))
    preds = load_preds(osp.join(args.outputs_dir, f"seq{seq}_preds_with_trkid.txt"))
    fn_entries = load_fn(osp.join(args.outputs_dir, f"seq{seq}_fn.txt"))
    unmatched_gt = load_unmatched_gt(osp.join(args.outputs_dir, f"seq{seq}_gts_unmatched.txt"))
    print(f"Loaded: {sum(len(v) for v in gts.values())} GT, {sum(len(v) for v in preds.values())} preds, "
          f"{len(fn_entries)} FN, {len(unmatched_gt)} unmatched GT", flush=True)

    cfg_path = osp.join(args.configs_dir, f"ASF_v2_0_seq{seq}_alvis.yml")
    local_cfg = cfg_from_yaml_file(cfg_path, cfg)
    dataset = KRadarFusion_v1_0(local_cfg, split="test")

    random.seed(args.seed)
    fp_rows = []
    for (s, frame, condition), rows in preds.items():
        for r in rows:
            if r["is_fp"]:
                fp_rows.append((s, frame, condition, r))
    sampled_fp = random.sample(fp_rows, min(args.n_fp, len(fp_rows)))
    print(f"Sampled {len(sampled_fp)} FP out of {len(fp_rows)} total", flush=True)

    items = []

    for s, frame, condition, r in sampled_fp:
        idx_datum = int(frame)
        gt_rows = gts.get((s, frame), [])
        cond_pred_rows = preds.get((s, frame, condition), [])
        highlight = kitti_row_to_native_box(r)
        img_name = f"fp_{s}_{frame}_{condition}.png"
        render_frame(dataset, idx_datum, gt_rows, cond_pred_rows, highlight,
                     osp.join(img_dir, img_name), f"FP seq{s} frame{frame} {condition}")
        items.append(dict(kind="FP", title=f"seq{s}_{frame} ({condition})", img=img_name,
                           desc=f"cls={r['cls_kw']} score={r['score']:.3f} "
                                f"loc=({r['loc_x']:.2f},{r['loc_y']:.2f},{r['loc_z']:.2f}) rot_y={r['rot_y']:.2f}"))

    for s, frame, condition, gt_line_idx in fn_entries:
        idx_datum = int(frame)
        gt_rows = gts.get((s, frame), [])
        cond_pred_rows = preds.get((s, frame, condition), [])
        if gt_line_idx >= len(gt_rows):
            print(f"WARNING: FN gt_line_idx {gt_line_idx} out of range for {s}_{frame} "
                  f"({len(gt_rows)} GT rows) -- skipping render", flush=True)
            continue
        g = gt_rows[gt_line_idx]
        highlight = dict(cls=CLS_KW_TO_NAME.get(g["cls_kw"], g["cls_kw"]), xc=g["loc_z"], yc=g["loc_x"],
                          l=g["l"], w=g["w"], theta=g["rot_y"])
        img_name = f"fn_{s}_{frame}_{condition}_{gt_line_idx}.png"
        render_frame(dataset, idx_datum, gt_rows, cond_pred_rows, highlight,
                     osp.join(img_dir, img_name), f"FN seq{s} frame{frame} {condition}")
        items.append(dict(kind="FN", title=f"seq{s}_{frame} ({condition})", img=img_name,
                           desc=f"missed GT: cls={g['cls_kw']} trk_id={g['trk_id']} "
                                f"loc=({g['loc_x']:.2f},{g['loc_y']:.2f},{g['loc_z']:.2f}) rot_y={g['rot_y']:.2f}"))

    for g in unmatched_gt:
        s, frame = g["seq"], g["frame"]
        idx_datum = int(frame)
        gt_rows = gts.get((s, frame), [])
        highlight = dict(cls=CLS_KW_TO_NAME.get(g["cls_kw"], g["cls_kw"]), xc=g["loc_z"], yc=g["loc_x"],
                          l=g["l"], w=g["w"], theta=g["rot_y"])
        img_name = f"unmatched_{s}_{frame}.png"
        render_frame(dataset, idx_datum, gt_rows, [], highlight,
                     osp.join(img_dir, img_name), f"Unmatched GT seq{s} frame{frame}")
        items.append(dict(kind="UNMATCHED", title=f"seq{s}_{frame}", img=img_name,
                           desc=f"cls={g['cls_kw']} (no dataset trk_id match within tolerance) "
                                f"loc=({g['loc_x']:.2f},{g['loc_y']:.2f},{g['loc_z']:.2f}) rot_y={g['rot_y']:.2f}"))

    html_path = osp.join(out_dir, "index.html")
    build_html(items, seq, html_path)
    print(f"Wrote {len(items)} cards -> {html_path}", flush=True)


if __name__ == "__main__":
    main()
