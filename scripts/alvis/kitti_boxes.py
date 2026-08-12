"""Parse and BEV-render KITTI-format detection boxes for the gallery.

Column layout (utils/util_pipeline.py:dict_datum_to_kitti):
    cls trunc occ alpha bbox_l bbox_t bbox_r bbox_b h w l loc_x loc_y loc_z rot_y [score]
      0    1   2    3     4      5      6      7    8 9 10  11    12    13    14    15

Written from native (radar/lidar-frame) box params (xc, yc, zc, rz, xl, yl, zl)
as (util_pipeline.py:346-349):
    box_dim     = "zl yl xl"   -> columns 8,9,10  = h, w, l
    box_centers = "yc zc xc"   -> columns 11,12,13 = loc_x, loc_y, loc_z
    rot_y       = rz           -> column 14

So the inverse, back to native (forward=x, lateral=y) BEV coordinates:
    xc (forward) = loc_z = col 13
    yc (lateral) = loc_x = col 11
    xl (extent along forward) = l = col 10
    yl (extent along lateral) = w = col 9
    theta (yaw)  = rot_y = col 14

Corner formula matches utils/util_geometry.py:Object3D exactly (standard
z-axis rotation of an axis-aligned box), restricted to the two BEV corners.
"""
from __future__ import annotations

import math
import os

import cv2
import numpy as np


def parse_kitti_boxes(path: str) -> list[dict]:
    """Parse one KITTI-format file into native BEV box params.

    Skips the 'dummy' placeholder row written for a frame with zero
    detections/objects (see util_pipeline.py:357).
    """
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 15 or parts[0] == "dummy":
                continue
            cls = parts[0]
            l, w = float(parts[10]), float(parts[9])
            xc, yc = float(parts[13]), float(parts[11])
            theta = float(parts[14])
            score = float(parts[15]) if len(parts) > 15 else None
            boxes.append({"cls": cls, "xc": xc, "yc": yc, "l": l, "w": w,
                         "theta": theta, "score": score})
    return boxes


def _corners_bev(box: dict) -> np.ndarray:
    """4 BEV corners (x, y), matching utils/util_geometry.py:Object3D exactly."""
    l, w, theta = box["l"], box["w"], box["theta"]
    cx = np.array([l, l, -l, -l]) / 2
    cy = np.array([w, -w, -w, w]) / 2
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rx = cx * cos_t - cy * sin_t + box["xc"]
    ry = cx * sin_t + cy * cos_t + box["yc"]
    return np.column_stack([rx, ry])


# BGR. Predictions in green (opacity by score), ground truth in yellow dashed.
_CLASS_COLOR = {
    "sed": (60, 220, 60), "bus": (60, 160, 255),
}
_GT_COLOR = (0, 230, 230)


def _dashed_polylines(canvas, pts, color, thickness=1, dash_len=6):
    n = len(pts)
    for i in range(n):
        p0, p1 = pts[i], pts[(i + 1) % n]
        dist = np.linalg.norm(p1 - p0)
        steps = max(int(dist // dash_len), 1)
        for s in range(0, steps, 2):
            a = p0 + (p1 - p0) * (s / steps)
            b = p0 + (p1 - p0) * (min(s + 1, steps) / steps)
            cv2.line(canvas, tuple(a.astype(int)), tuple(b.astype(int)), color, thickness)


def draw_boxes(image_path: str, pred_boxes: list[dict], gt_boxes: list[dict],
               ax_limits: tuple, img_w: int, img_h: int, margin: int = 60,
               show_gt: bool = True) -> None:
    """Overlay prediction (solid) and ground-truth (dashed) BEV boxes in place.

    Uses the exact same _normalize_to_pixel geometry as render_bev /
    render_bev_zeroout (img size, ax_limits, margin), so boxes land on the
    points they actually correspond to.
    """
    from visualize_effects_v2 import _normalize_to_pixel

    canvas = cv2.imread(image_path)
    if canvas is None:
        return

    if show_gt:
        for box in gt_boxes:
            corners = _corners_bev(box)
            cols, rows, _ = _normalize_to_pixel(corners, ax_limits, img_w, img_h, margin)
            pts = np.column_stack([cols, rows]).astype(np.float32)
            _dashed_polylines(canvas, pts, _GT_COLOR, thickness=1)

    for box in pred_boxes:
        corners = _corners_bev(box)
        cols, rows, _ = _normalize_to_pixel(corners, ax_limits, img_w, img_h, margin)
        pts = np.column_stack([cols, rows]).astype(np.int32)
        color = _CLASS_COLOR.get(box["cls"], (0, 255, 0))
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
        label = box["cls"]
        if box["score"] is not None:
            label += f" {box['score']:.2f}"
        lx, ly = int(pts[:, 0].min()), int(pts[:, 1].min()) - 4
        cv2.putText(canvas, label, (lx, max(ly, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, cv2.LINE_AA)

    cv2.imwrite(image_path, canvas)
