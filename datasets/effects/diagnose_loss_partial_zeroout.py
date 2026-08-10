#!/usr/bin/env python3
"""
Diagnostic: does the current zero-out `loss_partial` mechanism (radar/lidar/
camera) produce a visually legible corruption in a standard BEV scatter /
rendered image, or does it collapse invisibly onto the origin / degrade into
salt-and-pepper noise that reads as "clean-ish" at default plot settings?

This is a companion diagnostic to the earlier `visualize_effects.py` /
`noise-visual-inspection-report.md` pass, which was written against the OLD
Bernoulli-deletion `loss_partial` mechanism (sparse row deletion — points
vanish, row count shrinks). That mechanism has since been replaced (see
`docs/vAIlt/loss-partial-aimsf-replacement-report.md`) with a permutation-based
zero-out: row count is preserved, and a fraction of rows/pixels are set to
(0,0,0)-ish vectors instead of being deleted. This is a fundamentally
different visual failure mode — collapsed points stack at/near the world
origin (radar/lidar) or scatter as per-pixel salt-and-pepper noise (camera) —
and needed re-diagnosing from scratch, not a re-use of the old report's
numbers.

Does NOT modify noise_injection.py. All improvements here are visualization-
side only (dedicated coloring, zoomed origin inset, count annotations).

Run with the same venv used by the existing smoke tests in this directory:
    .smoke_venv/bin/python datasets/effects/diagnose_loss_partial_zeroout.py
"""

from __future__ import annotations

import os
import sys
import re
import struct
import copy
from pathlib import Path

import numpy as np
import cv2

_SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTDIR)

from noise_injection import (
    radar_loss_partial,
    lidar_loss_partial,
    camera_loss_partial,
    camera_loss_complete,
    _camera_keys,
)
import torch

# ── Paths (mirrors visualize_effects.py) ──
REPO_DIR = Path("/home/adhish/Productivity/AMSCUP/repos/K-Radar")
DATA_DIR = REPO_DIR / "data"
RDR_SPARSE_DIR = REPO_DIR / "preprocessed" / "rdr_sparse_data" / "rtnh_wider_1p_1"
OUTPUT_DIR = REPO_DIR / "outputs" / "noise_visual_inspection" / "loss_partial_zeroout_diagnostic"

# Same 3 non-consecutive frames as the original noise-visual-inspection-report.md,
# for direct comparability.
FRAMES = [
    (1, 33, 1, 2, "seq001_rdr033"),
    (21, 15, 11, 34, "seq021_rdr015"),
    (48, 35, 30, 89, "seq048_rdr035"),
]

MASTER_SEED = 42
FRACTION = 0.3  # matches configs/noise_injection_smoke_test.yml default

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ──────────────────────────────────────────────
#   Data loaders (copied from visualize_effects.py — read-only reuse)
# ──────────────────────────────────────────────

def load_rdr_sparse(seq: int, rdr_idx: int) -> np.ndarray:
    path = RDR_SPARSE_DIR / str(seq) / f"sprdr_{rdr_idx:05d}.npy"
    arr = np.load(str(path)).astype(np.float64)
    mask = ~((arr[:, 0] == 0) & (arr[:, 1] == 0) & (arr[:, 2] == 0))
    return arr[mask]


def load_lidar_pcd(seq: int, ldr64_idx: int) -> np.ndarray:
    path = DATA_DIR / str(seq) / "os2-64" / f"os2-64_{ldr64_idx:05d}.pcd"
    with open(str(path), "rb") as f:
        header_raw = b""
        while True:
            line = f.readline()
            header_raw += line
            if line.startswith(b"DATA"):
                data_start = f.tell()
                data_mode = line.decode("ascii").strip().split()[-1]
                break
    header = header_raw.decode("ascii")
    n_points = int(re.search(r"POINTS\s+(\d+)", header).group(1))
    fields = re.search(r"FIELDS\s+(.+)", header).group(1).split()

    if data_mode == "ascii":
        with open(str(path), "r") as f:
            all_lines = f.readlines()
        data_line_idx = next(i for i, l in enumerate(all_lines) if l.startswith("DATA"))
        values = np.loadtxt(all_lines[data_line_idx + 1:], dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(-1, len(fields))
    else:
        sizes = list(map(int, re.search(r"SIZE\s+(.+)", header).group(1).split()))
        types = re.search(r"TYPE\s+(.+)", header).group(1).split()
        row_size = sum(sizes)
        with open(str(path), "rb") as f:
            f.seek(data_start)
            raw = f.read(n_points * row_size)
        result = {}
        offset = 0
        for fname, sz, tp in zip(fields, sizes, types):
            fmt_char = {1: "B", 2: "H", 4: "f", 8: "d"}.get(sz, "f")
            if tp == "F":
                fmt_char = fmt_char.lower()
            fmt = "<" + fmt_char * n_points
            data = struct.unpack_from(fmt, raw, offset)
            result[fname] = np.array(data, dtype=np.float32)
            offset += sz * n_points
        values = np.column_stack([result[f] for f in fields])

    pc = values[:, :4].copy()
    mask = ~((pc[:, 0] == 0) & (pc[:, 1] == 0) & (pc[:, 2] == 0))
    return pc[mask]


def load_camera_img(seq: int, camf_idx: int) -> np.ndarray:
    path = DATA_DIR / str(seq) / "cam-front" / f"cam-front_{camf_idx:05d}.png"
    return cv2.imread(str(path))


def bgr_to_normalized_tensor(bgr_img: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return (t - IMAGENET_MEAN) / IMAGENET_STD


def tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    img = t.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = img * std + mean
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ──────────────────────────────────────────────
#   BEV rendering with zero-out-aware highlighting
# ──────────────────────────────────────────────

def _normalize_to_pixel(xy, ax_limits, img_w, img_h, margin):
    x_min, x_max, y_min, y_max = ax_limits
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)
    scale = min((img_w - 2 * margin) / x_range, (img_h - 2 * margin) / y_range)
    cx = (img_w - 2 * margin) / 2.0
    cy = (img_h - 2 * margin) / 2.0
    x_mid = (x_min + x_max) / 2.0
    y_mid = (y_min + y_max) / 2.0
    col = margin + cx + (xy[:, 0] - x_mid) * scale
    row = margin + cy - (xy[:, 1] - y_mid) * scale
    return col.astype(np.int32), row.astype(np.int32), scale


def render_bev_zeroout(
    pts_xy: np.ndarray,
    is_zeroed: np.ndarray,
    ax_limits,
    title_text: str,
    save_path: Path,
    img_w: int = 1000,
    img_h: int = 760,
    origin_zoom_radius: float = 3.0,
):
    """BEV scatter that distinguishes surviving points from zeroed (near-origin)
    points with a distinct color/marker, annotates the zeroed count, and draws
    a zoomed inset of the origin region so the collapsed cluster is legible
    even when it overlaps to a single pixel at full-scene scale.

    pts_xy    : (N,2) world x,y for ALL rows (including zeroed rows, which sit
                at/near (0,0) since noise_injection.py zeroes the full row).
    is_zeroed : (N,) bool mask, True for rows this call zeroed out.
    """
    margin = 60
    canvas = np.ones((img_h, img_w, 3), dtype=np.uint8) * 30

    n_total = len(pts_xy) if pts_xy is not None else 0
    n_zeroed = int(is_zeroed.sum()) if is_zeroed is not None and n_total else 0

    if pts_xy is None or n_total == 0:
        cv2.putText(canvas, "NO DATA", (img_w // 2 - 60, img_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
        cv2.imwrite(str(save_path), canvas)
        return

    cols, rows, scale = _normalize_to_pixel(pts_xy, ax_limits, img_w, img_h, margin)

    # Surviving points: neutral orange, small.
    survive_mask = ~is_zeroed
    for c, r in zip(cols[survive_mask], rows[survive_mask]):
        if 0 <= r < img_h and 0 <= c < img_w:
            cv2.circle(canvas, (int(c), int(r)), 2, (0, 180, 255), -1)

    # Zeroed points: distinct magenta, drawn last (on top) with slightly larger
    # radius + a thin outline ring so the collapsed stack is visible as ONE
    # marker even though hundreds of rows land on the identical pixel.
    zc, zr = cols[is_zeroed], rows[is_zeroed]
    for c, r in zip(zc, zr):
        if 0 <= r < img_h and 0 <= c < img_w:
            cv2.circle(canvas, (int(c), int(r)), 5, (255, 0, 255), -1)
            cv2.circle(canvas, (int(c), int(r)), 7, (255, 255, 255), 1)

    # Grid + axis ticks
    for tick_pos in np.linspace(ax_limits[0], ax_limits[1], 7):
        cx_ax, _, _ = _normalize_to_pixel(np.array([[tick_pos, 0]]), ax_limits, img_w, img_h, margin)
        xp = cx_ax[0]
        if 0 <= xp < img_w:
            cv2.line(canvas, (int(xp), margin), (int(xp), img_h - margin), (60, 60, 60), 1)
            cv2.putText(canvas, f"{tick_pos:.0f}", (int(xp) - 10, img_h - margin + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
    for tick_pos in np.linspace(ax_limits[2], ax_limits[3], 7):
        _, cy_ax, _ = _normalize_to_pixel(np.array([[0, tick_pos]]), ax_limits, img_w, img_h, margin)
        yp = cy_ax[0]
        if 0 <= yp < img_h:
            cv2.line(canvas, (margin, int(yp)), (img_w - margin - 220, int(yp)), (60, 60, 60), 1)
            cv2.putText(canvas, f"{tick_pos:.0f}", (margin - 40, int(yp) + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

    cv2.putText(canvas, "X (m)", (img_w // 2 - 15, img_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(canvas, "Y (m)", (5, img_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ── Zoomed inset of the origin region (bottom-right corner) ──
    inset_w, inset_h = 220, 220
    inset_x0 = img_w - margin - inset_w
    inset_y0 = img_h - margin - inset_h
    cv2.rectangle(canvas, (inset_x0, inset_y0), (inset_x0 + inset_w, inset_y0 + inset_h), (90, 90, 90), 1)
    cv2.putText(canvas, f"origin +/-{origin_zoom_radius:.0f}m", (inset_x0 + 4, inset_y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    inset_lims = (-origin_zoom_radius, origin_zoom_radius, -origin_zoom_radius, origin_zoom_radius)
    inset_margin = 10
    icols, irows, _ = _normalize_to_pixel(pts_xy, inset_lims, inset_w, inset_h, inset_margin)
    # Origin crosshair
    cv2.line(canvas, (inset_x0 + inset_w // 2, inset_y0), (inset_x0 + inset_w // 2, inset_y0 + inset_h), (60, 60, 60), 1)
    cv2.line(canvas, (inset_x0, inset_y0 + inset_h // 2), (inset_x0 + inset_w, inset_y0 + inset_h // 2), (60, 60, 60), 1)
    for c, r in zip(icols[survive_mask], irows[survive_mask]):
        if 0 <= r < inset_h and 0 <= c < inset_w:
            cv2.circle(canvas, (inset_x0 + int(c), inset_y0 + int(r)), 2, (0, 180, 255), -1)
    for c, r in zip(icols[is_zeroed], irows[is_zeroed]):
        if 0 <= r < inset_h and 0 <= c < inset_w:
            cv2.circle(canvas, (inset_x0 + int(c), inset_y0 + int(r)), 4, (255, 0, 255), -1)
            cv2.circle(canvas, (inset_x0 + int(c), inset_y0 + int(r)), 6, (255, 255, 255), 1)

    # ── Legend / counts panel (top-right) ──
    legend_x = img_w - margin - 230
    legend_y = margin
    cv2.rectangle(canvas, (legend_x - 8, legend_y - 18), (legend_x + 230, legend_y + 70), (45, 45, 45), -1)
    cv2.circle(canvas, (legend_x, legend_y), 4, (0, 180, 255), -1)
    cv2.putText(canvas, f"surviving: {n_total - n_zeroed}", (legend_x + 12, legend_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
    cv2.circle(canvas, (legend_x, legend_y + 22), 5, (255, 0, 255), -1)
    cv2.putText(canvas, f"zeroed (near-origin): {n_zeroed}", (legend_x + 12, legend_y + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 255), 1)
    frac_actual = n_zeroed / n_total if n_total else 0.0
    cv2.putText(canvas, f"total rows: {n_total}  (zeroed frac={frac_actual:.3f})",
                (legend_x + 12, legend_y + 49),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    lines = title_text.split("\n") if "\n" in title_text else [title_text]
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (10, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (220, 220, 220), 1)

    cv2.imwrite(str(save_path), canvas)


def render_bev_plain(pts_xy, ax_limits, title_text, save_path, img_w=1000, img_h=760):
    """Baseline/'default-style' BEV render — same style as the original
    visualize_effects.py render_bev, used to demonstrate the BEFORE (undiagnosed)
    appearance for comparison against render_bev_zeroout's AFTER."""
    margin = 60
    canvas = np.ones((img_h, img_w, 3), dtype=np.uint8) * 30
    n_total = len(pts_xy) if pts_xy is not None else 0
    if pts_xy is None or n_total == 0:
        cv2.putText(canvas, "NO DATA", (img_w // 2 - 60, img_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
        cv2.imwrite(str(save_path), canvas)
        return
    cols, rows, scale = _normalize_to_pixel(pts_xy, ax_limits, img_w, img_h, margin)
    for c, r in zip(cols, rows):
        if 0 <= r < img_h and 0 <= c < img_w:
            cv2.circle(canvas, (int(c), int(r)), 2, (0, 180, 255), -1)
    for tick_pos in np.linspace(ax_limits[0], ax_limits[1], 7):
        cx_ax, _, _ = _normalize_to_pixel(np.array([[tick_pos, 0]]), ax_limits, img_w, img_h, margin)
        xp = cx_ax[0]
        if 0 <= xp < img_w:
            cv2.line(canvas, (int(xp), margin), (int(xp), img_h - margin), (60, 60, 60), 1)
            cv2.putText(canvas, f"{tick_pos:.0f}", (int(xp) - 10, img_h - margin + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
    for tick_pos in np.linspace(ax_limits[2], ax_limits[3], 7):
        _, cy_ax, _ = _normalize_to_pixel(np.array([[0, tick_pos]]), ax_limits, img_w, img_h, margin)
        yp = cy_ax[0]
        if 0 <= yp < img_h:
            cv2.line(canvas, (margin, int(yp)), (img_w - margin - 40, int(yp)), (60, 60, 60), 1)
            cv2.putText(canvas, f"{tick_pos:.0f}", (margin - 40, int(yp) + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
    cv2.putText(canvas, "X (m)", (img_w // 2 - 15, img_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(canvas, "Y (m)", (5, img_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    n_str = f"{n_total} pts"
    cv2.putText(canvas, n_str, (img_w - margin - 80, margin + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    lines = title_text.split("\n") if "\n" in title_text else [title_text]
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (10, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (220, 220, 220), 1)
    cv2.imwrite(str(save_path), canvas)


# ──────────────────────────────────────────────
#   Verification: exact-count zero-out, row count preserved
# ──────────────────────────────────────────────

def verify_zeroout(arr_before: np.ndarray, arr_after: np.ndarray, fraction: float, label: str) -> dict:
    """Empirically check row-count preservation + expected zeroed-row count.
    Returns a dict of measured stats (does not trust prior report numbers)."""
    n_before = len(arr_before)
    n_after = len(arr_after)
    is_zero_row = np.all(arr_after == 0, axis=1)
    n_zero = int(is_zero_row.sum())
    expected_zero = int(n_before * fraction)
    stats = {
        "label": label,
        "n_before": n_before,
        "n_after": n_after,
        "row_count_preserved": n_before == n_after,
        "n_zeroed_rows": n_zero,
        "expected_zeroed_rows": expected_zero,
        "zeroed_frac_actual": n_zero / n_before if n_before else 0.0,
    }
    print(f"  [VERIFY {label}] n_before={n_before} n_after={n_after} "
          f"preserved={stats['row_count_preserved']} "
          f"zeroed={n_zero} expected={expected_zero}")
    assert n_before == n_after, f"{label}: row count changed! {n_before} -> {n_after}"
    assert n_zero == expected_zero, f"{label}: zeroed count {n_zero} != expected {expected_zero}"
    return stats, is_zero_row


def main():
    print(f"[START] Loss-partial zero-out visibility diagnostic (fraction={FRACTION})")
    all_stats = []

    for seq, rdr_idx, ldr_idx, camf_idx, label in FRAMES:
        print(f"\n{'='*60}\n[LOAD] {label} seq={seq} rdr={rdr_idx} lidar={ldr_idx} cam={camf_idx}")
        frame_dir = OUTPUT_DIR / f"frame_{label}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        rdr = load_rdr_sparse(seq, rdr_idx)
        lidar = load_lidar_pcd(seq, ldr_idx)
        cam = load_camera_img(seq, camf_idx)
        print(f"  radar: {len(rdr)} pts  lidar: {len(lidar)} pts  cam: {cam.shape}")

        def _lims(arr, pad=3):
            return [float(arr[:, 0].min()) - pad, float(arr[:, 0].max()) + pad,
                    float(arr[:, 1].min()) - pad, float(arr[:, 1].max()) + pad]

        rdr_lims = _lims(rdr, pad=3)
        lidar_lims = _lims(lidar, pad=3)

        # ── RADAR ──
        rng = np.random.default_rng(MASTER_SEED + hash(("radar", "loss_partial", label)) % (2**16))
        rdr_item = {"rdr_sparse": rdr.copy()}
        rdr_after = radar_loss_partial(copy.deepcopy(rdr_item), {"fraction": FRACTION}, rng)["rdr_sparse"]
        stats, is_zero = verify_zeroout(rdr, rdr_after, FRACTION, f"{label}/radar")
        all_stats.append(stats)

        render_bev_plain(rdr[:, :2], rdr_lims, f"RADAR clean | {label}\n{len(rdr)} pts",
                          frame_dir / "radar_clean.png")
        render_bev_plain(rdr_after[:, :2], rdr_lims,
                          f"RADAR loss_partial DEFAULT STYLE (fraction={FRACTION}) | {label}\n{len(rdr_after)} pts (unchanged, zeroed points invisible here)",
                          frame_dir / "radar_loss_partial_DEFAULT_before.png")
        render_bev_zeroout(rdr_after[:, :2], is_zero, rdr_lims,
                            f"RADAR loss_partial FIXED (fraction={FRACTION}) | {label}",
                            frame_dir / "radar_loss_partial_FIXED_after.png")

        # ── LIDAR ──
        rng = np.random.default_rng(MASTER_SEED + hash(("lidar", "loss_partial", label)) % (2**16))
        ldr_item = {"ldr64": lidar.copy()}
        ldr_after = lidar_loss_partial(copy.deepcopy(ldr_item), {"fraction": FRACTION}, rng)["ldr64"]
        stats, is_zero = verify_zeroout(lidar, ldr_after, FRACTION, f"{label}/lidar")
        all_stats.append(stats)

        render_bev_plain(lidar[:, :2], lidar_lims, f"LIDAR clean | {label}\n{len(lidar)} pts",
                          frame_dir / "lidar_clean.png")
        render_bev_plain(ldr_after[:, :2], lidar_lims,
                          f"LIDAR loss_partial DEFAULT STYLE (fraction={FRACTION}) | {label}\n{len(ldr_after)} pts (unchanged, zeroed points invisible here)",
                          frame_dir / "lidar_loss_partial_DEFAULT_before.png")
        render_bev_zeroout(ldr_after[:, :2], is_zero, lidar_lims,
                            f"LIDAR loss_partial FIXED (fraction={FRACTION}) | {label}",
                            frame_dir / "lidar_loss_partial_FIXED_after.png")

        # ── CAMERA ──
        cam_tensor = bgr_to_normalized_tensor(cam)
        item_partial = {"front0": cam_tensor.clone()}
        item_complete = {"front0": cam_tensor.clone()}
        rng_p = np.random.default_rng(MASTER_SEED + hash(("camera", "loss_partial", label)) % (2**16))
        rng_c = np.random.default_rng(MASTER_SEED + hash(("camera", "loss_complete", label)) % (2**16))

        out_partial = camera_loss_partial(item_partial, {"fraction": FRACTION}, rng_p)["front0"]
        out_complete = camera_loss_complete(item_complete, {}, rng_c)["front0"]

        bgr_clean = cam
        bgr_partial = tensor_to_bgr(out_partial)
        bgr_complete = tensor_to_bgr(out_complete)

        # Empirically verify camera zero-out fraction on the uint8 round-trip
        # (rounding through normalize/denormalize means exact-zero counts on
        # the tensor are only meaningful pre-round-trip; count on img_np the
        # same way noise_injection.py does, i.e. before the _from_nhwc call).
        from noise_injection import _to_nhwc
        clean_np = _to_nhwc(cam_tensor)
        flat_clean = clean_np.flatten()
        n_total_px = len(flat_clean)
        expected_zero_px = int(n_total_px * FRACTION)
        partial_np = _to_nhwc(out_partial)
        n_zero_px = int(np.sum(partial_np.flatten() == 0))
        print(f"  [VERIFY {label}/camera] total_scalars={n_total_px} "
              f"expected_zeroed>={expected_zero_px} observed_zero_scalars={n_zero_px} "
              f"(observed includes pre-existing dark/black pixels in the clean image, so "
              f"observed >= expected_zeroed is the correct check, not equality)")
        assert n_zero_px >= expected_zero_px, (
            f"{label}/camera: fewer zero scalars ({n_zero_px}) than the "
            f"{expected_zero_px} the permutation should have forced to zero"
        )

        combo_h = max(bgr_clean.shape[0], bgr_partial.shape[0], bgr_complete.shape[0])
        combo_w = bgr_clean.shape[1] + bgr_partial.shape[1] + bgr_complete.shape[1] + 20
        combo = np.ones((combo_h + 40, combo_w, 3), dtype=np.uint8) * 20
        x_off = 0
        for img, name in [(bgr_clean, "CLEAN"), (bgr_partial, f"LOSS_PARTIAL (frac={FRACTION}, salt-and-pepper)"),
                           (bgr_complete, "LOSS_COMPLETE (full blackout)")]:
            h, w = img.shape[:2]
            combo[40:40 + h, x_off:x_off + w] = img
            cv2.putText(combo, name, (x_off + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            x_off += w + 10
        cv2.imwrite(str(frame_dir / "camera_clean_vs_loss_partial_vs_loss_complete.png"), combo)

        all_stats.append({
            "label": f"{label}/camera",
            "n_before": n_total_px,
            "n_after": n_total_px,
            "row_count_preserved": True,
            "n_zeroed_rows": n_zero_px,
            "expected_zeroed_rows": expected_zero_px,
            "zeroed_frac_actual": n_zero_px / n_total_px,
        })

        print(f"  [OK] {label} — all modalities rendered")

    # ── Write summary CSV/markdown table alongside plots ──
    table_path = OUTPUT_DIR / "before_after_counts.md"
    with open(str(table_path), "w") as f:
        f.write("# Before/After Row Counts — loss_partial zero-out (fraction={:.2f})\n\n".format(FRACTION))
        f.write("| Frame/Modality | n_before | n_after | Row count preserved | Zeroed count | Expected zeroed | Actual zeroed frac |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for s in all_stats:
            f.write(f"| {s['label']} | {s['n_before']} | {s['n_after']} | "
                     f"{s['row_count_preserved']} | {s['n_zeroed_rows']} | "
                     f"{s['expected_zeroed_rows']} | {s['zeroed_frac_actual']:.4f} |\n")
    print(f"\n[DONE] Summary table: {table_path}")
    print("[DONE] All PNGs written under:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
