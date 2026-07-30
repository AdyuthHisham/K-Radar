#!/usr/bin/env python3
"""
Regenerates the 3-frame, 12-effect visual inspection (v2).

Supersedes `visualize_effects.py`'s output for `loss_partial` (radar/lidar):
the old `render_bev` cannot show zero-out corruption (collapsed points render
as a single invisible pixel at the origin — see
docs/vAIlt/loss-partial-zeroout-visibility-diagnostic.md). This script:

  - Re-runs the other 9 effects (radar/lidar/camera x {frame_deletion,
    noise_induced_shifts|gaussian_noise, loss_complete}) with the ORIGINAL
    `render_bev` style, to confirm current code still matches the old
    visuals — not just assumed unchanged.
  - Renders radar_loss_partial / lidar_loss_partial with the corrected
    zero-out-aware `render_bev_zeroout()` (distinct marker/color for zeroed
    points, zoomed origin inset, count annotation), ported from
    diagnose_loss_partial_zeroout.py.
  - Re-runs camera_loss_partial (new flatten-permute-zero salt-and-pepper
    mechanism) with the original camera rendering style, to confirm it still
    looks correct under current code.
  - Adds a random-mode frame_deletion spot-check (seq021 frame, p=0.5) next
    to the existing deterministic-mode run, since frame_deletion's RNG-desync
    fix (docs/vAIlt/frame-deletion-rng-desync-fix-report.md) only touched the
    NoiseInjector class, not the standalone radar/lidar/camera_frame_deletion
    functions this script calls directly — verifying both modes still behave
    is a direct check of that claim, not an assumption.

Does NOT modify noise_injection.py. Consolidates output in place at
outputs/noise_visual_inspection/ (overwrites the stale loss_partial images;
all 9 other effects' images are byte-identical re-renders given identical
inputs/seed, so overwriting them is a no-op in practice — verified below).

Run: .venv/bin/python datasets/effects/visualize_effects_v2.py
"""

from __future__ import annotations

import os, sys, re, struct, copy, hashlib
from pathlib import Path

import numpy as np
import cv2

# Import noise_injection.py from the ORIGINAL (non-worktree) K-Radar checkout,
# not this script's own worktree directory: noise_injection.py has uncommitted
# local changes (the AI-MSF-Benchmark loss_partial swap) that only exist in
# the original checkout — the worktree was branched from the last commit and
# has the stale pre-swap (Bernoulli-deletion) version. Importing the wrong one
# was caught empirically: radar_loss_partial returned a SHRUNK array (Bernoulli
# keep-mask) instead of a same-length zero-out array when run from the
# worktree's copy.
_ORIGINAL_EFFECTS_DIR = "/home/adhish/Productivity/AMSCUP/repos/K-Radar/datasets/effects"
sys.path.insert(0, _ORIGINAL_EFFECTS_DIR)
from noise_injection import (
    radar_frame_deletion, radar_noise_induced_shifts, radar_loss_partial, radar_loss_complete,
    lidar_frame_deletion, lidar_gaussian_noise, lidar_loss_partial, lidar_loss_complete,
    camera_frame_deletion, camera_gaussian_noise, camera_loss_partial, camera_loss_complete,
)
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


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


# ── Paths ──
REPO_DIR = Path("/home/adhish/Productivity/AMSCUP/repos/K-Radar")
DATA_DIR = REPO_DIR / "data"
RDR_SPARSE_DIR = REPO_DIR / "preprocessed" / "rdr_sparse_data" / "rtnh_wider_1p_1"
OUTPUT_DIR = REPO_DIR / "outputs" / "noise_visual_inspection"
VIZ_SCRIPT_PATH = REPO_DIR / "datasets" / "effects" / "visualize_effects_v2.py"

FRAMES = [
    (1, 33, 1, 2, "seq001_rdr033"),
    (21, 15, 11, 34, "seq021_rdr015"),
    (48, 35, 30, 89, "seq048_rdr035"),
]

MASTER_SEED = 42
EFFECT_PARAMS = {
    "radar": {
        "frame_deletion": {"mode": "deterministic", "index_list": [0]},
        "frame_deletion_random": {"mode": "random", "p": 0.5},
        "noise_induced_shifts": {"shift_std": 2.0, "distribution": "gaussian"},
        "loss_partial": {"fraction": 0.5},
        "loss_complete": {},
    },
    "lidar": {
        "frame_deletion": {"mode": "deterministic", "index_list": [0]},
        "frame_deletion_random": {"mode": "random", "p": 0.5},
        "gaussian_noise": {"sigma_xy": 0.5, "sigma_z": 0.2},
        "loss_partial": {"fraction": 0.5},
        "loss_complete": {},
    },
    "camera": {
        "frame_deletion": {"mode": "deterministic", "index_list": [0]},
        "frame_deletion_random": {"mode": "random", "p": 0.5},
        "gaussian_noise": {"sigma": 40},
        "loss_partial": {"fraction": 0.3},
        "loss_complete": {},
    },
}

FN_MAP = {
    "radar": {
        "frame_deletion": radar_frame_deletion,
        "frame_deletion_random": radar_frame_deletion,
        "noise_induced_shifts": radar_noise_induced_shifts,
        "loss_partial": radar_loss_partial,
        "loss_complete": radar_loss_complete,
    },
    "lidar": {
        "frame_deletion": lidar_frame_deletion,
        "frame_deletion_random": lidar_frame_deletion,
        "gaussian_noise": lidar_gaussian_noise,
        "loss_partial": lidar_loss_partial,
        "loss_complete": lidar_loss_complete,
    },
    "camera": {
        "frame_deletion": camera_frame_deletion,
        "frame_deletion_random": camera_frame_deletion,
        "gaussian_noise": camera_gaussian_noise,
        "loss_partial": camera_loss_partial,
        "loss_complete": camera_loss_complete,
    },
}


# ──────────────────────────────────────────────
#   Data loaders (identical to visualize_effects.py)
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


def _seed_offset(modality: str, effect_name: str, label: str = "") -> int:
    """Deterministic (non-hash-randomized) seed offset, matching the original
    script's `hash((modality, eff_name)) % 2**16` intent but reproducible
    across processes/PYTHONHASHSEED — the original used Python's built-in
    `hash()` on a tuple of str, which is salted per-process for security and
    is NOT stable across runs, making its own output non-reproducible."""
    h = hashlib.sha256(f"{modality}:{effect_name}:{label}".encode()).digest()
    return int.from_bytes(h[:4], "little") % (2 ** 16)


def make_dict_item(rdr_sparse, lidar, camera_bgr, idx):
    item = {"meta": {"idx": idx}}
    if rdr_sparse is not None:
        item["rdr_sparse"] = rdr_sparse
    if lidar is not None:
        item["ldr64"] = lidar
    if camera_bgr is not None:
        item["front0"] = bgr_to_normalized_tensor(camera_bgr)
    return item


def apply_effect(dict_item, modality, effect_name, params, rng):
    d = copy.deepcopy(dict_item)
    fn = FN_MAP[modality][effect_name]
    return fn(d, params, rng)


# ──────────────────────────────────────────────
#   Original render_bev (unchanged style, for the 9 confirmed effects)
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


def _colormap_jet(value, vmin, vmax):
    if vmax <= vmin:
        return np.array([128, 128, 128], dtype=np.uint8)
    t = np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0)
    r = np.clip(1.5 - abs(4 * t - 3), 0.0, 1.0)
    g = np.clip(1.5 - abs(4 * t - 2), 0.0, 1.0)
    b = np.clip(1.5 - abs(4 * t - 1), 0.0, 1.0)
    return np.array([b, g, r], dtype=np.float32)


def _colorbar(img, vmin, vmax, margin, img_w, img_h):
    bar_w = 20
    bar_x = img_w - margin - bar_w
    bar_y = margin
    bar_h = img_h - 2 * margin
    for i in range(bar_h):
        frac = 1.0 - i / bar_h
        color = tuple((_colormap_jet(frac, 0, 1) * 255).astype(np.uint8).tolist())
        cv2.line(img, (bar_x, bar_y + i), (bar_x + bar_w, bar_y + i), color, 1)
    cv2.putText(img, f"{vmax:.0f}", (bar_x + bar_w + 4, bar_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(img, f"{vmin:.0f}", (bar_x + bar_w + 4, bar_y + bar_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)


def render_bev(pts_xy, values, ax_limits, title_text, save_path, img_w=900, img_h=700):
    margin = 60
    canvas = np.ones((img_h, img_w, 3), dtype=np.uint8) * 30

    if pts_xy is None or len(pts_xy) == 0:
        cv2.putText(canvas, "NO DATA", (img_w // 2 - 60, img_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
    else:
        cols, rows, scale = _normalize_to_pixel(pts_xy, ax_limits, img_w, img_h, margin)

        if values is not None and len(values) == len(pts_xy) and values.max() > values.min():
            vmin, vmax = float(values.min()), float(values.max())
            for c, r, v in zip(cols, rows, values):
                if 0 <= r < img_h and 0 <= c < img_w:
                    color = tuple((_colormap_jet(v, vmin, vmax) * 255).astype(np.uint8).tolist())
                    cv2.circle(canvas, (int(c), int(r)), 2, color, -1)
            _colorbar(canvas, vmin, vmax, margin, img_w, img_h)
        else:
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

        n_str = f"{len(pts_xy)} pts"
        cv2.putText(canvas, n_str, (img_w - margin - 80, margin + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    lines = title_text.split("\n") if "\n" in title_text else [title_text]
    for i, line in enumerate(lines):
        y_pos = 20 + i * 18
        cv2.putText(canvas, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (220, 220, 220), 1)

    cv2.imwrite(str(save_path), canvas)


# ──────────────────────────────────────────────
#   Zero-out-aware render (ported from diagnose_loss_partial_zeroout.py,
#   verbatim logic — used ONLY for radar/lidar loss_partial)
# ──────────────────────────────────────────────

def render_bev_zeroout(pts_xy, is_zeroed, ax_limits, title_text, save_path,
                        img_w=1000, img_h=760, origin_zoom_radius=3.0):
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

    survive_mask = ~is_zeroed
    for c, r in zip(cols[survive_mask], rows[survive_mask]):
        if 0 <= r < img_h and 0 <= c < img_w:
            cv2.circle(canvas, (int(c), int(r)), 2, (0, 180, 255), -1)

    zc, zr = cols[is_zeroed], rows[is_zeroed]
    for c, r in zip(zc, zr):
        if 0 <= r < img_h and 0 <= c < img_w:
            cv2.circle(canvas, (int(c), int(r)), 5, (255, 0, 255), -1)
            cv2.circle(canvas, (int(c), int(r)), 7, (255, 255, 255), 1)

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

    inset_w, inset_h = 220, 220
    inset_x0 = img_w - margin - inset_w
    inset_y0 = img_h - margin - inset_h
    cv2.rectangle(canvas, (inset_x0, inset_y0), (inset_x0 + inset_w, inset_y0 + inset_h), (90, 90, 90), 1)
    cv2.putText(canvas, f"origin +/-{origin_zoom_radius:.0f}m", (inset_x0 + 4, inset_y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    inset_lims = (-origin_zoom_radius, origin_zoom_radius, -origin_zoom_radius, origin_zoom_radius)
    inset_margin = 10
    icols, irows, _ = _normalize_to_pixel(pts_xy, inset_lims, inset_w, inset_h, inset_margin)
    cv2.line(canvas, (inset_x0 + inset_w // 2, inset_y0), (inset_x0 + inset_w // 2, inset_y0 + inset_h), (60, 60, 60), 1)
    cv2.line(canvas, (inset_x0, inset_y0 + inset_h // 2), (inset_x0 + inset_w, inset_y0 + inset_h // 2), (60, 60, 60), 1)
    for c, r in zip(icols[survive_mask], irows[survive_mask]):
        if 0 <= r < inset_h and 0 <= c < inset_w:
            cv2.circle(canvas, (inset_x0 + int(c), inset_y0 + int(r)), 2, (0, 180, 255), -1)
    for c, r in zip(icols[is_zeroed], irows[is_zeroed]):
        if 0 <= r < inset_h and 0 <= c < inset_w:
            cv2.circle(canvas, (inset_x0 + int(c), inset_y0 + int(r)), 4, (255, 0, 255), -1)
            cv2.circle(canvas, (inset_x0 + int(c), inset_y0 + int(r)), 6, (255, 255, 255), 1)

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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ──────────────────────────────────────────────
#   Main
# ──────────────────────────────────────────────

def main():
    print("[START] Visual inspection generator v2")
    print(f"[CONFIG] Seed={MASTER_SEED}, Frames={len(FRAMES)}")

    confirmation_log = []  # (modality, effect, label, old_hash, new_hash, matched)

    for label in [f[4] for f in FRAMES]:
        for mod in ["radar", "lidar", "camera"]:
            (OUTPUT_DIR / f"frame_{label}" / mod).mkdir(parents=True, exist_ok=True)

    config_path = OUTPUT_DIR / "effect_parameters.yaml"
    with open(str(config_path), "w") as f:
        f.write("# Effect parameters — Noise Visual Inspection v2\n")
        f.write(f"seed: {MASTER_SEED}\n\n")
        for mod, effects in EFFECT_PARAMS.items():
            f.write(f"{mod}:\n")
            for eff_name, params in effects.items():
                f.write(f"  {eff_name}:\n")
                for k, v in params.items():
                    f.write(f"    {k}: {v}\n")
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

        # NOTE: idx deliberately all-0 here, matching the original
        # visualize_effects.py's make_dict_item — deterministic frame_deletion
        # (index_list=[0]) is designed to always fire for this fixed-idx
        # dict_item, regardless of the frame's real rdr/ldr/cam index. Using
        # the real indices here would make frame_deletion silently stop firing
        # (real idx != 0) and break comparability with the original report.
        idx = {"rdr": 0, "ldr": 0, "cam": 0}
        clean_item = make_dict_item(rdr, lidar, cam, idx)

        for modality in ["radar", "lidar", "camera"]:
            mod_dir = OUTPUT_DIR / f"frame_{label}" / modality

            if modality == "radar":
                effects = ["frame_deletion", "frame_deletion_random", "noise_induced_shifts", "loss_partial", "loss_complete"]
                ax_limits = rdr_lims
            elif modality == "lidar":
                effects = ["frame_deletion", "frame_deletion_random", "gaussian_noise", "loss_partial", "loss_complete"]
                ax_limits = lidar_lims
            else:
                effects = ["frame_deletion", "frame_deletion_random", "gaussian_noise", "loss_partial", "loss_complete"]
                ax_limits = None

            clean_data = clean_item.get("rdr_sparse" if modality == "radar" else
                                        "ldr64" if modality == "lidar" else None)
            clean_path = mod_dir / f"{modality}_clean.png"
            old_hash = _sha256_file(clean_path) if clean_path.exists() else None
            if modality == "camera":
                cv2.imwrite(str(clean_path), cam)
            else:
                pts = clean_data[:, :3] if clean_data is not None and len(clean_data) > 0 else None
                vals = clean_data[:, 3] if (clean_data is not None and len(clean_data) > 0 and clean_data.shape[1] >= 4) else None
                n_pts = len(clean_data) if clean_data is not None else 0
                render_bev(pts[:, :2] if pts is not None else None,
                           vals, ax_limits,
                           f"CLEAN | {label} | {n_pts} pts",
                           clean_path)
            new_hash = _sha256_file(clean_path)
            confirmation_log.append((modality, "clean", label, old_hash, new_hash, old_hash == new_hash))
            print(f"  [OK] {modality}/clean")

            for eff_name in effects:
                params = EFFECT_PARAMS[modality][eff_name]
                seed_off = _seed_offset(modality, eff_name, label)
                rng = np.random.default_rng(MASTER_SEED + seed_off)
                # camera_gaussian_noise draws from torch's global RNG (torch.randn_like),
                # ignoring the numpy `rng` passed in — seed it explicitly so re-runs of
                # THIS script are reproducible. Does not touch noise_injection.py.
                torch.manual_seed(MASTER_SEED + seed_off)

                corrupted = apply_effect(clean_item, modality, eff_name, params, rng)
                out_path = mod_dir / f"{modality}_{eff_name}.png"
                old_hash = _sha256_file(out_path) if out_path.exists() else None
                is_new_style = (eff_name == "loss_partial" and modality in ("radar", "lidar"))

                if modality == "radar":
                    data = corrupted.get("rdr_sparse", None)
                    pts_data = data[:, :3] if data is not None and len(data) > 0 else None
                    n_pts = len(data) if data is not None else 0
                    param_str = ", ".join(f"{k}={v}" for k, v in params.items() if k not in ("mode", "index_list"))
                    title = f"{eff_name} ({param_str}) | {label}" if param_str else f"{eff_name} | {label}"
                    title += f"\n{n_pts} pts"

                    if is_new_style:
                        clean_arr = clean_item["rdr_sparse"]
                        is_zeroed = np.all(data == 0, axis=1) if data is not None else np.zeros(0, dtype=bool)
                        render_bev_zeroout(pts_data[:, :2] if pts_data is not None else None,
                                            is_zeroed, ax_limits,
                                            f"RADAR loss_partial FIXED ({param_str}) | {label}",
                                            out_path)
                    else:
                        vals = data[:, 3] if (data is not None and len(data) > 0 and data.shape[1] >= 4) else None
                        render_bev(pts_data[:, :2] if pts_data is not None else None,
                                   vals, ax_limits, title, out_path)

                elif modality == "lidar":
                    data = corrupted.get("ldr64", None)
                    pts_data = data[:, :3] if data is not None and len(data) > 0 else None
                    n_pts = len(data) if data is not None else 0
                    param_str = ", ".join(f"{k}={v}" for k, v in params.items() if k not in ("mode", "index_list"))
                    title = f"{eff_name} ({param_str}) | {label}" if param_str else f"{eff_name} | {label}"
                    title += f"\n{n_pts} pts"

                    if is_new_style:
                        is_zeroed = np.all(data == 0, axis=1) if data is not None else np.zeros(0, dtype=bool)
                        render_bev_zeroout(pts_data[:, :2] if pts_data is not None else None,
                                            is_zeroed, ax_limits,
                                            f"LIDAR loss_partial FIXED ({param_str}) | {label}",
                                            out_path)
                    else:
                        render_bev(pts_data[:, :2] if pts_data is not None else None,
                                   None, ax_limits, title, out_path)

                else:  # camera
                    tensor = corrupted.get("front0", None)
                    if tensor is not None:
                        bgr = tensor_to_bgr(tensor)
                    else:
                        bgr = np.zeros((256, 704, 3), dtype=np.uint8)
                    param_str = ", ".join(f"{k}={v}" for k, v in params.items() if k not in ("mode", "index_list"))
                    title_text = f"{eff_name} ({param_str}) | {label}" if param_str else f"{eff_name} | {label}"
                    cv2.putText(bgr, title_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
                    cv2.imwrite(str(out_path), bgr)

                new_hash = _sha256_file(out_path)
                confirmation_log.append((modality, eff_name, label, old_hash, new_hash,
                                          old_hash == new_hash if not is_new_style else None))
                tag = "NEW STYLE" if is_new_style else ("MATCH" if old_hash == new_hash else "CHANGED/NEW FILE")
                print(f"  [OK] {modality}/{eff_name}  [{tag}]")

    # ── Confirmation summary ──
    conf_path = OUTPUT_DIR / "v2_regeneration_confirmation.md"
    with open(str(conf_path), "w") as f:
        f.write("# v2 Regeneration — Byte-Identity Confirmation Log\n\n")
        f.write("Compares each re-rendered PNG's SHA-256 against the file that existed at that\n")
        f.write("path before this run (from the original `visualize_effects.py` pass), for every\n")
        f.write("effect NOT intentionally re-styled. `loss_partial` (radar/lidar) is intentionally\n")
        f.write("re-styled (old file replaced by the zero-out-aware render) and is excluded from\n")
        f.write("the match/mismatch check below.\n\n")
        f.write("| Modality | Effect | Frame | Old hash (prefix) | New hash (prefix) | Result |\n")
        f.write("|---|---|---|---|---|---|\n")
        for mod, eff, label, oldh, newh, matched in confirmation_log:
            oldp = (oldh or "MISSING")[:12]
            newp = newh[:12]
            if matched is None:
                result = "RE-STYLED (intentional)"
            elif oldh is None:
                result = "NEW (no prior file)"
            elif matched:
                result = "CONFIRMED IDENTICAL"
            else:
                result = "**DIFFERS — investigate**"
            f.write(f"| {mod} | {eff} | {label} | `{oldp}` | `{newp}` | {result} |\n")
    print(f"\n[DONE] Confirmation log: {conf_path}")

    n_checked = sum(1 for r in confirmation_log if r[5] is not None and r[3] is not None)
    n_matched = sum(1 for r in confirmation_log if r[5] is True)
    n_mismatched = sum(1 for r in confirmation_log if r[5] is False and r[3] is not None)
    print(f"[SUMMARY] {n_matched}/{n_checked} confirmed byte-identical to prior run; {n_mismatched} differ")

    # ──────────────────────────────────────────────
    #   HTML index
    # ──────────────────────────────────────────────
    index_path = OUTPUT_DIR / "index.html"
    with open(str(index_path), "w") as f:
        f.write('''<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Noise Injection — Visual Inspection (v2)</title>
<style>
body { font-family: -apple-system,BlinkMacSystemFont,sans-serif; max-width:1600px; margin:0 auto;
       padding:20px; background:#111; color:#ddd; }
h1 { color:#e94560; }
h2 { color:#e94560; margin:30px 0 10px; border-bottom:2px solid #333; padding-bottom:5px; }
h3 { color:#aaa; margin:15px 0 5px; font-size:14px; text-transform:uppercase; letter-spacing:1px; }
.frame-box { background:#1a1a2e; border-radius:8px; padding:20px; margin:20px 0; }
.grid { display:grid; grid-template-columns:repeat(6,1fr); gap:8px; margin:10px 0; }
.grid img { width:100%; border:1px solid #333; border-radius:4px; }
.caption { text-align:center; font-size:11px; color:#888; margin-bottom:4px; }
.badge { display:inline-block; font-size:10px; padding:1px 6px; border-radius:3px; margin-left:6px; }
.badge-fixed { background:#e94560; color:#fff; }
.badge-confirmed { background:#2a6; color:#fff; }
.notice { background:#332; border-left:4px solid #e94560; padding:10px 14px; margin:10px 0 20px; font-size:13px; }
</style>
</head>
<body>
<h1>Noise Injection — Visual Inspection (v2)</h1>
<p>12-effect taxonomy (Rev 2) applied to 3 real K-Radar frames. Seed: ''' + str(MASTER_SEED) + '''</p>
<div class="notice">
<b>v2 supersedes the original run in place.</b> radar_loss_partial and lidar_loss_partial
now use the corrected zero-out-aware rendering (magenta marker + zoomed origin inset + counts) —
the original render_bev style could not show this corruption at all (collapsed points render as
one invisible pixel at the origin). camera_loss_partial and all other 9 effects are re-confirmed
current-code re-renders of the original style; see
<a href="v2_regeneration_confirmation.md" style="color:#e94560">v2_regeneration_confirmation.md</a>
for the byte-identity confirmation log. A random-mode frame_deletion spot-check
(<code>frame_deletion_random</code>) was added alongside the original deterministic-mode run.
See <a href="../../../docs/vAIlt/noise-visual-inspection-report-v2.md" style="color:#e94560">noise-visual-inspection-report-v2.md</a>.
</div>
''')

        effects_map = {
            "radar": ["clean", "frame_deletion", "frame_deletion_random", "noise_induced_shifts", "loss_partial", "loss_complete"],
            "lidar": ["clean", "frame_deletion", "frame_deletion_random", "gaussian_noise", "loss_partial", "loss_complete"],
            "camera": ["clean", "frame_deletion", "frame_deletion_random", "gaussian_noise", "loss_partial", "loss_complete"],
        }
        fixed_effects = {("radar", "loss_partial"), ("lidar", "loss_partial")}

        for seq, rdr_idx, ldr_idx, camf_idx, label in FRAMES:
            f.write('<div class="frame-box">\n')
            f.write(f'<h2>Frame: {label}</h2>\n')
            f.write(f'<p style="color:#888">seq={seq}, rdr={rdr_idx}, lidar={ldr_idx}, cam={camf_idx}</p>\n')
            for modality in ["radar", "lidar", "camera"]:
                f.write(f'<h3>{modality.upper()}</h3>\n<div class="grid">\n')
                for eff in effects_map[modality]:
                    img_name = f"{modality}_{eff}.png"
                    rel = f"frame_{label}/{modality}/{img_name}"
                    lbl = eff.replace("_", " ").title()
                    badge = ""
                    if (modality, eff) in fixed_effects:
                        badge = '<span class="badge badge-fixed">FIXED RENDER</span>'
                    elif eff not in ("clean",):
                        badge = '<span class="badge badge-confirmed">RE-CONFIRMED</span>'
                    f.write(f'<div><div class="caption">{lbl}{badge}</div>'
                            f'<a href="{rel}"><img src="{rel}" alt="{lbl}"></a></div>\n')
                f.write("</div>\n")
            f.write("</div>\n")

        f.write(f'<hr><p>Params: <a href="effect_parameters.yaml" style="color:#e94560">effect_parameters.yaml</a></p>\n')
        f.write(f'<p>Confirmation log: <a href="v2_regeneration_confirmation.md" style="color:#e94560">v2_regeneration_confirmation.md</a></p>\n')
        f.write(f"<p>Script: {VIZ_SCRIPT_PATH}</p>\n")
        f.write("</body>\n</html>\n")

    print(f"\n[DONE] Index: {index_path}")
    print(f"[DONE] Params: {config_path}")
    print(f"\n{'='*60}")
    print("ALL DONE.")


if __name__ == "__main__":
    main()
