#!/usr/bin/env python3
"""
Standalone visualization: applies all 12 Rev-2 noise-injection effects
to real K-Radar frames and generates BEV/camera PNGs for manual inspection.

Usage via env with cv2+numpy:
  /path/to/venv/bin/python datasets/effects/visualize_effects.py

No matplotlib dependency — uses cv2 drawing for BEV plots.
"""

from __future__ import annotations

import os, sys, re, struct, copy, json
from pathlib import Path

import numpy as np
import cv2

# ── Import noise-injection module directly ──
_SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTDIR)
from noise_injection import (
    radar_frame_deletion, radar_noise_induced_shifts, radar_loss_partial, radar_loss_complete,
    lidar_frame_deletion, lidar_gaussian_noise, lidar_loss_partial, lidar_loss_complete,
    camera_frame_deletion, camera_gaussian_noise, camera_loss_partial, camera_loss_complete,
)
# For camera effects we need tensor conversion helpers + torch
from noise_injection import _camera_keys, _zero_camera_tensor
import torch

# Manual tensor conversion (no torchvision dependency)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def bgr_to_normalized_tensor(bgr_img: np.ndarray) -> torch.Tensor:
    """BGR uint8 (H,W,3) → normalized (3,H,W) torch Tensor."""
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return (t - IMAGENET_MEAN) / IMAGENET_STD

def tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    """Normalized (3,H,W) tensor → BGR uint8 numpy."""
    img = t.cpu().numpy().transpose(1, 2, 0)                # (H,W,3)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = img * std + mean
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# ── Paths ──
REPO_DIR = Path('/home/adhish/Productivity/AMSCUP/repos/K-Radar')
DATA_DIR = REPO_DIR / 'data'
RDR_SPARSE_DIR = REPO_DIR / 'preprocessed' / 'rdr_sparse_data' / 'rtnh_wider_1p_1'
OUTPUT_DIR = REPO_DIR / 'outputs' / 'noise_visual_inspection'
VIZ_SCRIPT_PATH = REPO_DIR / 'datasets' / 'effects' / 'visualize_effects.py'

# ── Frame selection (3 non-consecutive frames from different sequences) ──
# Each: (seq, rdr_idx, ldr64_idx, camf_idx, label_name)
FRAMES = [
    (1,  33,  1,  2,  'seq001_rdr033'),   # Seq 1  rdr=33, lidar=1,  cam=2   Sedan ~24m
    (21, 15, 11, 34, 'seq021_rdr015'),   # Seq 21 rdr=15, lidar=11, cam=34  Sedans 2-5m urban
    (48, 35, 30, 89, 'seq048_rdr035'),   # Seq 48 rdr=35, lidar=30, cam=89  Bus/Truck 14-58m
]

# ── Effect parameters (moderate, visually obvious) ──
MASTER_SEED = 42
EFFECT_PARAMS = {
    'radar': {
        'frame_deletion':       {'mode': 'deterministic', 'index_list': [0]},
        'noise_induced_shifts': {'shift_std': 2.0, 'distribution': 'gaussian'},
        'loss_partial':         {'fraction': 0.5},
        'loss_complete':        {},
    },
    'lidar': {
        'frame_deletion':       {'mode': 'deterministic', 'index_list': [0]},
        'gaussian_noise':       {'sigma_xy': 0.5, 'sigma_z': 0.2},
        'loss_partial':         {'fraction': 0.5},
        'loss_complete':        {},
    },
    'camera': {
        'frame_deletion':       {'mode': 'deterministic', 'index_list': [0]},
        'gaussian_noise':       {'sigma': 40},
        'loss_partial':         {'fraction': 0.3},
        'loss_complete':        {},
    },
}

# ──────────────────────────────────────────────
#   Data loaders
# ──────────────────────────────────────────────

def load_rdr_sparse(seq: int, rdr_idx: int) -> np.ndarray:
    """Load preprocessed radar sparse points, removing zero-padding."""
    path = RDR_SPARSE_DIR / str(seq) / f'sprdr_{rdr_idx:05d}.npy'
    arr = np.load(str(path)).astype(np.float64)
    mask = ~((arr[:, 0] == 0) & (arr[:, 1] == 0) & (arr[:, 2] == 0))
    return arr[mask]


def load_lidar_pcd(seq: int, ldr64_idx: int) -> np.ndarray:
    """Parse PCD file (ASCII or binary) and return [x, y, z, intensity]."""
    path = DATA_DIR / str(seq) / 'os2-64' / f'os2-64_{ldr64_idx:05d}.pcd'
    with open(str(path), 'rb') as f:
        header_raw = b''
        while True:
            line = f.readline()
            header_raw += line
            if line.startswith(b'DATA'):
                data_start = f.tell()
                data_mode = line.decode('ascii').strip().split()[-1]
                break
    header = header_raw.decode('ascii')
    n_points = int(re.search(r'POINTS\s+(\d+)', header).group(1))
    fields = re.search(r'FIELDS\s+(.+)', header).group(1).split()

    if data_mode == 'ascii':
        # ASCII mode: parse space-separated float lines
        with open(str(path), 'r') as f:
            all_lines = f.readlines()
        # Find the DATA ascii line in text mode
        data_line_idx = next(i for i, l in enumerate(all_lines) if l.startswith('DATA'))
        values = np.loadtxt(all_lines[data_line_idx + 1:], dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(-1, len(fields))
    else:
        # Binary mode
        sizes = list(map(int, re.search(r'SIZE\s+(.+)', header).group(1).split()))
        types = re.search(r'TYPE\s+(.+)', header).group(1).split()
        row_size = sum(sizes)
        with open(str(path), 'rb') as f:
            f.seek(data_start)
            raw = f.read(n_points * row_size)
        result = {}
        offset = 0
        for fname, sz, tp in zip(fields, sizes, types):
            fmt_char = {1: 'B', 2: 'H', 4: 'f', 8: 'd'}.get(sz, 'f')
            if tp == 'F':
                fmt_char = fmt_char.lower()
            fmt = '<' + fmt_char * n_points
            data = struct.unpack_from(fmt, raw, offset)
            result[fname] = np.array(data, dtype=np.float32)
            offset += sz * n_points
        values = np.column_stack([result[f] for f in fields])

    # Extract [x, y, z, intensity] — these are always the first 4 columns
    pc = values[:, :4].copy()
    # Filter zero-padding rows
    mask = ~((pc[:, 0] == 0) & (pc[:, 1] == 0) & (pc[:, 2] == 0))
    return pc[mask]


def load_camera_img(seq: int, camf_idx: int) -> np.ndarray:
    """Load camera front image as BGR uint8 array (cv2 native)."""
    path = DATA_DIR / str(seq) / 'cam-front' / f'cam-front_{camf_idx:05d}.png'
    return cv2.imread(str(path))


def make_dict_item(rdr_sparse, lidar, camera_bgr):
    """Build dict_item matching K-Radar conventions for the noise injector."""
    item = {'meta': {'idx': {'rdr': 0, 'ldr': 0, 'cam': 0}}}
    if rdr_sparse is not None:
        item['rdr_sparse'] = rdr_sparse
    if lidar is not None:
        item['ldr64'] = lidar
    if camera_bgr is not None:
        item['front0'] = bgr_to_normalized_tensor(camera_bgr)
    return item


def apply_effect(dict_item, modality, effect_name, params, rng):
    """Apply one effect to a deep-copied dict_item."""
    fn_map = {
        'radar': {
            'frame_deletion':       radar_frame_deletion,
            'noise_induced_shifts': radar_noise_induced_shifts,
            'loss_partial':         radar_loss_partial,
            'loss_complete':        radar_loss_complete,
        },
        'lidar': {
            'frame_deletion':       lidar_frame_deletion,
            'gaussian_noise':       lidar_gaussian_noise,
            'loss_partial':         lidar_loss_partial,
            'loss_complete':        lidar_loss_complete,
        },
        'camera': {
            'frame_deletion':       camera_frame_deletion,
            'gaussian_noise':       camera_gaussian_noise,
            'loss_partial':         camera_loss_partial,
            'loss_complete':        camera_loss_complete,
        },
    }
    d = copy.deepcopy(dict_item)
    fn = fn_map[modality][effect_name]
    return fn(d, params, rng)


# ──────────────────────────────────────────────
#   BEV rendering with cv2 (no matplotlib)
# ──────────────────────────────────────────────

COLORMAP_SIZE = 256


def _normalize_to_pixel(xy, ax_limits, img_w, img_h, margin):
    """Map world coords (x,y) to pixel (col,row) with margin and equal aspect."""
    x_min, x_max, y_min, y_max = ax_limits
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)
    # Keep aspect ratio equal
    scale = min((img_w - 2*margin) / x_range, (img_h - 2*margin) / y_range)
    # Center in canvas
    cx = (img_w - 2*margin) / 2.0
    cy = (img_h - 2*margin) / 2.0
    x_mid = (x_min + x_max) / 2.0
    y_mid = (y_min + y_max) / 2.0
    col = margin + cx + (xy[:, 0] - x_mid) * scale
    row = margin + cy - (xy[:, 1] - y_mid) * scale  # flip-y for image coords
    return col.astype(np.int32), row.astype(np.int32), scale


def _colormap_jet(value, vmin, vmax):
    """Manual jet-like colormap: value in [vmin,vmax] → BGR uint8."""
    if vmax <= vmin:
        return np.array([128, 128, 128], dtype=np.uint8)
    t = np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0)
    # Jet-like: blue → cyan → green → yellow → red
    r = np.clip(1.5 - abs(4*t - 3), 0.0, 1.0)
    g = np.clip(1.5 - abs(4*t - 2), 0.0, 1.0)
    b = np.clip(1.5 - abs(4*t - 1), 0.0, 1.0)
    return np.array([b, g, r], dtype=np.float32)  # BGR for cv2

def _colorbar(img, vmin, vmax, margin, img_w, img_h):
    """Draw a vertical color bar on the right edge."""
    bar_w = 20
    bar_x = img_w - margin - bar_w
    bar_y = margin
    bar_h = img_h - 2 * margin
    for i in range(bar_h):
        frac = 1.0 - i / bar_h  # top=high, bottom=low
        color = tuple((_colormap_jet(frac, 0, 1) * 255).astype(np.uint8).tolist())
        cv2.line(img, (bar_x, bar_y + i), (bar_x + bar_w, bar_y + i), color, 1)
    # Labels
    cv2.putText(img, f'{vmax:.0f}', (bar_x + bar_w + 4, bar_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(img, f'{vmin:.0f}', (bar_x + bar_w + 4, bar_y + bar_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)


def render_bev(pts_xy, values, ax_limits, title_text, save_path, img_w=800, img_h=600):
    """Render a BEV scatter plot on a cv2 canvas and save as PNG.

    pts_xy: (N,2) array of x,y world coords
    values: (N,) array of scalar values to color by (or None for uniform)
    ax_limits: [xmin, xmax, ymin, ymax]
    """
    margin = 60
    canvas = np.ones((img_h, img_w, 3), dtype=np.uint8) * 30  # dark bg

    if pts_xy is None or len(pts_xy) == 0:
        cv2.putText(canvas, 'NO DATA', (img_w//2-60, img_h//2),
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
            # Uniform color
            for c, r in zip(cols, rows):
                if 0 <= r < img_h and 0 <= c < img_w:
                    cv2.circle(canvas, (int(c), int(r)), 2, (0, 180, 255), -1)

        # Grid
        for tick_pos in np.linspace(ax_limits[0], ax_limits[1], 7):
            cx_ax, _, _ = _normalize_to_pixel(np.array([[tick_pos, 0]]), ax_limits, img_w, img_h, margin)
            xp = cx_ax[0]
            if 0 <= xp < img_w:
                cv2.line(canvas, (int(xp), margin), (int(xp), img_h-margin), (60, 60, 60), 1)
                cv2.putText(canvas, f'{tick_pos:.0f}', (int(xp)-10, img_h-margin+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        for tick_pos in np.linspace(ax_limits[2], ax_limits[3], 7):
            _, cy_ax, _ = _normalize_to_pixel(np.array([[0, tick_pos]]), ax_limits, img_w, img_h, margin)
            yp = cy_ax[0]
            if 0 <= yp < img_h:
                cv2.line(canvas, (margin, int(yp)), (img_w-margin-40, int(yp)), (60, 60, 60), 1)
                cv2.putText(canvas, f'{tick_pos:.0f}', (margin-40, int(yp)+3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

        # Axis labels
        cv2.putText(canvas, 'X (m)', (img_w//2-15, img_h-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, 'Y (m)', (5, img_h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Point count
        n_str = f'{len(pts_xy)} pts'
        cv2.putText(canvas, n_str, (img_w - margin - 80, margin + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Title (multi-line)
    lines = title_text.split('\n') if '\n' in title_text else [title_text]
    for i, line in enumerate(lines):
        y_pos = 20 + i * 18
        cv2.putText(canvas, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (220, 220, 220), 1)

    cv2.imwrite(str(save_path), canvas)
    # Return axis limits for consistency across variants
    return ax_limits


# ──────────────────────────────────────────────
#   Main
# ──────────────────────────────────────────────

def main():
    print(f'[START] Visual inspection generator')
    print(f'[CONFIG] Seed={MASTER_SEED}, Frames={len(FRAMES)}')

    # Create output directories
    for label in [f[4] for f in FRAMES]:
        for mod in ['radar', 'lidar', 'camera']:
            (OUTPUT_DIR / f'frame_{label}' / mod).mkdir(parents=True, exist_ok=True)

    # Write effect params config
    config_path = OUTPUT_DIR / 'effect_parameters.yaml'
    with open(str(config_path), 'w') as f:
        f.write(f'# Effect parameters — Noise Visual Inspection\n')
        f.write(f'seed: {MASTER_SEED}\n\n')
        for mod, effects in EFFECT_PARAMS.items():
            f.write(f'{mod}:\n')
            for eff_name, params in effects.items():
                f.write(f'  {eff_name}:\n')
                for k, v in params.items():
                    f.write(f'    {k}: {v}\n')
                f.write('\n')

    for seq, rdr_idx, ldr_idx, camf_idx, label in FRAMES:
        print(f'\n{"="*60}')
        print(f'[LOAD] {label}  seq={seq}  rdr={rdr_idx}  lidar={ldr_idx}  cam={camf_idx}')

        # ── Load raw data ──
        rdr = load_rdr_sparse(seq, rdr_idx)
        lidar = load_lidar_pcd(seq, ldr_idx)
        cam = load_camera_img(seq, camf_idx)
        print(f'  radar: {len(rdr)} pts  lidar: {len(lidar)} pts  cam: {cam.shape}')

        # ── Axis limits (computed from clean data, shared across effects) ──
        def _lims(arr, pad=3):
            return [float(arr[:,0].min())-pad, float(arr[:,0].max())+pad,
                    float(arr[:,1].min())-pad, float(arr[:,1].max())+pad]
        rdr_lims = _lims(rdr, pad=3)
        lidar_lims = _lims(lidar, pad=3)

        # ── Build clean dict_item ──
        clean_item = make_dict_item(rdr, lidar, cam)

        for modality in ['radar', 'lidar', 'camera']:
            mod_dir = OUTPUT_DIR / f'frame_{label}' / modality

            if modality == 'radar':
                effects = ['frame_deletion', 'noise_induced_shifts', 'loss_partial', 'loss_complete']
                ax_limits = rdr_lims
            elif modality == 'lidar':
                effects = ['frame_deletion', 'gaussian_noise', 'loss_partial', 'loss_complete']
                ax_limits = lidar_lims
            else:
                effects = ['frame_deletion', 'gaussian_noise', 'loss_partial', 'loss_complete']
                ax_limits = None

            # ── Clean baseline ──
            clean_data = clean_item.get('rdr_sparse' if modality == 'radar' else
                                        'ldr64' if modality == 'lidar' else None)
            if modality == 'camera':
                # Save camera clean as separate file
                cv2.imwrite(str(mod_dir / f'{modality}_clean.png'), cam)
                # Also save camera with NO effect for index
            else:
                pts = clean_data[:, :3] if clean_data is not None and len(clean_data) > 0 else None
                vals = clean_data[:, 3] if (clean_data is not None and len(clean_data) > 0 and clean_data.shape[1] >= 4) else None
                n_pts = len(clean_data) if clean_data is not None else 0
                render_bev(pts[:, :2] if pts is not None else None,
                          vals, ax_limits,
                          f'CLEAN | {label} | {n_pts} pts',
                          mod_dir / f'{modality}_clean.png',
                          img_w=900, img_h=700)
            print(f'  [OK] {modality}/clean')

            # ── Apply and save each effect ──
            for eff_name in effects:
                params = EFFECT_PARAMS[modality][eff_name]
                rng = np.random.default_rng(MASTER_SEED + hash((modality, eff_name)) % (2**16))

                corrupted = apply_effect(clean_item, modality, eff_name, params, rng)

                # ── Extract data for plotting ──
                if modality == 'radar':
                    data = corrupted.get('rdr_sparse', None)
                    pts_data = data[:, :3] if data is not None and len(data) > 0 else None
                    vals = data[:, 3] if (data is not None and len(data) > 0 and data.shape[1] >= 4) else None
                    n_pts = len(data) if data is not None else 0
                    param_str = ', '.join(f'{k}={v}' for k, v in params.items() if k not in ('mode', 'index_list'))
                    title = f'{eff_name} ({param_str}) | {label}' if param_str else f'{eff_name} | {label}'
                    title += f'\n{n_pts} pts'
                    render_bev(pts_data[:, :2] if pts_data is not None else None,
                              vals, ax_limits, title,
                              mod_dir / f'{modality}_{eff_name}.png',
                              img_w=900, img_h=700)

                elif modality == 'lidar':
                    data = corrupted.get('ldr64', None)
                    pts_data = data[:, :3] if data is not None and len(data) > 0 else None
                    n_pts = len(data) if data is not None else 0
                    param_str = ', '.join(f'{k}={v}' for k, v in params.items() if k not in ('mode', 'index_list'))
                    title = f'{eff_name} ({param_str}) | {label}' if param_str else f'{eff_name} | {label}'
                    title += f'\n{n_pts} pts'
                    render_bev(pts_data[:, :2] if pts_data is not None else None,
                              None, ax_limits, title,
                              mod_dir / f'{modality}_{eff_name}.png',
                              img_w=900, img_h=700)

                else:  # camera
                    tensor = corrupted.get('front0', None)
                    if tensor is not None:
                        bgr = tensor_to_bgr(tensor)
                    else:
                        bgr = np.zeros((256, 704, 3), dtype=np.uint8)
                    param_str = ', '.join(f'{k}={v}' for k, v in params.items() if k not in ('mode', 'index_list'))
                    title_text = f'{eff_name} ({param_str}) | {label}' if param_str else f'{eff_name} | {label}'
                    # Draw title on image
                    cv2.putText(bgr, title_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
                    cv2.imwrite(str(mod_dir / f'{modality}_{eff_name}.png'), bgr)

                print(f'  [OK] {modality}/{eff_name}')

    # ──────────────────────────────────────────────
    #   Generate HTML index
    # ──────────────────────────────────────────────
    index_path = OUTPUT_DIR / 'index.html'
    with open(str(index_path), 'w') as f:
        f.write('''<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Noise Injection — Visual Inspection</title>
<style>
body { font-family: -apple-system,BlinkMacSystemFont,sans-serif; max-width:1600px; margin:0 auto;
       padding:20px; background:#111; color:#ddd; }
h1 { color:#e94560; }
h2 { color:#e94560; margin:30px 0 10px; border-bottom:2px solid #333; padding-bottom:5px; }
h3 { color:#aaa; margin:15px 0 5px; font-size:14px; text-transform:uppercase; letter-spacing:1px; }
.frame-box { background:#1a1a2e; border-radius:8px; padding:20px; margin:20px 0; }
.grid { display:grid; grid-template-columns:repeat(5,1fr); gap:8px; margin:10px 0; }
.grid img { width:100%; border:1px solid #333; border-radius:4px; }
.caption { text-align:center; font-size:11px; color:#888; margin-bottom:4px; }
</style>
</head>
<body>
<h1>Noise Injection — Visual Inspection</h1>
<p>12-effect taxonomy (Rev 2) applied to 3 real K-Radar frames.
Seed: ''' + str(MASTER_SEED) + '''</p>
''')

        effects_map = {
            'radar':  ['clean', 'frame_deletion', 'noise_induced_shifts', 'loss_partial', 'loss_complete'],
            'lidar':  ['clean', 'frame_deletion', 'gaussian_noise', 'loss_partial', 'loss_complete'],
            'camera': ['clean', 'frame_deletion', 'gaussian_noise', 'loss_partial', 'loss_complete'],
        }

        for seq, rdr_idx, ldr_idx, camf_idx, label in FRAMES:
            f.write(f'<div class="frame-box">\n')
            f.write(f'<h2>Frame: {label}</h2>\n')
            f.write(f'<p style="color:#888">seq={seq}, rdr={rdr_idx}, lidar={ldr_idx}, cam={camf_idx}</p>\n')
            for modality in ['radar', 'lidar', 'camera']:
                f.write(f'<h3>{modality.upper()}</h3>\n<div class="grid">\n')
                for eff in effects_map[modality]:
                    img_name = f'{modality}_{eff}.png'
                    rel = f'frame_{label}/{modality}/{img_name}'
                    lbl = eff.replace('_',' ').title()
                    f.write(f'<div><div class="caption">{lbl}</div>'
                            f'<a href="{rel}"><img src="{rel}" alt="{lbl}"></a></div>\n')
                f.write('</div>\n')
            f.write('</div>\n')

        f.write(f'<hr><p>Params: <a href="effect_parameters.yaml" style="color:#e94560">effect_parameters.yaml</a></p>\n')
        f.write(f'<p>Script: {VIZ_SCRIPT_PATH}</p>\n')
        f.write('</body>\n</html>\n')

    print(f'\n[DONE] Index: {index_path}')
    print(f'[DONE] Params: {config_path}')

    # ──────────────────────────────────────────────
    #   Write vAIlt report
    # ──────────────────────────────────────────────
    report_dir = REPO_DIR.parent.parent / 'docs' / 'vAIlt'
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / 'noise-visual-inspection-report.md'

    with open(str(report_path), 'w') as f:
        f.write(f'''# Noise Visual Inspection Report

## Frames Selected (non-consecutive)

| # | Label | Seq | RDR idx | LiDAR idx | Cam idx | Objects | Scene |
|---|-------|-----|---------|-----------|---------|---------|-------|
| 1 | {FRAMES[0][4]} | {FRAMES[0][0]} | {FRAMES[0][1]} | {FRAMES[0][2]} | {FRAMES[0][3]} | Sedan ~24m | Highway |
| 2 | {FRAMES[1][4]} | {FRAMES[1][0]} | {FRAMES[1][1]} | {FRAMES[1][2]} | {FRAMES[1][3]} | Sedans ~2-5m | Urban |
| 3 | {FRAMES[2][4]} | {FRAMES[2][0]} | {FRAMES[2][1]} | {FRAMES[2][2]} | {FRAMES[2][3]} | Bus/Truck 14-58m | Mixed |

## Effect Parameters

See `effect_parameters.yaml` for full details. Key choices:

- **Radar noise_induced_shifts**: shift_std=2.0m (Gaussian) — shifts each detection by ~2m RMS
- **Radar loss_partial**: fraction=0.5 — drops 50% of points
- **LiDAR gaussian_noise**: sigma_xy=0.5m — adds 0.5m std jitter to xy positions
- **LiDAR loss_partial**: fraction=0.5 — drops 50% of points
- **Camera gaussian_noise**: sigma=40 (pixel-value units, ~15% of 255 range)
- **Camera loss_partial**: fraction=0.3 — blacks out ~30% of image area

## Output

- Root: `{OUTPUT_DIR}`
- Index: `{OUTPUT_DIR}/index.html`
- Params: `{OUTPUT_DIR}/effect_parameters.yaml`
- Script: `{VIZ_SCRIPT_PATH}`
- Report: `{report_path}`
''')
    print(f'[DONE] Report: {report_path}')
    print(f'\n{"="*60}')
    print('ALL DONE. Open index.html in a browser to inspect.')


if __name__ == '__main__':
    main()