"""
Inference-time noise-injection module for K-Radar robustness testing.

Operates on post-__getitem__ dict_item as a standalone post-processing step.
No dependency on cfg_effect, build_dataset(), or any training-path code.

Data-structure conventions (from K-Radar Fusion dataset):
    rdr_sparse    : (N, 4+)  numpy — [x, y, z, power (, doppler)]
    rdr_polar_3d  : (2, R, A, E)  numpy — [0]=power plane, [1]=doppler plane
    ldr64         : (N, M)   numpy — [x, y, z, ...] (M >= 3)
    pc100p        : (N, 5+)  numpy — [x, y, z, power, doppler, ...]
    camera img    : (3, H, W) torch.Tensor — ToTensor() + Normalize(dset_mean, dset_std)
"""

from __future__ import annotations

import io
from typing import Any, Callable

import cv2
import numpy as np
import torch

# --- Import config: works both as package member (from .config) and standalone ---
try:
    from .config import EffectConfig  # when loaded as part of dataseta.effects package
except ImportError:
    from config import EffectConfig   # when loaded standalone (test script)

# ──────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────


def _sub_rng(master: np.random.Generator, name: str, seed_offset: int = 0) -> np.random.Generator:
    """Derive a deterministic sub-generator from the master RNG and effect name."""
    sub_seed = hash((master.bit_generator.state["state"]["state"], name, seed_offset)) & 0xFFFFFFFF
    return np.random.default_rng(sub_seed)


def _has_key(d: dict, key: str) -> bool:
    """True if the dict_item has this sensor key and it's not None."""
    return key in d and d[key] is not None


# ══════════════════════════════════════════════
# RADAR EFFECTS  (R1–R7)
# ══════════════════════════════════════════════


def power_gaussian_noise(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """R1 — Additive Gaussian noise to radar power values.

    Operates on ``rdr_sparse[:, 3]`` (power channel) and on
    ``rdr_polar_3d[0]`` (power plane) if present.

    Parameters
    ----------
    std : float
        Standard deviation of the additive Gaussian noise (in power units).
    clip_min : float, optional
        Floor value after adding noise (default: -inf).
    """
    std = params.get("std", 0.05)
    clip_min = params.get("clip_min", -np.inf)

    if _has_key(dict_item, "rdr_sparse"):
        arr = dict_item["rdr_sparse"]
        if arr.shape[1] >= 4:
            noise = rng.normal(0.0, std, size=arr[:, 3].shape)
            arr[:, 3] = np.clip(arr[:, 3] + noise, clip_min, None)
    if _has_key(dict_item, "rdr_polar_3d"):
        pw = dict_item["rdr_polar_3d"][0]
        noise = rng.normal(0.0, std, size=pw.shape)
        dict_item["rdr_polar_3d"][0] = np.clip(pw + noise, clip_min, None)
    if _has_key(dict_item, "pc100p") and dict_item["pc100p"].shape[1] >= 4:
        arr = dict_item["pc100p"]
        noise = rng.normal(0.0, std, size=arr[:, 3].shape)
        arr[:, 3] = np.clip(arr[:, 3] + noise, clip_min, None)
    return dict_item


def sparse_point_dropout(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """R2 — Randomly drop rows from radar sparse point cloud.

    Parameters
    ----------
    rate : float
        Fraction of points to drop (0 = keep all, 1 = drop all).
    """
    rate = params.get("rate", 0.3)
    if _has_key(dict_item, "rdr_sparse"):
        n = len(dict_item["rdr_sparse"])
        keep = rng.uniform(size=n) >= rate
        dict_item["rdr_sparse"] = dict_item["rdr_sparse"][keep]
    if _has_key(dict_item, "pc100p"):
        n = len(dict_item["pc100p"])
        keep = rng.uniform(size=n) >= rate
        dict_item["pc100p"] = dict_item["pc100p"][keep]
    return dict_item


def range_attenuation(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """R3 — Exponential range-based power attenuation.

    Scales power by ``exp(-alpha * max(r - r0, 0))`` where ``r`` is
    Euclidean distance from origin.

    Parameters
    ----------
    alpha : float
        Attenuation coefficient (dB/m or neper/m  — applied as
        ``exp(-alpha * dr)``).
    r0 : float, optional
        Start range in metres (default 0.0).
    """
    alpha = params.get("alpha", 0.01)
    r0 = params.get("r0", 0.0)

    if _has_key(dict_item, "rdr_sparse"):
        arr = dict_item["rdr_sparse"]
        if arr.shape[1] >= 4:
            r = np.linalg.norm(arr[:, :3], axis=1)
            atten = np.exp(-alpha * np.maximum(r - r0, 0.0))
            arr[:, 3] = arr[:, 3] * atten
    if _has_key(dict_item, "pc100p") and dict_item["pc100p"].shape[1] >= 4:
        arr = dict_item["pc100p"]
        r = np.linalg.norm(arr[:, :3], axis=1)
        atten = np.exp(-alpha * np.maximum(r - r0, 0.0))
        arr[:, 3] = arr[:, 3] * atten
    return dict_item


def doppler_corruption(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """R4 — Doppler channel corruption.

    Parameters
    ----------
    mode : str
        One of ``'gaussian'`` (add noise), ``'zero'`` (zero the channel),
        ``'mask'`` (zero a random fraction of doppler bins).
    std : float, optional
        Gaussian noise sigma (only used when ``mode='gaussian'``).
    mask_rate : float, optional
        Fraction of bins to zero (only used when ``mode='mask'``).
    """
    mode = params.get("mode", "gaussian")
    std = params.get("std", 0.1)
    mask_rate = params.get("mask_rate", 0.3)

    # rdr_polar_3d
    if _has_key(dict_item, "rdr_polar_3d"):
        dop = dict_item["rdr_polar_3d"][1]
        if mode == "gaussian":
            dop += rng.normal(0.0, std, size=dop.shape)
        elif mode == "zero":
            dop[:] = 0.0
        elif mode == "mask":
            mask = rng.uniform(size=dop.shape) < mask_rate
            dop[mask] = 0.0

    # pc100p has doppler at column 4
    if _has_key(dict_item, "pc100p") and dict_item["pc100p"].shape[1] >= 5:
        arr = dict_item["pc100p"]
        if mode == "gaussian":
            arr[:, 4] += rng.normal(0.0, std, size=arr[:, 4].shape)
        elif mode == "zero":
            arr[:, 4] = 0.0
        elif mode == "mask":
            mask = rng.uniform(size=arr[:, 4].shape) < mask_rate
            arr[mask, 4] = 0.0

    return dict_item


def ghost_points(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """R5 — Inject random false-positive radar points.

    Injects ``num`` new rows into ``rdr_sparse``. Positions are uniformly
    sampled from the sensor FOV (default cube: [-50, 50] m per axis).
    Power values are sampled from the observed power distribution of
    existing points (or from a uniform distribution if no points exist).

    Parameters
    ----------
    num : int
        Number of ghost points to inject.
    x_range : tuple[float, float], optional
        X-axis bounds (default (-50, 50)).
    y_range : tuple[float, float], optional
        Y-axis bounds (default (-50, 50)).
    z_range : tuple[float, float], optional
        Z-axis bounds (default (-3, 5)).
    power_bounds : tuple[float, float], optional
        Power range for ghost points (min, max) when no reference exists.
    """
    num = params.get("num", 5)
    x_range = params.get("x_range", (-50.0, 50.0))
    y_range = params.get("y_range", (-50.0, 50.0))
    z_range = params.get("z_range", (-3.0, 5.0))
    power_bounds = params.get("power_bounds", (0.0, 1.0))

    if _has_key(dict_item, "rdr_sparse"):
        arr = dict_item["rdr_sparse"]
        n_cols = arr.shape[1]
        ghosts = np.zeros((num, n_cols))
        ghosts[:, 0] = rng.uniform(*x_range, size=num)
        ghosts[:, 1] = rng.uniform(*y_range, size=num)
        ghosts[:, 2] = rng.uniform(*z_range, size=num)
        if n_cols >= 4:
            if len(arr) > 0:
                # Sample from observed power distribution
                ghosts[:, 3] = rng.choice(arr[:, 3], size=num)
            else:
                ghosts[:, 3] = rng.uniform(*power_bounds, size=num)
        if n_cols >= 5:
            # Copy doppler from existing if available
            if len(arr) > 0:
                ghosts[:, 4] = rng.choice(arr[:, 4], size=num)
            else:
                ghosts[:, 4] = 0.0
        dict_item["rdr_sparse"] = np.concatenate([arr, ghosts], axis=0)
    return dict_item


def azimuth_jitter(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """R6 — Angular jitter on radar point positions.

    Perturbs each point's azimuth around the sensor origin with
    ``N(0, sigma_deg)`` degrees.

    Parameters
    ----------
    sigma_deg : float
        Standard deviation of angular jitter in degrees.
    per_axis : bool, optional
        If True, apply independent jitter to azimuth and elevation
        (default False, jitter only azimuth).
    """
    sigma_deg = params.get("sigma_deg", 1.0)
    per_axis = params.get("per_axis", False)
    sigma_rad = np.deg2rad(sigma_deg)

    if _has_key(dict_item, "rdr_sparse"):
        arr = dict_item["rdr_sparse"]
        x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
        r = np.linalg.norm(arr[:, :3], axis=1) + 1e-12
        az = np.arctan2(y, x)
        el = np.arcsin(z / r)

        az += rng.normal(0.0, sigma_rad, size=az.shape)
        if per_axis:
            el += rng.normal(0.0, sigma_rad, size=el.shape)

        arr[:, 0] = r * np.cos(el) * np.cos(az)
        arr[:, 1] = r * np.cos(el) * np.sin(az)
        arr[:, 2] = r * np.sin(el)
    return dict_item


def power_saturation(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """R7 — Clamp radar power values to a ceiling.

    Parameters
    ----------
    ceiling : float
        Maximum allowed power value.
    floor : float, optional
        Minimum allowed power value (default -inf).
    """
    ceiling = params.get("ceiling", 100.0)
    floor = params.get("floor", -np.inf)

    if _has_key(dict_item, "rdr_sparse") and dict_item["rdr_sparse"].shape[1] >= 4:
        dict_item["rdr_sparse"][:, 3] = np.clip(dict_item["rdr_sparse"][:, 3], floor, ceiling)
    if _has_key(dict_item, "rdr_polar_3d"):
        dict_item["rdr_polar_3d"][0] = np.clip(dict_item["rdr_polar_3d"][0], floor, ceiling)
    if _has_key(dict_item, "pc100p") and dict_item["pc100p"].shape[1] >= 4:
        dict_item["pc100p"][:, 3] = np.clip(dict_item["pc100p"][:, 3], floor, ceiling)
    return dict_item


# ══════════════════════════════════════════════
# LIDAR EFFECTS  (L1–L5)
# ══════════════════════════════════════════════


def lidar_random_dropout(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """L1 — Uniform-random dropout of LiDAR points.

    Parameters
    ----------
    rate : float
        Fraction of points to drop (0 = keep all, 1 = drop all).
    """
    rate = params.get("rate", 0.3)
    if _has_key(dict_item, "ldr64"):
        n = len(dict_item["ldr64"])
        keep = rng.uniform(size=n) >= rate
        dict_item["ldr64"] = dict_item["ldr64"][keep]
    return dict_item


def lidar_fov_occlusion(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """L2 — FOV occlusion by azimuth sector.

    Keeps only points whose azimuth is **outside** ``[min_deg, max_deg]``
    (or **inside** if ``invert=True``).

    Parameters
    ----------
    min_deg : float
        Start of occluded sector in degrees.
    max_deg : float
        End of occluded sector in degrees.
    invert : bool, optional
        If True, keep points *inside* the sector instead (default False).
    """
    min_deg = params.get("min_deg", -30.0)
    max_deg = params.get("max_deg", 30.0)
    invert = params.get("invert", False)

    if _has_key(dict_item, "ldr64"):
        pc = dict_item["ldr64"]
        az = np.arctan2(pc[:, 1], pc[:, 0])  # radians in [-pi, pi]
        min_rad, max_rad = np.deg2rad(min_deg), np.deg2rad(max_deg)

        if invert:
            # Keep points inside the sector
            mask = (az >= min_rad) & (az <= max_rad)
        else:
            # Keep points outside the sector (original K-Radar behaviour)
            mask = (az < min_rad) | (az > max_rad)
        dict_item["ldr64"] = pc[mask]
    return dict_item


def lidar_position_noise(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """L3 — Additive Gaussian noise to LiDAR (x, y, z) positions.

    Parameters
    ----------
    sigma_xy : float
        Std dev of noise in x and y (metres).
    sigma_z : float, optional
        Std dev of noise in z (metres), defaults to sigma_xy.
    """
    sigma_xy = params.get("sigma_xy", 0.1)
    sigma_z = params.get("sigma_z", sigma_xy)

    if _has_key(dict_item, "ldr64"):
        pc = dict_item["ldr64"]
        noise_xy = rng.normal(0.0, sigma_xy, size=(len(pc), 2))
        noise_z = rng.normal(0.0, sigma_z, size=len(pc))
        pc[:, 0] += noise_xy[:, 0]
        pc[:, 1] += noise_xy[:, 1]
        pc[:, 2] += noise_z
    return dict_item


def lidar_range_dropout(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """L4 — Drop LiDAR points outside a specified range interval.

    Parameters
    ----------
    r_min : float, optional
        Minimum range in metres (default 0.0).
    r_max : float, optional
        Maximum range in metres (default 200.0).
    """
    r_min = params.get("r_min", 0.0)
    r_max = params.get("r_max", 200.0)

    if _has_key(dict_item, "ldr64"):
        pc = dict_item["ldr64"]
        r = np.linalg.norm(pc[:, :3], axis=1)
        mask = (r >= r_min) & (r <= r_max)
        dict_item["ldr64"] = pc[mask]
    return dict_item


def lidar_downsampling(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """L5 — Uniform-stride downsampling of LiDAR point cloud.

    Parameters
    ----------
    stride : int
        Keep every Nth point (stride of 4 keeps 1/4 of points).
    """
    stride = params.get("stride", 4)
    if _has_key(dict_item, "ldr64"):
        dict_item["ldr64"] = dict_item["ldr64"][::stride]
    return dict_item


# ══════════════════════════════════════════════
# CAMERA EFFECTS  (C1–C6)
# ══════════════════════════════════════════════

# Camera helper: find all camera-image keys in the dict_item.
# These are the keys from dset.list_dealt_cams (e.g. front0, front1, ...)
# stored as (3, H, W) torch.Tensor (ToTensor + Normalize).


def _camera_keys(d: dict) -> list[str]:
    """Return camera-image keys in the dict_item (batched torch.Tensor)."""
    known = {"front0", "front1", "left0", "left1", "right0", "right1", "rear0", "rear1"}
    return [k for k in known if k in d and isinstance(d[k], torch.Tensor)]


def _to_nhwc(img: torch.Tensor) -> np.ndarray:
    """Convert (3, H, W) normalized torch.Tensor → (H, W, 3) uint8 numpy."""
    # Undo ImageNet-style normalization approximately: clamp to reasonable range
    # then rescale [0,1] → uint8
    img_np = img.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    # Simple de-normalize: shift + scale to [0,1]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = img_np * std + mean
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    return img_np


def _from_nhwc(arr: np.ndarray) -> torch.Tensor:
    """Convert (H, W, 3) uint8 numpy → (3, H, W) normalized torch.Tensor."""
    img_f = arr.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_f = (img_f - mean) / std
    return torch.from_numpy(img_f.transpose(2, 0, 1))


def camera_gaussian_noise(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """C1 — Additive Gaussian noise per pixel per channel.

    Parameters
    ----------
    sigma : float
        Noise std in 0–255 pixel-value units (default 10).
    """
    sigma = params.get("sigma", 10.0)
    # Convert pixel-space sigma to normalized-tensor-space sigma
    sigma_norm = sigma / 255.0 / 0.229  # approximate — using mean std across channels
    for k in _camera_keys(dict_item):
        img = dict_item[k]
        noise = torch.randn_like(img) * sigma_norm
        dict_item[k] = img + noise
    return dict_item


def camera_motion_blur(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """C2 — Directional motion blur via OpenCV filter2D.

    Parameters
    ----------
    kernel_size : int
        Odd-sized kernel (e.g. 5, 9). Larger = more blur.
    angle_deg : float
        Blur direction in degrees (0 = horizontal right).
    intensity : float, optional
        Blur weight; 1.0 = full kernel, <1 = blend with original (default 1.0).
    """
    ksize = params.get("kernel_size", 5)
    angle_deg = params.get("angle_deg", 45.0)
    intensity = params.get("intensity", 1.0)
    ksize = ksize if ksize % 2 == 1 else ksize + 1  # enforce odd

    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    c = ksize // 2
    rad = np.deg2rad(angle_deg)
    dx, dy = int(round(c * np.cos(rad))), int(round(c * np.sin(rad)))
    if dx == 0 and dy == 0:
        dx = c  # fallback to horizontal when angle is axis-aligned
    kernel[c - dy, c + dx] = 1.0 if abs(dx) + abs(dy) > 0 else 1.0
    kernel = kernel / kernel.sum()

    for k in _camera_keys(dict_item):
        img_np = _to_nhwc(dict_item[k])
        blurred = cv2.filter2D(img_np, -1, kernel)
        if intensity < 1.0:
            blurred = cv2.addWeighted(img_np, 1.0 - intensity, blurred, intensity, 0)
        dict_item[k] = _from_nhwc(blurred)
    return dict_item


def camera_brightness(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """C3 — Brightness adjustment by multiplicative factor.

    Parameters
    ----------
    factor : float
        Multiplier applied to all pixel values (before clipping to [0,1]).
    """
    factor = params.get("factor", 1.5)
    for k in _camera_keys(dict_item):
        img = dict_item[k]
        # De-normalize to [0,1], multiply, re-normalize
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1)
        img_pixel = img * std + mean  # [0, 1] approx
        img_pixel = torch.clamp(img_pixel * factor, 0.0, 1.0)
        dict_item[k] = (img_pixel - mean) / std
    return dict_item


def camera_patch_occlusion(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """C4 — Random rectangular patch occlusion.

    Parameters
    ----------
    num_patches : int
        Number of patches to draw (default 3).
    max_size_h : float, optional
        Max patch height as fraction of image height (default 0.15).
    max_size_w : float, optional
        Max patch width as fraction of image width (default 0.15).
    fill : str, optional
        Fill type: ``'black'``, ``'white'``, ``'mean'``, or ``'random'``
        (default ``'black'``).
    """
    num_patches = params.get("num_patches", 3)
    max_size_h = params.get("max_size_h", 0.15)
    max_size_w = params.get("max_size_w", 0.15)
    fill = params.get("fill", "black")

    for k in _camera_keys(dict_item):
        img_np = _to_nhwc(dict_item[k])  # (H, W, 3) uint8
        H, W = img_np.shape[:2]
        for _ in range(num_patches):
            ph = max(1, int(rng.uniform(0.02, max_size_h) * H))
            pw = max(1, int(rng.uniform(0.02, max_size_w) * W))
            y0 = rng.integers(0, H - ph) if H > ph else 0
            x0 = rng.integers(0, W - pw) if W > pw else 0
            if fill == "black":
                img_np[y0 : y0 + ph, x0 : x0 + pw] = 0
            elif fill == "white":
                img_np[y0 : y0 + ph, x0 : x0 + pw] = 255
            elif fill == "mean":
                patch_mean = img_np[y0 : y0 + ph, x0 : x0 + pw].mean(axis=(0, 1), keepdims=True).astype(np.uint8)
                img_np[y0 : y0 + ph, x0 : x0 + pw] = patch_mean
            elif fill == "random":
                rand_color = rng.integers(0, 256, size=(3,), dtype=np.uint8)
                img_np[y0 : y0 + ph, x0 : x0 + pw] = rand_color.reshape(1, 1, 3)
        dict_item[k] = _from_nhwc(img_np)
    return dict_item


def camera_jpeg_compression(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """C5 — JPEG compression/decompression at a given quality level.

    Parameters
    ----------
    quality : int
        JPEG quality 1–100 (lower = more artifacts, default 50).
    """
    quality = params.get("quality", 50)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    for k in _camera_keys(dict_item):
        img_np = _to_nhwc(dict_item[k])
        success, enc = cv2.imencode(".jpg", img_np, encode_params)
        if success:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            dict_item[k] = _from_nhwc(dec)
    return dict_item


def camera_gamma_distortion(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """C6 — Gamma (power-law) distortion.

    Parameters
    ----------
    gamma : float
        Gamma value (typical 0.5–2.0). <1 brightens, >1 darkens.
    """
    gamma = params.get("gamma", 1.5)
    inv_gamma = 1.0 / gamma

    for k in _camera_keys(dict_item):
        img_np = _to_nhwc(dict_item[k])  # (H, W, 3) uint8
        # Apply gamma to uint8 via LUT
        table = (np.arange(256, dtype=np.float32) / 255.0) ** inv_gamma * 255.0
        table = np.clip(table, 0, 255).astype(np.uint8)
        img_np = cv2.LUT(img_np, table)
        dict_item[k] = _from_nhwc(img_np)
    return dict_item


# ══════════════════════════════════════════════
# EFFECT REGISTRIES
# ══════════════════════════════════════════════

RADAR_EFFECTS: dict[str, Callable] = {
    "power_gaussian_noise": power_gaussian_noise,
    "sparse_point_dropout": sparse_point_dropout,
    "range_attenuation": range_attenuation,
    "doppler_corruption": doppler_corruption,
    "ghost_points": ghost_points,
    "azimuth_jitter": azimuth_jitter,
    "power_saturation": power_saturation,
}

LIDAR_EFFECTS: dict[str, Callable] = {
    "random_dropout": lidar_random_dropout,
    "fov_occlusion": lidar_fov_occlusion,
    "position_noise": lidar_position_noise,
    "range_dropout": lidar_range_dropout,
    "downsampling": lidar_downsampling,
}

CAMERA_EFFECTS: dict[str, Callable] = {
    "gaussian_noise": camera_gaussian_noise,
    "motion_blur": camera_motion_blur,
    "brightness": camera_brightness,
    "patch_occlusion": camera_patch_occlusion,
    "jpeg_compression": camera_jpeg_compression,
    "gamma_distortion": camera_gamma_distortion,
}

ALL_EFFECTS: dict[str, dict[str, Callable]] = {
    "radar": RADAR_EFFECTS,
    "lidar": LIDAR_EFFECTS,
    "camera": CAMERA_EFFECTS,
}

# Default ordering within each modality (recommended application order).
# Applied in list order; reorder by passing effects in the desired order.
DEFAULT_ORDER_RADAR = [
    "azimuth_jitter",          # R6
    "power_saturation",        # R7
    "power_gaussian_noise",    # R1
    "doppler_corruption",      # R4
    "range_attenuation",       # R3
    "sparse_point_dropout",    # R2
    "ghost_points",            # R5
]

DEFAULT_ORDER_LIDAR = [
    "range_dropout",           # L4
    "fov_occlusion",           # L2
    "position_noise",          # L3
    "random_dropout",          # L1
    "downsampling",            # L5
]

DEFAULT_ORDER_CAMERA = [
    "brightness",              # C3
    "gamma_distortion",        # C6
    "gaussian_noise",          # C1
    "motion_blur",             # C2
    "patch_occlusion",         # C4
    "jpeg_compression",        # C5
]


# ══════════════════════════════════════════════
# NoiseInjector class
# ══════════════════════════════════════════════


class NoiseInjector:
    """Inference-time corruption module for K-Radar evaluation.

    Operates on post-__getitem__ dict_item.  Composable, probabilistic,
    seedable, and fully independent of the training pipeline.

    Usage::

        injector = NoiseInjector(config)
        dict_item = injector(dict_item)   # after dataset[idx_frame]
    """

    def __init__(self, config: EffectConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Resolve effect lists — use default ordering if no explicit order given
        radar_effects = config.radar or []
        lidar_effects = config.lidar or []
        camera_effects = config.camera or []

        self._radar_effects = self._resolve("radar", radar_effects)
        self._lidar_effects = self._resolve("lidar", lidar_effects)
        self._camera_effects = self._resolve("camera", camera_effects)

    def _resolve(
        self, modality: str, user_effects: list[Any]
    ) -> list[tuple[str, Callable, float, dict, np.random.Generator]]:
        """Resolve (name, fn, p, params, sub_rng) for each configured effect."""
        registry = ALL_EFFECTS[modality]
        resolved: list[tuple[str, Callable, float, dict, np.random.Generator]] = []
        for i, eff in enumerate(user_effects):
            name = eff.name if hasattr(eff, "name") else eff["name"]
            p = eff.p if hasattr(eff, "p") else eff.get("p", 1.0)
            params = eff.params if hasattr(eff, "params") else eff.get("params", {})
            fn = registry.get(name)
            if fn is None:
                raise ValueError(f"Unknown {modality} effect: {name!r}. "
                                 f"Available: {list(registry.keys())}")
            sub_seed = hash((self.config.seed, modality, name, i)) & 0xFFFFFFFF
            sub_rng = np.random.default_rng(sub_seed)
            resolved.append((name, fn, p, params, sub_rng))
        return resolved

    def _apply_list(
        self, d: dict, effects: list[tuple[str, Callable, float, dict, np.random.Generator]]
    ) -> dict:
        """Apply a list of effects with per-effect probability gating."""
        for name, fn, p, params, sub_rng in effects:
            if sub_rng.uniform() < p:
                d = fn(d, params, sub_rng)
        return d

    def __call__(self, dict_item: dict) -> dict:
        """Apply configured corruptions to one sample's dict_item.

        Order: radar → LiDAR → camera (by increasing data size).
        Modifies dict_item in-place and returns it.

        After all effects, injects metadata under
        ``dict_item['meta']['noise_injection']``.
        """
        if self._radar_effects:
            dict_item = self._apply_list(dict_item, self._radar_effects)
        if self._lidar_effects:
            dict_item = self._apply_list(dict_item, self._lidar_effects)
        if self._camera_effects:
            dict_item = self._apply_list(dict_item, self._camera_effects)

        # Record metadata
        meta = self._metadata()
        if "meta" not in dict_item:
            dict_item["meta"] = {}
        dict_item["meta"]["noise_injection"] = meta
        return dict_item

    def _metadata(self) -> dict:
        """Return a summary of the current config for logging / audit."""
        return {
            "seed": self.config.seed,
            "radar": [{"name": n, "p": p, "params": pr} for n, _, p, pr, _ in self._radar_effects],
            "lidar": [{"name": n, "p": p, "params": pr} for n, _, p, pr, _ in self._lidar_effects],
            "camera": [{"name": n, "p": p, "params": pr} for n, _, p, pr, _ in self._camera_effects],
        }