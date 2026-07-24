"""
Inference-time noise-injection module for K-Radar robustness testing.

Operates on post-__getitem__ dict_item as a standalone post-processing step.
No dependency on cfg_effect, build_dataset(), or any training-path code.

Data-structure conventions (from K-Radar Fusion dataset):
    rdr_sparse    : (N, 4+)  numpy — [x, y, z, power (, doppler)]
    rdr_polar_3d  : (2, R, A, E)  numpy — [0]=power plane, [1]=doppler plane
    ldr64         : (N, M)   numpy — [x, y, z, intensity, ring, ...] (M >= 3)
    pc100p        : (N, 5+)  numpy — [x, y, z, power, doppler, ...]
    camera img    : (3, H, W) torch.Tensor — ToTensor() + Normalize(dset_mean, dset_std)

TAXONOMY (Rev 2): 12 effects, 4 per modality:
  Radar:     frame_deletion, noise_induced_shifts, loss_partial, loss_complete
  LiDAR:     frame_deletion, gaussian_noise,        loss_partial, loss_complete
  Camera:    frame_deletion, gaussian_noise,        loss_partial, loss_complete

Legacy effects from Rev 1 (18-effect design) are preserved below under
"# LEGACY EFFECTS" for manual import but are NOT loaded into the active
registries or used by NoiseInjector.
"""

from __future__ import annotations

import io
from typing import Any, Callable

import cv2
import numpy as np
import torch

# --- Import config: works both as package member (from .config) and standalone ---
try:
    from .config import EffectConfig  # when loaded as part of datasets.effects package
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
    """True if the dict_item has this sensor key and it's not None (data is present)."""
    return key in d and d[key] is not None


def _set_none(d: dict, keys: list[str]) -> None:
    """Set the given keys to None (sensor blackout / frame deleted)."""
    for k in keys:
        if k in d:
            d[k] = None


def _get_modality_idx(d: dict, modality: str) -> int | None:
    """Extract the frame index for a modality from dict_item['meta']['idx']."""
    meta = d.get("meta", {})
    idx_dict = meta.get("idx", {})
    return idx_dict.get(modality, None)


# ══════════════════════════════════════════════
# SHARED: Frame deletion (deterministic + random per modality)
# ══════════════════════════════════════════════


def _frame_deletion_check(params: dict, rng: np.random.Generator,
                           frame_index: int | None = None) -> bool:
    """Return True if this frame should be deleted (decider shared by all modalities).

    Parameters (from params dict):
        mode : str — 'deterministic' or 'random' (default: 'random')
        interval : int — delete every Nth frame (mode='deterministic', no index_list)
        index_list : list[int] — explicit indices to delete (mode='deterministic')
        p : float — per-frame deletion probability (mode='random', default: 0.1)
        frame_index : int | None — caller-provided frame index for deterministic mode
    """
    mode = params.get("mode", "random")
    if mode == "deterministic":
        index_list = params.get("index_list", None)
        if index_list is not None:
            # index_list mode: delete if frame_index is in the list
            if frame_index is None:
                # Can't check list without frame_index — don't delete (graceful skip)
                return False
            return int(frame_index) in index_list
        else:
            # interval mode: delete every Nth frame
            interval = params.get("interval", 10)
            if frame_index is None:
                return False  # no frame index available — don't delete
            return int(frame_index) % interval == 0
    else:  # 'random'
        p = params.get("p", 0.1)
        return rng.uniform() < p


# ──────────────────────────────────────────────
# RADAR EFFECTS  (4 effects)
# ──────────────────────────────────────────────


def radar_frame_deletion(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """R1 — Radar frame deletion (deterministic or random).

    When fired, sets ``rdr_sparse``, ``rdr_polar_3d``, and ``pc100p`` to None.
    Frame index read from ``dict_item['meta']['idx']['rdr']`` for deterministic mode.
    """
    frame_index = _get_modality_idx(dict_item, "rdr")
    if _frame_deletion_check(params, rng, frame_index):
        _set_none(dict_item, ["rdr_sparse", "rdr_polar_3d", "pc100p"])
    return dict_item


def radar_noise_induced_shifts(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """R2 — Shift radar detection positions by a noise process.

    Each point in ``rdr_sparse[:, :3]`` and ``pc100p[:, :3]`` is displaced
    by a random vector.  This is a *coordinate shift* effect — distinct from
    additive power noise (old power_gaussian_noise) and distinct from azimuth
    jitter (old azimuth_jitter).

    Parameters
    ----------
    shift_std : float, optional
        Standard deviation of displacement in metres (default 1.0).
    distribution : str, optional
        'gaussian' (default) or 'uniform'.
    clip_radius : float, optional
        Max displacement radius in metres.  If set, displacements exceeding
        this are clipped to the radius (direction preserved).
    """
    shift_std = params.get("shift_std", 1.0)
    distribution = params.get("distribution", "gaussian")
    clip_radius = params.get("clip_radius", None)

    for key in ["rdr_sparse", "pc100p"]:
        if _has_key(dict_item, key) and dict_item[key].shape[1] >= 3:
            arr = dict_item[key]
            n = len(arr)
            # Sample displacement vectors
            if distribution == "uniform":
                # Uniform in a sphere of radius shift_std*sqrt(3) ≈ covering
                # a cube of side 2*shift_std, then optionally clip
                dx = rng.uniform(-shift_std, shift_std, size=n)
                dy = rng.uniform(-shift_std, shift_std, size=n)
                dz = rng.uniform(-shift_std, shift_std, size=n)
            else:  # 'gaussian'
                dx = rng.normal(0.0, shift_std, size=n)
                dy = rng.normal(0.0, shift_std, size=n)
                dz = rng.normal(0.0, shift_std, size=n)

            # Optional clip to max radius
            if clip_radius is not None and clip_radius > 0:
                disp = np.column_stack([dx, dy, dz])
                dist = np.linalg.norm(disp, axis=1)
                overshoot = dist > clip_radius
                if np.any(overshoot):
                    scale = np.where(overshoot, clip_radius / np.maximum(dist, 1e-12), 1.0)
                    disp = disp * scale[:, None]
                    dx, dy, dz = disp[:, 0], disp[:, 1], disp[:, 2]

            arr[:, 0] += dx
            arr[:, 1] += dy
            arr[:, 2] += dz

    return dict_item


def radar_loss_partial(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """R3 — Partial radar data loss by Bernoulli sampling on detections.

    Each point in ``rdr_sparse`` and ``pc100p`` is independently dropped
    with probability ``fraction``.

    Parameters
    ----------
    fraction : float, optional
        Fraction of points/returns to remove (default 0.3).
    """
    fraction = params.get("fraction", 0.3)
    for key in ["rdr_sparse", "pc100p"]:
        if _has_key(dict_item, key):
            arr = dict_item[key]
            n = len(arr)
            keep = rng.uniform(size=n) >= fraction
            dict_item[key] = arr[keep]
    return dict_item


def radar_loss_complete(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """R4 — Full radar data blackout for this frame.

    Sets ``rdr_sparse``, ``rdr_polar_3d``, and ``pc100p`` to None.
    Semantically distinct from ``loss_partial(fraction=1.0)`` — this
    represents a total sensor blackout, not extreme sampling.

    Parameters: (none required)
    """
    _set_none(dict_item, ["rdr_sparse", "rdr_polar_3d", "pc100p"])
    return dict_item


# ──────────────────────────────────────────────
# LIDAR EFFECTS  (4 effects)
# ──────────────────────────────────────────────


def lidar_frame_deletion(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """L1 — LiDAR frame deletion.

    When fired, sets ``ldr64`` to None.
    Frame index read from ``dict_item['meta']['idx']['ldr']``.
    """
    frame_index = _get_modality_idx(dict_item, "ldr")
    if _frame_deletion_check(params, rng, frame_index):
        _set_none(dict_item, ["ldr64"])
    return dict_item


def lidar_gaussian_noise(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """L2 — Additive Gaussian noise to LiDAR positions and intensity.

    Adds independent N(0, sigma) noise to (x, y, z) coordinates and
    optionally to the intensity channel (column 3 in ``ldr64``).

    Parameters
    ----------
    sigma_xy : float, optional
        Std dev of noise in x and y (metres, default 0.1).
    sigma_z : float, optional
        Std dev of noise in z (metres), defaults to sigma_xy.
    sigma_intensity : float, optional
        Std dev of noise on intensity channel (default 0.0 — no noise).
    """
    sigma_xy = params.get("sigma_xy", 0.1)
    sigma_z = params.get("sigma_z", sigma_xy)
    sigma_intensity = params.get("sigma_intensity", 0.0)

    if _has_key(dict_item, "ldr64"):
        pc = dict_item["ldr64"]
        n = len(pc)
        # Position noise
        noise_xy = rng.normal(0.0, sigma_xy, size=(n, 2))
        noise_z = rng.normal(0.0, sigma_z, size=n)
        pc[:, 0] += noise_xy[:, 0]
        pc[:, 1] += noise_xy[:, 1]
        pc[:, 2] += noise_z
        # Optional intensity noise (column 3)
        if sigma_intensity > 0.0 and pc.shape[1] >= 4:
            noise_intensity = rng.normal(0.0, sigma_intensity, size=n)
            pc[:, 3] += noise_intensity

    return dict_item


def lidar_loss_partial(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """L3 — Partial LiDAR data loss by Bernoulli sampling.

    Each point in ``ldr64`` independently dropped with probability ``fraction``.

    Parameters
    ----------
    fraction : float, optional
        Fraction of points to remove (default 0.3).
    """
    fraction = params.get("fraction", 0.3)
    if _has_key(dict_item, "ldr64"):
        n = len(dict_item["ldr64"])
        keep = rng.uniform(size=n) >= fraction
        dict_item["ldr64"] = dict_item["ldr64"][keep]
    return dict_item


def lidar_loss_complete(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """L4 — Full LiDAR blackout.

    Sets ``ldr64`` to None.

    Parameters: (none required)
    """
    _set_none(dict_item, ["ldr64"])
    return dict_item


# ──────────────────────────────────────────────
# CAMERA EFFECTS  (4 effects)
# ──────────────────────────────────────────────


def _camera_keys(d: dict) -> list[str]:
    """Return camera-image keys in the dict_item (batched torch.Tensor)."""
    known = {"front0", "front1", "left0", "left1", "right0", "right1", "rear0", "rear1"}
    return [k for k in known if k in d and isinstance(d[k], torch.Tensor)]


def _to_nhwc(img: torch.Tensor) -> np.ndarray:
    """Convert (3, H, W) normalized torch.Tensor → (H, W, 3) uint8 numpy."""
    img_np = img.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
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


def _zero_camera_tensor(img: torch.Tensor) -> torch.Tensor:
    """Return a zero-filled tensor in normalized space ≈ black pixel values.

    Black in uint8 (0,0,0) maps to approx (-mean/std) = (-2.12, -2.04, -1.80).
    """
    c, h, w = img.shape
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1)
    black = (torch.zeros_like(img) - mean) / std
    return black


def camera_frame_deletion(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """C1 — Camera frame deletion.

    When fired, all camera-image keys are replaced with a zero-filled
    normalized tensor (black frame). Frame index read from
    ``dict_item['meta']['idx']['cam']``.
    """
    frame_index = _get_modality_idx(dict_item, "cam")
    if _frame_deletion_check(params, rng, frame_index):
        for k in _camera_keys(dict_item):
            dict_item[k] = _zero_camera_tensor(dict_item[k])
    return dict_item


def camera_gaussian_noise(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """C2 — Additive Gaussian noise per pixel per channel.

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


def camera_loss_partial(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """C3 — Partial image data loss via single contiguous zeroed region.

    Zeros out a single rectangular region covering approximately ``fraction``
    of the image area.  Region is randomly positioned.

    Distinct from multi-patch occlusion — this is the canonical "partial loss"
    corruption for images.

    Parameters
    ----------
    fraction : float, optional
        Fraction of image area to zero out (default 0.2).
    """
    fraction = params.get("fraction", 0.2)
    if fraction <= 0.0:
        return dict_item
    if fraction >= 1.0:
        # Complete blackout via the region mechanism; clip to 99% to avoid
        # degenerate region size.
        fraction = 0.99

    for k in _camera_keys(dict_item):
        img_np = _to_nhwc(dict_item[k])  # (H, W, 3) uint8
        H, W = img_np.shape[:2]

        # Region size: target_fraction = (ph * pw) / (H * W)
        # Fix aspect ratio ~1 (square-ish), compute side length
        target_area = fraction * H * W
        side = int(np.sqrt(target_area))
        ph = min(max(1, side), H)
        pw = min(max(1, side), W)
        # Adjust to get as close as possible to target_area
        # Scale one dimension to hit area more closely
        if ph * pw > 0:
            ratio = np.clip(target_area / (ph * pw), 0.5, 2.0)
            if ratio >= 1.0:
                pw = min(W, int(pw * np.sqrt(ratio)))
            else:
                ph = min(H, int(ph / np.sqrt(ratio)))

        ph = max(1, min(ph, H))
        pw = max(1, min(pw, W))

        y0 = rng.integers(0, H - ph + 1) if H > ph else 0
        x0 = rng.integers(0, W - pw + 1) if W > pw else 0

        # Zero out the region
        img_np[y0: y0 + ph, x0: x0 + pw] = 0
        dict_item[k] = _from_nhwc(img_np)

    return dict_item


def camera_loss_complete(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """C4 — Full image blackout.

    Replaces all camera-image tensors with zero-filled normalized tensors.

    Parameters: (none required)
    """
    for k in _camera_keys(dict_item):
        dict_item[k] = _zero_camera_tensor(dict_item[k])
    return dict_item


# ══════════════════════════════════════════════
# NEW TAXONOMY REGISTRIES (Rev 2 — active)
# ══════════════════════════════════════════════

RADAR_EFFECTS: dict[str, Callable] = {
    "frame_deletion":       radar_frame_deletion,
    "noise_induced_shifts": radar_noise_induced_shifts,
    "loss_partial":         radar_loss_partial,
    "loss_complete":        radar_loss_complete,
}

LIDAR_EFFECTS: dict[str, Callable] = {
    "frame_deletion":       lidar_frame_deletion,
    "gaussian_noise":       lidar_gaussian_noise,
    "loss_partial":         lidar_loss_partial,
    "loss_complete":        lidar_loss_complete,
}

CAMERA_EFFECTS: dict[str, Callable] = {
    "frame_deletion":       camera_frame_deletion,
    "gaussian_noise":       camera_gaussian_noise,
    "loss_partial":         camera_loss_partial,
    "loss_complete":        camera_loss_complete,
}

ALL_EFFECTS: dict[str, dict[str, Callable]] = {
    "radar":  RADAR_EFFECTS,
    "lidar":  LIDAR_EFFECTS,
    "camera": CAMERA_EFFECTS,
}

# Default ordering within each modality (recommended application order).
# Frame deletion first (empties sensor data; subsequent effects skip via _has_key).
# Loss complete should go last (it unconditionally blacks out; anything after is wasted).
DEFAULT_ORDER_RADAR = [
    "frame_deletion",       # R1 — first, so other effects skip if frame gone
    "noise_induced_shifts", # R2
    "loss_partial",         # R3
    "loss_complete",        # R4 — last (unconditional blackout)
]

DEFAULT_ORDER_LIDAR = [
    "frame_deletion",       # L1
    "gaussian_noise",       # L2
    "loss_partial",         # L3
    "loss_complete",        # L4
]

DEFAULT_ORDER_CAMERA = [
    "frame_deletion",       # C1
    "gaussian_noise",       # C2
    "loss_partial",         # C3
    "loss_complete",        # C4
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
        dict_item = injector(dict_item, frame_index=idx)  # after dataset[idx]
    """

    def __init__(self, config: EffectConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Resolve effect lists
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
                raise ValueError(
                    f"Unknown {modality} effect: {name!r}. "
                    f"Available: {list(registry.keys())}"
                )
            sub_seed = hash((self.config.seed, modality, name, i)) & 0xFFFFFFFF
            sub_rng = np.random.default_rng(sub_seed)
            resolved.append((name, fn, p, params, sub_rng))
        return resolved

    def _apply_list(
        self,
        d: dict,
        effects: list[tuple[str, Callable, float, dict, np.random.Generator]],
        frame_index: int | None = None,
    ) -> dict:
        """Apply a list of effects with per-effect probability gating.

        Frame deletion effects are checked first.  If they fire, the sensor
        key(s) are set to None and subsequent effects skip via _has_key().
        """
        for name, fn, p, params, sub_rng in effects:
            if sub_rng.uniform() < p:
                # Inject frame_index into params for frame_deletion effects
                if name == "frame_deletion":
                    params = dict(params)  # shallow copy so we don't mutate the stored dict
                    params["_caller_frame_index"] = frame_index
                d = fn(d, params, sub_rng)
        return d

    def __call__(self, dict_item: dict, frame_index: int | None = None) -> dict:
        """Apply configured corruptions to one sample's dict_item.

        Args:
            dict_item: K-Radar frame dict from dataset[idx].
            frame_index: Optional global frame index.  Required by deterministic
                         frame_deletion (interval/index_list modes).  If omitted,
                         only random-mode frame deletion will work.

        Order: radar -> LiDAR -> camera.  Frame deletion applied first within
        each modality so subsequent effects skip empty sensor keys.

        Returns the modified dict_item (modifies in-place).
        """
        if self._radar_effects:
            dict_item = self._apply_list(dict_item, self._radar_effects, frame_index)
        if self._lidar_effects:
            dict_item = self._apply_list(dict_item, self._lidar_effects, frame_index)
        if self._camera_effects:
            dict_item = self._apply_list(dict_item, self._camera_effects, frame_index)

        meta = self._metadata()
        if "meta" not in dict_item:
            dict_item["meta"] = {}
        dict_item["meta"]["noise_injection"] = meta
        return dict_item

    def should_skip(self, frame_index: int, modality: str) -> bool:
        """Pre-check whether a frame should be skipped for a given modality.

        Called by the eval script BEFORE loading the frame to avoid wasted I/O.
        Only works for ``frame_deletion`` effects configured in the given modality.

        Args:
            frame_index: Global frame index.
            modality: One of ``'radar'``, ``'lidar'``, ``'camera'``.

        Returns:
            True if this frame should be skipped (data not loaded).
        """
        effect_list = {
            "radar": self._radar_effects,
            "lidar": self._lidar_effects,
            "camera": self._camera_effects,
        }.get(modality, [])
        for name, fn, p, params, sub_rng in effect_list:
            if name == "frame_deletion" and sub_rng.uniform() < p:
                # Deterministic check needs frame_index
                if _frame_deletion_check(params, sub_rng, frame_index):
                    return True
        return False

    def _metadata(self) -> dict:
        """Return a summary of the current config for logging / audit."""
        return {
            "seed": self.config.seed,
            "radar": [{"name": n, "p": p, "params": pr} for n, _, p, pr, _ in self._radar_effects],
            "lidar": [{"name": n, "p": p, "params": pr} for n, _, p, pr, _ in self._lidar_effects],
            "camera": [{"name": n, "p": p, "params": pr} for n, _, p, pr, _ in self._camera_effects],
        }


# ══════════════════════════════════════════════
# LEGACY EFFECTS (Rev 1 — preserved for manual import)
# ══════════════════════════════════════════════
#
# These are the 18 effects from the original design.  They are NOT loaded
# into ALL_EFFECTS and NOT used by NoiseInjector.  Code that directly
# imports them (e.g. individual function references in old test scripts)
# will still work, but they are not part of the active taxonomy.
#
# To use a legacy effect::
#     from noise_injection import power_gaussian_noise
#     d = power_gaussian_noise(d, params, rng)
# ══════════════════════════════════════════════

# --- RADAR LEGACY (7 effects) ---


def power_gaussian_noise(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """[LEGACY] R1 — Additive Gaussian noise to radar power values."""
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
    """[LEGACY] R2 — Randomly drop rows from radar sparse point cloud."""
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
    """[LEGACY] R3 — Exponential range-based power attenuation."""
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
    """[LEGACY] R4 — Doppler channel corruption."""
    mode = params.get("mode", "gaussian")
    std = params.get("std", 0.1)
    mask_rate = params.get("mask_rate", 0.3)
    if _has_key(dict_item, "rdr_polar_3d"):
        dop = dict_item["rdr_polar_3d"][1]
        if mode == "gaussian":
            dop += rng.normal(0.0, std, size=dop.shape)
        elif mode == "zero":
            dop[:] = 0.0
        elif mode == "mask":
            mask = rng.uniform(size=dop.shape) < mask_rate
            dop[mask] = 0.0
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
    """[LEGACY] R5 — Inject random false-positive radar points."""
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
                ghosts[:, 3] = rng.choice(arr[:, 3], size=num)
            else:
                ghosts[:, 3] = rng.uniform(*power_bounds, size=num)
        if n_cols >= 5:
            if len(arr) > 0:
                ghosts[:, 4] = rng.choice(arr[:, 4], size=num)
            else:
                ghosts[:, 4] = 0.0
        dict_item["rdr_sparse"] = np.concatenate([arr, ghosts], axis=0)
    return dict_item


def azimuth_jitter(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """[LEGACY] R6 — Angular jitter on radar point positions."""
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
    """[LEGACY] R7 — Clamp radar power values to a ceiling."""
    ceiling = params.get("ceiling", 100.0)
    floor = params.get("floor", -np.inf)
    if _has_key(dict_item, "rdr_sparse") and dict_item["rdr_sparse"].shape[1] >= 4:
        dict_item["rdr_sparse"][:, 3] = np.clip(dict_item["rdr_sparse"][:, 3], floor, ceiling)
    if _has_key(dict_item, "rdr_polar_3d"):
        dict_item["rdr_polar_3d"][0] = np.clip(dict_item["rdr_polar_3d"][0], floor, ceiling)
    if _has_key(dict_item, "pc100p") and dict_item["pc100p"].shape[1] >= 4:
        dict_item["pc100p"][:, 3] = np.clip(dict_item["pc100p"][:, 3], floor, ceiling)
    return dict_item


# --- LIDAR LEGACY (5 effects) ---


def lidar_random_dropout(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """[LEGACY] L1 — Uniform-random dropout of LiDAR points."""
    rate = params.get("rate", 0.3)
    if _has_key(dict_item, "ldr64"):
        n = len(dict_item["ldr64"])
        keep = rng.uniform(size=n) >= rate
        dict_item["ldr64"] = dict_item["ldr64"][keep]
    return dict_item


def lidar_fov_occlusion(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """[LEGACY] L2 — FOV occlusion by azimuth sector."""
    min_deg = params.get("min_deg", -30.0)
    max_deg = params.get("max_deg", 30.0)
    invert = params.get("invert", False)
    if _has_key(dict_item, "ldr64"):
        pc = dict_item["ldr64"]
        az = np.arctan2(pc[:, 1], pc[:, 0])
        min_rad, max_rad = np.deg2rad(min_deg), np.deg2rad(max_deg)
        if invert:
            mask = (az >= min_rad) & (az <= max_rad)
        else:
            mask = (az < min_rad) | (az > max_rad)
        dict_item["ldr64"] = pc[mask]
    return dict_item


def lidar_position_noise(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """[LEGACY] L3 — Additive Gaussian noise to LiDAR positions only."""
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
    """[LEGACY] L4 — Drop LiDAR points outside a specified range interval."""
    r_min = params.get("r_min", 0.0)
    r_max = params.get("r_max", 200.0)
    if _has_key(dict_item, "ldr64"):
        pc = dict_item["ldr64"]
        r = np.linalg.norm(pc[:, :3], axis=1)
        mask = (r >= r_min) & (r <= r_max)
        dict_item["ldr64"] = pc[mask]
    return dict_item


def lidar_downsampling(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """[LEGACY] L5 — Uniform-stride downsampling of LiDAR point cloud."""
    stride = params.get("stride", 4)
    if _has_key(dict_item, "ldr64"):
        dict_item["ldr64"] = dict_item["ldr64"][::stride]
    return dict_item


# --- CAMERA LEGACY (6 effects) ---


def camera_motion_blur(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """[LEGACY] C2 — Directional motion blur via OpenCV filter2D."""
    ksize = params.get("kernel_size", 5)
    angle_deg = params.get("angle_deg", 45.0)
    intensity = params.get("intensity", 1.0)
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    c = ksize // 2
    rad = np.deg2rad(angle_deg)
    dx, dy = int(round(c * np.cos(rad))), int(round(c * np.sin(rad)))
    if dx == 0 and dy == 0:
        dx = c
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
    """[LEGACY] C3 — Brightness adjustment by multiplicative factor."""
    factor = params.get("factor", 1.5)
    for k in _camera_keys(dict_item):
        img = dict_item[k]
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1)
        img_pixel = img * std + mean
        img_pixel = torch.clamp(img_pixel * factor, 0.0, 1.0)
        dict_item[k] = (img_pixel - mean) / std
    return dict_item


def camera_patch_occlusion(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """[LEGACY] C4 — Random rectangular patch occlusion."""
    num_patches = params.get("num_patches", 3)
    max_size_h = params.get("max_size_h", 0.15)
    max_size_w = params.get("max_size_w", 0.15)
    fill = params.get("fill", "black")
    for k in _camera_keys(dict_item):
        img_np = _to_nhwc(dict_item[k])
        H, W = img_np.shape[:2]
        for _ in range(num_patches):
            ph = max(1, int(rng.uniform(0.02, max_size_h) * H))
            pw = max(1, int(rng.uniform(0.02, max_size_w) * W))
            y0 = rng.integers(0, H - ph) if H > ph else 0
            x0 = rng.integers(0, W - pw) if W > pw else 0
            if fill == "black":
                img_np[y0: y0 + ph, x0: x0 + pw] = 0
            elif fill == "white":
                img_np[y0: y0 + ph, x0: x0 + pw] = 255
            elif fill == "mean":
                patch_mean = img_np[y0: y0 + ph, x0: x0 + pw].mean(axis=(0, 1), keepdims=True).astype(np.uint8)
                img_np[y0: y0 + ph, x0: x0 + pw] = patch_mean
            elif fill == "random":
                rand_color = rng.integers(0, 256, size=(3,), dtype=np.uint8)
                img_np[y0: y0 + ph, x0: x0 + pw] = rand_color.reshape(1, 1, 3)
        dict_item[k] = _from_nhwc(img_np)
    return dict_item


def camera_jpeg_compression(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """[LEGACY] C5 — JPEG compression/decompression at a given quality level."""
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
    """[LEGACY] C6 — Gamma (power-law) distortion."""
    gamma = params.get("gamma", 1.5)
    inv_gamma = 1.0 / gamma
    for k in _camera_keys(dict_item):
        img_np = _to_nhwc(dict_item[k])
        table = (np.arange(256, dtype=np.float32) / 255.0) ** inv_gamma * 255.0
        table = np.clip(table, 0, 255).astype(np.uint8)
        img_np = cv2.LUT(img_np, table)
        dict_item[k] = _from_nhwc(img_np)
    return dict_item


# --- LEGACY REGISTRIES ---

RADAR_EFFECTS_LEGACY: dict[str, Callable] = {
    "power_gaussian_noise":     power_gaussian_noise,
    "sparse_point_dropout":     sparse_point_dropout,
    "range_attenuation":        range_attenuation,
    "doppler_corruption":       doppler_corruption,
    "ghost_points":             ghost_points,
    "azimuth_jitter":           azimuth_jitter,
    "power_saturation":         power_saturation,
}

LIDAR_EFFECTS_LEGACY: dict[str, Callable] = {
    "random_dropout":   lidar_random_dropout,
    "fov_occlusion":    lidar_fov_occlusion,
    "position_noise":   lidar_position_noise,
    "range_dropout":    lidar_range_dropout,
    "downsampling":     lidar_downsampling,
}

CAMERA_EFFECTS_LEGACY: dict[str, Callable] = {
    # NOTE: camera_gaussian_noise is NOT listed here. It was carried forward
    # into the active CAMERA_EFFECTS registry without changes — the new-taxonomy
    # function IS the canonical version there is no separate legacy copy.
    # Accessible via CAMERA_EFFECTS["gaussian_noise"] or by importing
    # camera_gaussian_noise directly at module level.
    "motion_blur":      camera_motion_blur,
    "brightness":       camera_brightness,
    "patch_occlusion":  camera_patch_occlusion,
    "jpeg_compression": camera_jpeg_compression,
    "gamma_distortion": camera_gamma_distortion,
}

ALL_EFFECTS_LEGACY: dict[str, dict[str, Callable]] = {
    "radar":  RADAR_EFFECTS_LEGACY,
    "lidar":  LIDAR_EFFECTS_LEGACY,
    "camera": CAMERA_EFFECTS_LEGACY,
}