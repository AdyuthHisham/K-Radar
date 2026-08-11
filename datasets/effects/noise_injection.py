"""
Inference-time noise-injection module for K-Radar robustness testing.

Operates on post-__getitem__ dict_item as a standalone post-processing step.
No dependency on cfg_effect, build_dataset(), or any training-path code.

Data-structure conventions (from K-Radar Fusion dataset):
    rdr_sparse    : (N, 4+)  numpy — [x, y, z, power (, doppler)]
    rdr_polar_3d  : (2, R, A, E)  numpy — [0]=power plane, [1]=doppler plane
    ldr64         : (N, M)   numpy — [x, y, z, intensity, ring, ...] (M >= 3)
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


# dict_item['meta']['idx'] is keyed by the dataset's own sensor names, which do
# not match the injector's modality names: KRadarFusion_v1_0 writes
# {'rdr', 'ldr64', 'ldr128', 'camf', 'camr', 'tstamp'}. Only 'rdr' happens to
# coincide, so before this mapping existed deterministic frame_deletion silently
# never fired for lidar or camera (frame_index came back None, and
# _frame_deletion_check treats None as "don't delete"). Candidates are tried in
# order and the first present key wins.
# Keyed by BOTH the short sensor names the effect functions pass ('rdr'/'ldr'/
# 'cam') and the long modality names used in configs and registries
# ('radar'/'lidar'/'camera'), so either spelling resolves.
_MODALITY_IDX_KEYS: dict[str, tuple[str, ...]] = {
    "rdr":    ("rdr", "radar"),
    "radar":  ("rdr", "radar"),
    "ldr":    ("ldr64", "ldr", "ldr128", "lidar"),
    "lidar":  ("ldr64", "ldr", "ldr128", "lidar"),
    "cam":    ("camf", "cam", "camr", "camera"),
    "camera": ("camf", "cam", "camr", "camera"),
}


def _get_modality_idx(d: dict, modality: str) -> int | None:
    """Extract the frame index for a modality from dict_item['meta']['idx'].

    Returns an int, or None when no index is available for this modality (in
    which case deterministic frame_deletion declines to delete). Values in
    meta['idx'] are zero-padded strings such as '00150', so they are coerced.
    """
    idx_dict = d.get("meta", {}).get("idx", {})
    for key in _MODALITY_IDX_KEYS.get(modality, (modality,)):
        if key in idx_dict:
            value = idx_dict[key]
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


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


# Keys cleared by frame_deletion, per modality. Shared with NoiseInjector's cached
# should_skip()-consumption path (_apply_cached_frame_deletion) so both call sites
# clear exactly the same keys.
_RADAR_FRAME_DELETION_KEYS = ["rdr_sparse", "rdr_polar_3d"]
_LIDAR_FRAME_DELETION_KEYS = ["ldr64"]


def radar_frame_deletion(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """R1 — Radar frame deletion (deterministic or random).

    When fired, sets ``rdr_sparse`` and ``rdr_polar_3d`` to None.
    Frame index read from ``dict_item['meta']['idx']['rdr']`` for deterministic mode.
    """
    frame_index = _get_modality_idx(dict_item, "rdr")
    if _frame_deletion_check(params, rng, frame_index):
        _set_none(dict_item, _RADAR_FRAME_DELETION_KEYS)
    return dict_item


def radar_noise_induced_shifts(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """R2 — Shift radar detection positions by a noise process.

    Each point in ``rdr_sparse[:, :3]`` is displaced
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

    for key in ["rdr_sparse"]:
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
    """R3 — Partial radar data loss via zero-out (extrapolated from AI-MSF-Benchmark's LiDAR mechanism).

    Zeroes out a subset of rows in ``rdr_sparse`` via exact-count permutation
    selection.  Array row count is preserved (no deletion).

    NOTE: AI-MSF-Benchmark does not define a radar operator.  This implementation
    extrapolates the same zero-out-via-permutation principle used by
    AI-MSF-Benchmark's LiDAR operator onto radar's column format.  Not a direct port.

    Parameters
    ----------
    fraction : float, optional
        Fraction of points/returns to zero-out (default 0.3).
    """
    fraction = params.get("fraction", 0.3)
    if fraction <= 0.0:
        return dict_item
    for key in ["rdr_sparse"]:
        if _has_key(dict_item, key):
            arr = dict_item[key]
            n = len(arr)
            c = arr.shape[1]
            loss_num = int(n * fraction)
            if loss_num > 0:
                index = rng.permutation(n)[:loss_num]
                arr[index] = np.zeros(c, dtype=arr.dtype)
    return dict_item


def radar_loss_complete(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """R4 — Full radar data blackout for this frame.

    Sets ``rdr_sparse`` and ``rdr_polar_3d`` to None.
    Semantically distinct from ``loss_partial(fraction=1.0)`` — this
    represents a total sensor blackout, not extreme sampling.

    Parameters: (none required)
    """
    _set_none(dict_item, ["rdr_sparse", "rdr_polar_3d"])
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
        _set_none(dict_item, _LIDAR_FRAME_DELETION_KEYS)
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
    """L3 — Partial LiDAR data loss via zero-out (ported from AI-MSF-Benchmark).

    Zeroes out a subset of points in ``ldr64`` using exact-count permutation
    selection.  Replaces the previous Bernoulli deletion with AI-MSF-Benchmark's
    zero-out approach.  Array row count is preserved.

    Source: ``corruption/operator/lidar_operator.py:29-34``

    Parameters
    ----------
    fraction : float, optional
        Fraction of points to zero-out (default 0.3).
    """
    fraction = params.get("fraction", 0.3)
    if fraction <= 0.0:
        return dict_item
    if _has_key(dict_item, "ldr64"):
        arr = dict_item["ldr64"]
        n = len(arr)
        c = arr.shape[1]
        loss_num = int(n * fraction)
        if loss_num > 0:
            index = rng.permutation(n)[:loss_num]
            arr[index] = np.zeros(c, dtype=arr.dtype)
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
    """C3 — Partial image data loss via per-pixel dropout (ported from AI-MSF-Benchmark).

    Flattens the full (H, W, 3) uint8 image, selects exact-count indices via
    permutation, and zeroes them.  Individual colour channels of the same pixel
    can be dropped independently (a confirmed property of the source mechanism
    from ``corruption/operator/image_operator.py:126-134``).

    Replaces the previous rectangular block-occlusion implementation.

    NOTE: This operates on the flattened (H*W*3,) array, NOT grouped by pixel.
    This means R/G/B channels of the same pixel are zeroed independently.

    Parameters
    ----------
    fraction : float, optional
        Fraction of individual scalar pixel values to zero-out (default 0.2).
        Uses continuous semantics (superset of AI-MSF-Benchmark's 5-level
        discrete severity) for compatibility with existing K-Radar configs.
    """
    fraction = params.get("fraction", 0.2)
    if fraction <= 0.0:
        return dict_item

    for k in _camera_keys(dict_item):
        img_np = _to_nhwc(dict_item[k])  # (H, W, 3) uint8
        shape = img_np.shape

        # Flatten to 1D, select exact count, zero out, reshape back.
        # This matches AI-MSF-Benchmark's ImageOperator.loss_partial:
        #   arr = image.copy().flatten()
        #   loss_num = int(len(arr) * params)
        #   index = np.random.permutation(len(arr))[:loss_num]
        #   arr[index] = 0
        #   image = arr.reshape(img_shape)
        flat = img_np.flatten()
        n_total = len(flat)
        loss_num = int(n_total * fraction)
        if loss_num > 0:
            index = rng.permutation(n_total)[:loss_num]
            flat[index] = 0

        dict_item[k] = _from_nhwc(flat.reshape(shape))

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

        # should_skip() frame_deletion decisions, keyed by (modality, frame_index).
        # Written by should_skip(), read-and-popped by the matching __call__() so the
        # two call sites make the same decision for the same frame instead of each
        # drawing independently from the shared per-effect sub_rng. See _apply_list()
        # and should_skip() below, and the fix report for lifecycle rationale.
        self._skip_cache: dict[tuple[str, int], bool] = {}

    def clear_skip_cache(self) -> None:
        """Reset the should_skip() decision cache.

        The cache persists for the injector's lifetime by default (matching the
        per-effect sub_rng streams, which are also never reset mid-lifetime) and
        self-cleans on the normal should_skip() -> __call__() path: each entry is
        popped the moment __call__() consumes it. It only accumulates when
        should_skip() is called but __call__() is deliberately never invoked for
        that frame (the documented "skip loading entirely" usage) — bounded by
        roughly (frames visited x deletion rate) entries, i.e. a small fraction of
        the dataset size, not unbounded. Call this explicitly if reusing one
        injector across many passes over the same frame indices and that bound
        is still undesirable (e.g. should_skip called far more often than __call__).
        """
        self._skip_cache.clear()

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

    @staticmethod
    def _apply_cached_frame_deletion(modality: str, d: dict) -> dict:
        """Apply a should_skip()-decided frame_deletion outcome without drawing rng.

        Mirrors the clearing behaviour of radar_frame_deletion / lidar_frame_deletion /
        camera_frame_deletion exactly (same keys), but never re-consults ``rng`` — the
        decision was already made by should_skip() and is being replayed here so the
        two call sites can't disagree.
        """
        if modality == "radar":
            _set_none(d, _RADAR_FRAME_DELETION_KEYS)
        elif modality == "lidar":
            _set_none(d, _LIDAR_FRAME_DELETION_KEYS)
        elif modality == "camera":
            for k in _camera_keys(d):
                d[k] = _zero_camera_tensor(d[k])
        return d

    def _apply_list(
        self,
        d: dict,
        effects: list[tuple[str, Callable, float, dict, np.random.Generator]],
        frame_index: int | None = None,
        modality: str | None = None,
    ) -> dict:
        """Apply a list of effects with per-effect probability gating.

        Frame deletion effects are checked first.  If they fire, the sensor
        key(s) are set to None and subsequent effects skip via _has_key().

        For ``frame_deletion``, if should_skip() was already called for this
        exact (modality, frame_index) — i.e. there's a matching entry in
        ``self._skip_cache`` — that cached decision is replayed instead of
        drawing fresh randomness, so should_skip() and __call__() can never
        disagree about the same frame. Callers who never call should_skip()
        never populate the cache, so this is a no-op for them and their rng
        consumption / decisions are unchanged.
        """
        for name, fn, p, params, sub_rng in effects:
            if name == "frame_deletion" and modality is not None and frame_index is not None:
                cache_key = (modality, frame_index)
                if cache_key in self._skip_cache:
                    deleted = self._skip_cache.pop(cache_key)
                    if deleted:
                        d = self._apply_cached_frame_deletion(modality, d)
                    continue
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
            frame_index: Optional global frame index. Used to (a) key the
                         should_skip() decision cache (see _apply_list) and
                         (b) populate ``params["_caller_frame_index"]``, which
                         no effect function currently reads — deterministic
                         frame_deletion is driven solely by
                         ``dict_item['meta']['idx'][modality]``, populated by
                         the caller/dataset independently of this argument.

        Order: radar -> LiDAR -> camera.  Frame deletion applied first within
        each modality so subsequent effects skip empty sensor keys.

        Returns the modified dict_item (modifies in-place).
        """
        if self._radar_effects:
            dict_item = self._apply_list(dict_item, self._radar_effects, frame_index, "radar")
        if self._lidar_effects:
            dict_item = self._apply_list(dict_item, self._lidar_effects, frame_index, "lidar")
        if self._camera_effects:
            dict_item = self._apply_list(dict_item, self._camera_effects, frame_index, "camera")

        meta = self._metadata()
        if "meta" not in dict_item:
            dict_item["meta"] = {}
        dict_item["meta"]["noise_injection"] = meta
        return dict_item

    def should_skip(self, frame_index: int, modality: str) -> bool:
        """Pre-check whether a frame should be skipped for a given modality.

        Called by the eval script BEFORE loading the frame to avoid wasted I/O.
        Only works for ``frame_deletion`` effects configured in the given modality.

        The decision made here is cached (keyed by ``(modality, frame_index)``)
        and is the ACTUAL decision — a subsequent __call__() for the same
        frame_index will consult and replay this cached decision (via
        _apply_list) instead of drawing fresh randomness, so should_skip() and
        __call__() are guaranteed consistent for the same frame. If __call__()
        is never invoked for this frame (the intended "skip loading" usage),
        the cache entry is simply never consumed; see clear_skip_cache().

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
        if not any(name == "frame_deletion" for name, *_ in effect_list):
            # No frame_deletion configured for this modality — nothing to decide
            # or cache; avoid leaving a dead cache entry that _apply_list can
            # never consume.
            return False
        result = False
        for name, fn, p, params, sub_rng in effect_list:
            if name == "frame_deletion" and sub_rng.uniform() < p:
                # Deterministic check needs frame_index
                if _frame_deletion_check(params, sub_rng, frame_index):
                    result = True
                    break
        self._skip_cache[(modality, frame_index)] = result
        return result

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
    return dict_item


def sparse_point_dropout(dict_item: dict, params: dict, rng: np.random.Generator) -> dict:
    """[LEGACY] R2 — Randomly drop rows from radar sparse point cloud."""
    rate = params.get("rate", 0.3)
    if _has_key(dict_item, "rdr_sparse"):
        n = len(dict_item["rdr_sparse"])
        keep = rng.uniform(size=n) >= rate
        dict_item["rdr_sparse"] = dict_item["rdr_sparse"][keep]
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