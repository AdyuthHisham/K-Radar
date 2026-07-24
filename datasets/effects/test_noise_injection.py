"""
Standalone unit tests for the noise-injection module (Rev 2 taxonomy).

Tests all 12 effects:
  Radar:     frame_deletion, noise_induced_shifts, loss_partial, loss_complete
  LiDAR:     frame_deletion, gaussian_noise,        loss_partial, loss_complete
  Camera:    frame_deletion, gaussian_noise,        loss_partial, loss_complete

Plus injector integration tests and legacy-import compatibility check.
Runs on synthetic dummy data — no K-Radar pipeline dependency.

Imports directly from the effects directory (not through the root datasets/
package) to avoid pulling in open3d-dependent K-Radar modules.
"""

import sys
import os
_TESTDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _TESTDIR)

import math
import numpy as np
import torch

import config as _config_mod
import noise_injection as _noise_mod

EffectConfig = _config_mod.EffectConfig
Effect       = _config_mod.Effect
NoiseInjector = _noise_mod.NoiseInjector

# New taxonomy (Rev 2) imports
radar_frame_deletion       = _noise_mod.radar_frame_deletion
radar_noise_induced_shifts = _noise_mod.radar_noise_induced_shifts
radar_loss_partial         = _noise_mod.radar_loss_partial
radar_loss_complete        = _noise_mod.radar_loss_complete
lidar_frame_deletion       = _noise_mod.lidar_frame_deletion
lidar_gaussian_noise       = _noise_mod.lidar_gaussian_noise
lidar_loss_partial         = _noise_mod.lidar_loss_partial
lidar_loss_complete        = _noise_mod.lidar_loss_complete
camera_frame_deletion      = _noise_mod.camera_frame_deletion
camera_gaussian_noise      = _noise_mod.camera_gaussian_noise
camera_loss_partial        = _noise_mod.camera_loss_partial
camera_loss_complete       = _noise_mod.camera_loss_complete

# Registry imports
RADAR_EFFECTS   = _noise_mod.RADAR_EFFECTS
LIDAR_EFFECTS   = _noise_mod.LIDAR_EFFECTS
CAMERA_EFFECTS  = _noise_mod.CAMERA_EFFECTS
ALL_EFFECTS     = _noise_mod.ALL_EFFECTS
DEFAULT_ORDER_RADAR  = _noise_mod.DEFAULT_ORDER_RADAR
DEFAULT_ORDER_LIDAR  = _noise_mod.DEFAULT_ORDER_LIDAR
DEFAULT_ORDER_CAMERA = _noise_mod.DEFAULT_ORDER_CAMERA

# Legacy imports (for compatibility check)
RADAR_EFFECTS_LEGACY   = _noise_mod.RADAR_EFFECTS_LEGACY
LIDAR_EFFECTS_LEGACY   = _noise_mod.LIDAR_EFFECTS_LEGACY
CAMERA_EFFECTS_LEGACY  = _noise_mod.CAMERA_EFFECTS_LEGACY


# ══════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════


def _make_rdr_sparse(n=200, cols=5):
    """Synthetic rdr_sparse: [x, y, z, power, doppler]."""
    d = np.random.default_rng(0)
    arr = np.column_stack([
        d.uniform(-50, 50, n),   # x
        d.uniform(-50, 50, n),   # y
        d.uniform(-3, 5, n),     # z
        d.uniform(0, 10, n),     # power
        d.uniform(-5, 5, n),     # doppler
    ])
    return arr[:, :cols]


def _make_rdr_polar_3d():
    """Synthetic rdr_polar_3d: (2, 256, 107, 37)."""
    return np.random.default_rng(1).uniform(0, 10, size=(2, 256, 107, 37)).astype(np.float32)


def _make_pc100p(n=200):
    """Synthetic pc100p: [x, y, z, pw, dop]."""
    return _make_rdr_sparse(n, 5).astype(np.float32)


def _make_ldr64(n=500):
    """Synthetic ldr64: [x, y, z, intensity, ring, ...]."""
    d = np.random.default_rng(2)
    arr = np.column_stack([
        d.uniform(-50, 50, n),   # x
        d.uniform(-50, 50, n),   # y
        d.uniform(-3, 5, n),     # z
        d.uniform(0, 255, n),    # intensity
        d.uniform(0, 63, n),     # ring
    ])
    return arr


def _make_camera_img(H=128, W=256):
    """Synthetic camera image as normalized torch.Tensor (3, H, W)."""
    rng = np.random.default_rng(3)
    raw = rng.uniform(0, 255, (H, W, 3)).astype(np.uint8)
    img_f = raw.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_f = (img_f - mean) / std
    return torch.from_numpy(img_f.transpose(2, 0, 1))


def _make_dict_item(with_rdr_sparse=True, with_rdr_polar=True, with_pc100p=True,
                     with_ldr64=True, with_camera=True, rdr_cols=5,
                     rdr_frame_idx=0, ldr_frame_idx=0, cam_frame_idx=0):
    item = {"meta": {
        "seq": "1",
        "idx": {"rdr": rdr_frame_idx, "ldr": ldr_frame_idx, "cam": cam_frame_idx},
    }}
    if with_rdr_sparse:
        item["rdr_sparse"] = _make_rdr_sparse(200, rdr_cols)
    if with_rdr_polar:
        item["rdr_polar_3d"] = _make_rdr_polar_3d()
    if with_pc100p:
        item["pc100p"] = _make_pc100p(200)
    if with_ldr64:
        item["ldr64"] = _make_ldr64(500)
    if with_camera:
        item["front0"] = _make_camera_img()
        item["front1"] = _make_camera_img()
    return item


# ── Test results accumulator ──

_pass_count = 0
_fail_count = 0
_results: list[str] = []


def _check(description: str, condition: bool, detail: str = ""):
    global _pass_count, _fail_count
    if condition:
        _pass_count += 1
        _results.append(f"  PASS  {description}")
    else:
        _fail_count += 1
        msg = f"  FAIL  {description}"
        if detail:
            msg += f"  -- {detail}"
        _results.append(msg)


def _print_results():
    print(f"\n{'='*72}")
    for r in _results:
        print(r)
    total = _pass_count + _fail_count
    print(f"{'='*72}")
    if _fail_count == 0:
        print(f"  ALL {total} TESTS PASSED")
    else:
        print(f"  {_fail_count}/{total} TESTS FAILED")
    print(f"{'='*72}\n")
    return _fail_count == 0


# ══════════════════════════════════════════════
# RADAR TESTS
# ══════════════════════════════════════════════


def test_radar_frame_deletion_deterministic_interval():
    """R1: deterministic frame deletion by interval."""
    rng = np.random.default_rng(200)
    # Frame 20, interval=10 -> should be deleted
    d = _make_dict_item(rdr_frame_idx=20)
    d = radar_frame_deletion(d, {"mode": "deterministic", "interval": 10}, rng)
    _check("R1 del interval: rdr_sparse is None",
           d["rdr_sparse"] is None)
    _check("R1 del interval: rdr_polar_3d is None",
           d["rdr_polar_3d"] is None)
    _check("R1 del interval: pc100p is None",
           d["pc100p"] is None)

    # Frame 21 -> not deleted
    d2 = _make_dict_item(rdr_frame_idx=21)
    d2 = radar_frame_deletion(d2, {"mode": "deterministic", "interval": 10}, rng)
    _check("R1 del interval: frame 21 not deleted",
           d2["rdr_sparse"] is not None)


def test_radar_frame_deletion_deterministic_index_list():
    """R1: deterministic frame deletion by index list."""
    rng = np.random.default_rng(201)
    d = _make_dict_item(rdr_frame_idx=5)
    d = radar_frame_deletion(d, {"mode": "deterministic", "index_list": [5, 10, 15]}, rng)
    _check("R1 del index_list: rdr_sparse is None",
           d["rdr_sparse"] is None)

    d2 = _make_dict_item(rdr_frame_idx=6)
    d2 = radar_frame_deletion(d2, {"mode": "deterministic", "index_list": [5, 10, 15]}, rng)
    _check("R1 del index_list: frame 6 not deleted",
           d2["rdr_sparse"] is not None)


def test_radar_frame_deletion_random():
    """R1: random frame deletion with p=1.0 -> always deletes."""
    rng = np.random.default_rng(202)
    d = _make_dict_item()
    d = radar_frame_deletion(d, {"mode": "random", "p": 1.0}, rng)
    _check("R1 del random p=1: rdr_sparse is None",
           d["rdr_sparse"] is None)
    _check("R1 del random p=1: rdr_polar_3d is None",
           d["rdr_polar_3d"] is None)
    _check("R1 del random p=1: pc100p is None",
           d["pc100p"] is None)

    # p=0.0 -> never deletes
    d2 = _make_dict_item()
    d2 = radar_frame_deletion(d2, {"mode": "random", "p": 0.0}, rng)
    _check("R1 del random p=0: rdr_sparse not None",
           d2["rdr_sparse"] is not None)


def test_radar_noise_induced_shifts():
    """R2: noise-induced shifts change point positions (not power)."""
    rng = np.random.default_rng(203)

    d = _make_dict_item(with_rdr_polar=False, with_pc100p=False)
    old_xyz = d["rdr_sparse"][:, :3].copy()
    old_pw = d["rdr_sparse"][:, 3].copy()

    d = radar_noise_induced_shifts(d, {"shift_std": 2.0}, rng)
    delta = d["rdr_sparse"][:, :3] - old_xyz

    # Position should have changed
    mean_disp = np.mean(np.linalg.norm(delta, axis=1))
    _check("R2 shifts: mean displacement > 0",
           mean_disp > 0.5,
           f"mean_disp={mean_disp:.3f}")

    # Std of displacement ~2.0
    std_disp = np.std(delta)
    _check("R2 shifts: disp std within 40% of shift_std",
           abs(std_disp - 2.0) < 0.8,
           f"std_disp={std_disp:.3f}")

    # Power should NOT have changed (this is a coordinate shift, not power noise)
    _check("R2 shifts: power values unchanged",
           np.allclose(d["rdr_sparse"][:, 3], old_pw, atol=1e-6),
           f"max_power_diff={np.max(np.abs(d['rdr_sparse'][:,3] - old_pw)):.2e}")

    # Test with clip_radius
    d2 = _make_dict_item(with_rdr_polar=False, with_pc100p=False)
    d2 = radar_noise_induced_shifts(d2, {"shift_std": 10.0, "clip_radius": 3.0}, rng)
    delta2 = d2["rdr_sparse"][:, :3] - _make_dict_item(
        with_rdr_polar=False, with_pc100p=False)["rdr_sparse"][:, :3]
    max_disp = np.max(np.linalg.norm(delta2, axis=1))
    _check("R2 shifts: clip_radius respected",
           max_disp <= 3.0 + 1e-6,
           f"max_disp={max_disp:.3f}")


def test_radar_loss_partial():
    """R3: partial loss removes a fraction of points."""
    rng = np.random.default_rng(204)

    d = _make_dict_item(with_rdr_polar=False, with_pc100p=False)
    n_orig = len(d["rdr_sparse"])
    d = radar_loss_partial(d, {"fraction": 0.5}, rng)
    ratio = len(d["rdr_sparse"]) / n_orig
    _check("R3 partial: fraction=0.5 -> count within 5% of 0.5",
           0.45 <= ratio <= 0.55,
           f"ratio={ratio:.3f}")

    # fraction=0.0 -> no change
    d2 = _make_dict_item(with_rdr_polar=False, with_pc100p=False)
    n2 = len(d2["rdr_sparse"])
    d2 = radar_loss_partial(d2, {"fraction": 0.0}, rng)
    _check("R3 partial: fraction=0.0 -> count unchanged",
           len(d2["rdr_sparse"]) == n2)

    # fraction=1.0 -> empty array (NOT None — distinct from loss_complete)
    d3 = _make_dict_item(with_rdr_polar=False, with_pc100p=False)
    d3 = radar_loss_partial(d3, {"fraction": 1.0}, rng)
    _check("R3 partial: fraction=1.0 -> empty array, not None",
           len(d3["rdr_sparse"]) == 0 and d3["rdr_sparse"] is not None)


def test_radar_loss_complete():
    """R4: complete loss sets all radar keys to None."""
    rng = np.random.default_rng(205)
    d = _make_dict_item()
    d = radar_loss_complete(d, {}, rng)
    _check("R4 complete: rdr_sparse is None",
           d["rdr_sparse"] is None)
    _check("R4 complete: rdr_polar_3d is None",
           d["rdr_polar_3d"] is None)
    _check("R4 complete: pc100p is None",
           d["pc100p"] is None)


def test_radar_loss_partial_vs_complete_distinction():
    """Partial (fraction=1.0) produces empty array; complete produces None."""
    rng = np.random.default_rng(206)
    d_partial = _make_dict_item(with_rdr_polar=False, with_pc100p=False)
    d_complete = _make_dict_item(with_rdr_polar=False, with_pc100p=False)

    d_partial = radar_loss_partial(d_partial, {"fraction": 1.0}, rng)
    d_complete = radar_loss_complete(d_complete, {}, rng)

    _check("R3 vs R4: partial -> array, not None",
           isinstance(d_partial["rdr_sparse"], np.ndarray) and len(d_partial["rdr_sparse"]) == 0)
    _check("R3 vs R4: complete -> None",
           d_complete["rdr_sparse"] is None)


# ══════════════════════════════════════════════
# LIDAR TESTS
# ══════════════════════════════════════════════


def test_lidar_frame_deletion():
    """L1: frame deletion sets ldr64 to None."""
    rng = np.random.default_rng(207)

    # Deterministic: interval
    d = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                         with_camera=False, ldr_frame_idx=30)
    d = lidar_frame_deletion(d, {"mode": "deterministic", "interval": 10}, rng)
    _check("L1 del interval: ldr64 is None",
           d["ldr64"] is None)

    d2 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_camera=False, ldr_frame_idx=31)
    d2 = lidar_frame_deletion(d2, {"mode": "deterministic", "interval": 10}, rng)
    _check("L1 del interval: frame 31 not deleted",
           d2["ldr64"] is not None)

    # Index list
    d3 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_camera=False, ldr_frame_idx=3)
    d3 = lidar_frame_deletion(d3, {"mode": "deterministic", "index_list": [3, 7]}, rng)
    _check("L1 del index_list: ldr64 is None",
           d3["ldr64"] is None)

    # Random: p=1.0
    d4 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_camera=False)
    d4 = lidar_frame_deletion(d4, {"mode": "random", "p": 1.0}, rng)
    _check("L1 del random p=1: ldr64 is None",
           d4["ldr64"] is None)


def test_lidar_gaussian_noise():
    """L2: Gaussian noise on LiDAR positions (and optionally intensity)."""
    rng = np.random.default_rng(208)

    d = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                         with_camera=False)
    old_pc = d["ldr64"][:, :3].copy()
    old_intensity = d["ldr64"][:, 3].copy()

    d = lidar_gaussian_noise(d, {"sigma_xy": 0.2, "sigma_z": 0.1, "sigma_intensity": 5.0}, rng)
    delta = d["ldr64"][:, :3] - old_pc

    std_xy = np.std(delta[:, :2])
    std_z = np.std(delta[:, 2])
    _check("L2 noise xy: std within 30% of sigma_xy",
           abs(std_xy - 0.2) < 0.06,
           f"std_xy={std_xy:.4f}")
    _check("L2 noise z: std within 30% of sigma_z",
           abs(std_z - 0.1) < 0.03,
           f"std_z={std_z:.4f}")

    # Intensity should have changed (sigma_intensity > 0)
    intensity_delta = d["ldr64"][:, 3] - old_intensity
    _check("L2 noise: intensity changed with sigma_intensity=5",
           np.std(intensity_delta) > 1.0,
           f"std_intensity={np.std(intensity_delta):.3f}")

    # sigma_intensity=0 -> no intensity change
    d2 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_camera=False)
    old_i2 = d2["ldr64"][:, 3].copy()
    d2 = lidar_gaussian_noise(d2, {"sigma_xy": 0.1, "sigma_intensity": 0.0}, rng)
    _check("L2 noise: sigma_intensity=0 -> intensity unchanged",
           np.allclose(d2["ldr64"][:, 3], old_i2, atol=1e-6))


def test_lidar_loss_partial():
    """L3: partial LiDAR loss removes fraction of points."""
    rng = np.random.default_rng(209)
    d = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                         with_camera=False)
    n_orig = len(d["ldr64"])
    d = lidar_loss_partial(d, {"fraction": 0.25}, rng)
    ratio = len(d["ldr64"]) / n_orig
    _check("L3 partial: fraction=0.25 -> count within 5% of 0.75",
           0.70 <= ratio <= 0.80,
           f"ratio={ratio:.3f}")

    d2 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_camera=False)
    d2 = lidar_loss_partial(d2, {"fraction": 0.0}, rng)
    _check("L3 partial: fraction=0 -> no change",
           len(d2["ldr64"]) == 500)

    d3 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_camera=False)
    d3 = lidar_loss_partial(d3, {"fraction": 1.0}, rng)
    _check("L3 partial: fraction=1 -> empty array, not None",
           len(d3["ldr64"]) == 0 and d3["ldr64"] is not None)


def test_lidar_loss_complete():
    """L4: complete LiDAR loss sets ldr64 to None."""
    rng = np.random.default_rng(210)
    d = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                         with_camera=False)
    d = lidar_loss_complete(d, {}, rng)
    _check("L4 complete: ldr64 is None",
           d["ldr64"] is None)


# ══════════════════════════════════════════════
# CAMERA TESTS
# ══════════════════════════════════════════════


def test_camera_frame_deletion():
    """C1: camera frame deletion zeros images."""
    rng = np.random.default_rng(211)
    d = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                         with_ldr64=False, cam_frame_idx=40)
    d = camera_frame_deletion(d, {"mode": "deterministic", "interval": 10}, rng)
    # Black pixel in normalized space ≈ -mean/std ≈ -2.1
    _check("C1 del interval: front0 all near-black",
           torch.all(d["front0"] < -1.5).item(),
           f"min={d['front0'].min().item():.3f}")
    _check("C1 del interval: front1 also zeroed",
           torch.all(d["front1"] < -1.5).item())

    # Frame 41 not deleted
    d2 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_ldr64=False, cam_frame_idx=41)
    d2 = camera_frame_deletion(d2, {"mode": "deterministic", "interval": 10}, rng)
    _check("C1 del interval: frame 41 not all-black",
           torch.any(d2["front0"] > -1.0).item())

    # Random p=1.0
    d3 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_ldr64=False)
    d3 = camera_frame_deletion(d3, {"mode": "random", "p": 1.0}, rng)
    _check("C1 del random p=1: front0 all near-black",
           torch.all(d3["front0"] < -1.5).item())


def test_camera_gaussian_noise():
    """C2: Gaussian noise on camera images."""
    rng = np.random.default_rng(212)
    d = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                         with_ldr64=False)
    old = d["front0"].clone()
    d = camera_gaussian_noise(d, {"sigma": 0.0}, rng)
    _check("C2 noise sigma=0: unchanged",
           torch.allclose(d["front0"], old, atol=1e-7))

    d2 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_ldr64=False)
    d2 = camera_gaussian_noise(d2, {"sigma": 25.0}, rng)
    diff = d2["front0"] - _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False,
                                           with_pc100p=False, with_ldr64=False)["front0"]
    _check("C2 noise sigma=25: std > 0",
           torch.std(diff).item() > 0,
           f"std={torch.std(diff).item():.4f}")

    _check("C2 noise: both cameras processed",
           "front1" in d2 and d2["front0"].shape == d2["front1"].shape)


def test_camera_loss_partial():
    """C3: partial camera loss zeros a contiguous region."""
    rng = np.random.default_rng(213)
    d = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                         with_ldr64=False)
    H, W = 128, 256
    d["front0"] = _make_camera_img(H, W)

    # fraction=0.0 -> no change
    old = d["front0"].clone()
    d2 = camera_loss_partial(d, {"fraction": 0.0}, rng)
    _check("C3 partial fraction=0: near-unchanged (round-trip tolerance)",
           torch.allclose(d2["front0"], old, atol=0.02))

    # fraction=0.3 -> black region present
    d3 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_ldr64=False)
    d3["front0"] = _make_camera_img(H, W)
    d3 = camera_loss_partial(d3, {"fraction": 0.3}, rng)
    # Black pixels in normalized space are < -1.5
    _check("C3 partial fraction=0.3: some near-black pixels present",
           (d3["front0"] < -1.5).any().item(),
           f"min={d3['front0'].min().item():.3f}")

    # Fraction close to 1.0 (clipped to 0.99)
    d4 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_ldr64=False)
    d4["front0"] = _make_camera_img(H, W)
    d4 = camera_loss_partial(d4, {"fraction": 0.99}, rng)
    # Most but not all pixels should be black
    black_frac = (d4["front0"] < -1.5).sum().item() / d4["front0"].numel()
    _check("C3 partial fraction=0.99: most pixels black",
           black_frac > 0.5,
           f"black_frac={black_frac:.3f}")


def test_camera_loss_complete():
    """C4: complete camera loss sets all pixels to black."""
    rng = np.random.default_rng(214)
    d = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                         with_ldr64=False)
    d = camera_loss_complete(d, {}, rng)
    # All pixels should be near-black (< -1.5 in normalized space)
    _check("C4 complete: all front0 pixels near-black",
           torch.all(d["front0"] < -1.5).item(),
           f"min={d['front0'].min().item():.3f}, max={d['front0'].max().item():.3f}")
    _check("C4 complete: all front1 pixels near-black",
           torch.all(d["front1"] < -1.5).item())


def test_camera_loss_partial_single_region():
    """C3: partial loss produces exactly one contiguous zeroed region (not N patches)."""
    rng = np.random.default_rng(215)
    d = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                         with_ldr64=False)
    H, W = 128, 256
    d["front0"] = _make_camera_img(H, W)
    d = camera_loss_partial(d, {"fraction": 0.2}, rng)

    # Find black region(s) in uint8 space
    img_np = d["front0"].cpu().numpy().transpose(1, 2, 0)
    # De-normalize: find pixels with all channels < 0.01
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_uint8 = np.clip((img_np * std + mean) * 255.0, 0, 255).astype(np.uint8)
    black_mask = np.all(img_uint8 == 0, axis=2)

    # Count connected components of black regions
    try:
        from scipy import ndimage as ndi
        labeled, n_regions = ndi.label(black_mask)
        _check("C3 partial: exactly 1 zeroed region",
               n_regions == 1,
               f"n_regions={n_regions}")
    except ImportError:
        # scipy not installed — skip the connected-components check
        # Still verify at least one black pixel exists
        _check("C3 partial: at least one black pixel (scipy N/A)",
               black_mask.sum() > 0,
               f"black_pixels={black_mask.sum()}")


# ══════════════════════════════════════════════
# REGISTRY TESTS
# ══════════════════════════════════════════════


def test_registry_counts():
    """Each modality registry has exactly 4 effects."""
    _check("REG: radar has 4 effects",
           len(RADAR_EFFECTS) == 4,
           f"got {len(RADAR_EFFECTS)}: {list(RADAR_EFFECTS.keys())}")
    _check("REG: lidar has 4 effects",
           len(LIDAR_EFFECTS) == 4,
           f"got {len(LIDAR_EFFECTS)}: {list(LIDAR_EFFECTS.keys())}")
    _check("REG: camera has 4 effects",
           len(CAMERA_EFFECTS) == 4,
           f"got {len(CAMERA_EFFECTS)}: {list(CAMERA_EFFECTS.keys())}")


def test_registry_effect_names():
    """Registries contain exactly the expected effect names."""
    expected_radar = {"frame_deletion", "noise_induced_shifts", "loss_partial", "loss_complete"}
    expected_lidar = {"frame_deletion", "gaussian_noise", "loss_partial", "loss_complete"}
    expected_camera = {"frame_deletion", "gaussian_noise", "loss_partial", "loss_complete"}

    _check("REG: radar names match",
           set(RADAR_EFFECTS.keys()) == expected_radar,
           f"extra={set(RADAR_EFFECTS.keys()) - expected_radar}, "
           f"missing={expected_radar - set(RADAR_EFFECTS.keys())}")
    _check("REG: lidar names match",
           set(LIDAR_EFFECTS.keys()) == expected_lidar)
    _check("REG: camera names match",
           set(CAMERA_EFFECTS.keys()) == expected_camera)


def test_default_ordering():
    """Default ordering lists reference valid registry names."""
    for name in DEFAULT_ORDER_RADAR:
        _check(f"REG default radar: {name} in RADAR_EFFECTS",
               name in RADAR_EFFECTS)
    for name in DEFAULT_ORDER_LIDAR:
        _check(f"REG default lidar: {name} in LIDAR_EFFECTS",
               name in LIDAR_EFFECTS)
    for name in DEFAULT_ORDER_CAMERA:
        _check(f"REG default camera: {name} in CAMERA_EFFECTS",
               name in CAMERA_EFFECTS)

    # Frame deletion should be first in all defaults
    _check("REG: frame_deletion first in radar default",
           DEFAULT_ORDER_RADAR[0] == "frame_deletion")
    _check("REG: frame_deletion first in lidar default",
           DEFAULT_ORDER_LIDAR[0] == "frame_deletion")
    _check("REG: frame_deletion first in camera default",
           DEFAULT_ORDER_CAMERA[0] == "frame_deletion")

    # Loss complete should be last in all defaults
    _check("REG: loss_complete last in radar default",
           DEFAULT_ORDER_RADAR[-1] == "loss_complete")
    _check("REG: loss_complete last in lidar default",
           DEFAULT_ORDER_LIDAR[-1] == "loss_complete")
    _check("REG: loss_complete last in camera default",
           DEFAULT_ORDER_CAMERA[-1] == "loss_complete")


def test_legacy_registries_preserved():
    """Legacy registries still exist with the old effect names."""
    _check("REG legacy: radar has 7 effects",
           len(RADAR_EFFECTS_LEGACY) == 7,
           f"got {len(RADAR_EFFECTS_LEGACY)}")
    _check("REG legacy: lidar has 5 effects",
           len(LIDAR_EFFECTS_LEGACY) == 5)
    _check("REG legacy: camera has 5 effects (gaussian_noise carried forward, no legacy copy)",
           len(CAMERA_EFFECTS_LEGACY) == 5)

    # Key legacy names present
    _check("REG legacy: power_gaussian_noise present",
           "power_gaussian_noise" in RADAR_EFFECTS_LEGACY)
    _check("REG legacy: lidar_fov_occlusion present (as fov_occlusion)",
           "fov_occlusion" in LIDAR_EFFECTS_LEGACY)
    _check("REG legacy: camera_motion_blur present",
           "motion_blur" in CAMERA_EFFECTS_LEGACY)


# ══════════════════════════════════════════════
# NOISEINJECTOR INTEGRATION TESTS
# ══════════════════════════════════════════════


def test_injector_basic():
    """Injector runs end-to-end with one new-taxonomy effect per modality."""
    config = EffectConfig(
        seed=42,
        radar=[Effect("noise_induced_shifts", p=1.0, params={"shift_std": 0.5})],
        lidar=[Effect("loss_partial", p=1.0, params={"fraction": 0.2})],
        camera=[Effect("gaussian_noise", p=1.0, params={"sigma": 5})],
    )
    injector = NoiseInjector(config)
    d = _make_dict_item()
    d = injector(d)

    _check("INJ basic: rdr_sparse modified",
           "rdr_sparse" in d)
    _check("INJ basic: ldr64 count reduced",
           len(d["ldr64"]) < 500,
           f"count={len(d['ldr64'])}")
    _check("INJ basic: metadata injected",
           "meta" in d and "noise_injection" in d["meta"])
    meta = d["meta"]["noise_injection"]
    _check("INJ basic: radar effect listed in metadata",
           len(meta["radar"]) == 1)
    _check("INJ basic: metadata seed match",
           meta["seed"] == 42)


def test_injector_frame_deletion_deterministic():
    """Frame deletion with deterministic interval via injector."""
    config = EffectConfig(
        seed=42,
        radar=[Effect("frame_deletion", p=1.0,
                       params={"mode": "deterministic", "interval": 10})],
    )
    injector = NoiseInjector(config)

    # Frame 0 -> deleted (0 % 10 == 0)
    d0 = _make_dict_item(rdr_frame_idx=0)
    d0 = injector(d0, frame_index=0)
    _check("INJ frame_del: frame 0 rdr_sparse is None",
           d0["rdr_sparse"] is None)

    # Frame 5 -> not deleted
    d5 = _make_dict_item(rdr_frame_idx=5)
    d5 = injector(d5, frame_index=5)
    _check("INJ frame_del: frame 5 rdr_sparse not None",
           d5["rdr_sparse"] is not None)


def test_injector_prob_zero():
    """p=0.0 -> no effects applied."""
    config = EffectConfig(
        seed=42,
        radar=[Effect("noise_induced_shifts", p=0.0, params={"shift_std": 0.5})],
        lidar=[Effect("loss_partial", p=0.0, params={"fraction": 0.2})],
    )
    injector = NoiseInjector(config)
    d = _make_dict_item()
    n_rdr = len(d["rdr_sparse"])
    n_ldr = len(d["ldr64"])
    d = injector(d)
    _check("INJ p=0: rdr count unchanged",
           len(d["rdr_sparse"]) == n_rdr)
    _check("INJ p=0: ldr count unchanged",
           len(d["ldr64"]) == n_ldr)
    _check("INJ p=0: metadata still present",
           "noise_injection" in d["meta"])


def test_injector_empty_config():
    """No effect lists -> identity pass-through."""
    config = EffectConfig(seed=42)
    injector = NoiseInjector(config)
    d = _make_dict_item()
    n_rdr = len(d["rdr_sparse"])
    n_ldr = len(d["ldr64"])
    d = injector(d)
    _check("INJ empty config: rdr unchanged",
           len(d["rdr_sparse"]) == n_rdr)
    _check("INJ empty config: ldr unchanged",
           len(d["ldr64"]) == n_ldr)
    _check("INJ empty config: noise_injection exists",
           "noise_injection" in d["meta"])


def test_injector_unknown_effect():
    """Unknown effect name raises ValueError."""
    config = EffectConfig(
        radar=[Effect("nonexistent_effect", p=1.0)],
    )
    try:
        NoiseInjector(config)
        _check("INJ unknown effect: ValueError raised", False)
    except ValueError as e:
        _check("INJ unknown effect: ValueError raised", True, str(e)[:80])


def test_injector_seed_determinism():
    """Same seed -> same output for stochastic effects."""
    def run_injector(seed):
        config = EffectConfig(
            seed=seed,
            radar=[Effect("loss_partial", p=1.0, params={"fraction": 0.5})],
        )
        injector = NoiseInjector(config)
        d = _make_dict_item()
        return injector(d)["rdr_sparse"]

    r1 = run_injector(42)
    r2 = run_injector(42)
    _check("INJ determinism: same seed -> same output",
           np.array_equal(r1, r2),
           f"r1.shape={r1.shape} r2.shape={r2.shape}")

    r3 = run_injector(99)
    _check("INJ determinism: different seed -> different output",
           not np.array_equal(r1, r3))


def test_injector_no_corruption_on_empty_keys():
    """Injector handles missing sensor keys gracefully."""
    config = EffectConfig(
        radar=[Effect("noise_induced_shifts", p=1.0, params={"shift_std": 0.5})],
        lidar=[Effect("loss_partial", p=1.0, params={"fraction": 0.2})],
    )
    injector = NoiseInjector(config)
    d = {"meta": {}}
    d = injector(d)
    _check("INJ missing keys: no crash, metadata present",
           "noise_injection" in d["meta"])


def test_injector_should_skip():
    """should_skip pre-check works for deterministic frame deletion."""
    config = EffectConfig(
        seed=42,
        radar=[Effect("frame_deletion", p=1.0,
                       params={"mode": "deterministic", "interval": 10})],
    )
    injector = NoiseInjector(config)

    _check("INJ should_skip: frame 0 -> True",
           injector.should_skip(0, "radar") is True)
    _check("INJ should_skip: frame 5 -> False",
           injector.should_skip(5, "radar") is False)
    _check("INJ should_skip: frame 10 -> True",
           injector.should_skip(10, "radar") is True)

    # No frame_deletion for lidar -> always False
    _check("INJ should_skip: lidar no frame_del -> False",
           injector.should_skip(0, "lidar") is False)


def test_injector_legacy_effect_rejected():
    """Legacy effect names raise ValueError (not in active registries)."""
    config = EffectConfig(
        radar=[Effect("power_gaussian_noise", p=1.0)],
    )
    try:
        NoiseInjector(config)
        _check("INJ legacy rejected: ValueError raised", False)
    except ValueError:
        _check("INJ legacy rejected: ValueError raised", True)


def test_injector_frame_deletion_index_list():
    """Frame deletion via explicit index list."""
    config = EffectConfig(
        seed=42,
        lidar=[Effect("frame_deletion", p=1.0,
                       params={"mode": "deterministic", "index_list": [3, 7, 15]})],
    )
    injector = NoiseInjector(config)

    d3 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_camera=False, ldr_frame_idx=3)
    d3 = injector(d3, frame_index=3)
    _check("INJ index_list: frame 3 -> ldr64 is None",
           d3["ldr64"] is None)

    d5 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_camera=False, ldr_frame_idx=5)
    d5 = injector(d5, frame_index=5)
    _check("INJ index_list: frame 5 -> ldr64 not None",
           d5["ldr64"] is not None)


def test_injector_has_key_skips_none():
    """Effects skip sensor keys set to None by frame_deletion."""
    config = EffectConfig(
        seed=42,
        radar=[
            Effect("frame_deletion", p=1.0,
                   params={"mode": "deterministic", "interval": 1}),
            Effect("noise_induced_shifts", p=1.0, params={"shift_std": 1.0}),
        ],
    )
    injector = NoiseInjector(config)
    d = _make_dict_item(rdr_frame_idx=0)
    d = injector(d, frame_index=0)
    # Frame deletion fired -> rdr_sparse is None -> noise_induced_shifts should
    # not crash (it will skip via _has_key)
    _check("INJ has_key skip: rdr_sparse is None (not crash)",
           d["rdr_sparse"] is None)
    _check("INJ has_key skip: metadata still populated",
           "noise_injection" in d["meta"])


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

if __name__ == "__main__":
    # Radar
    test_radar_frame_deletion_deterministic_interval()
    test_radar_frame_deletion_deterministic_index_list()
    test_radar_frame_deletion_random()
    test_radar_noise_induced_shifts()
    test_radar_loss_partial()
    test_radar_loss_complete()
    test_radar_loss_partial_vs_complete_distinction()

    # LiDAR
    test_lidar_frame_deletion()
    test_lidar_gaussian_noise()
    test_lidar_loss_partial()
    test_lidar_loss_complete()

    # Camera
    test_camera_frame_deletion()
    test_camera_gaussian_noise()
    test_camera_loss_partial()
    test_camera_loss_complete()
    test_camera_loss_partial_single_region()

    # Registries
    test_registry_counts()
    test_registry_effect_names()
    test_default_ordering()
    test_legacy_registries_preserved()

    # Injector integration
    test_injector_basic()
    test_injector_frame_deletion_deterministic()
    test_injector_prob_zero()
    test_injector_empty_config()
    test_injector_unknown_effect()
    test_injector_seed_determinism()
    test_injector_no_corruption_on_empty_keys()
    test_injector_should_skip()
    test_injector_legacy_effect_rejected()
    test_injector_frame_deletion_index_list()
    test_injector_has_key_skips_none()

    success = _print_results()
    sys.exit(0 if success else 1)