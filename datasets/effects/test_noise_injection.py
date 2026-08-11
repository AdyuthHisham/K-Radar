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


def _make_kradar_dict_item(rdr="00182", ldr64="00150", camf="00449",
                           ldr128="00150", camr="00451",
                           tstamp="1643292961.606203716"):
    """A dict_item whose meta['idx'] matches the REAL K-Radar dataset exactly.

    _make_dict_item above keys meta['idx'] with the injector's own modality
    names ('rdr'/'ldr'/'cam'), which is why the suite could not catch the bug
    where deterministic frame_deletion silently never fired for lidar or
    camera: the dataset actually writes

        dict_idx = dict(rdr=rdr, ldr64=ldr64, camf=camf,
                        ldr128=ldr128, camr=camr, tstamp=tstamp)

    (kradar_fusion_v1_0.py:442, and identically in kradar_detection_v2_0.py:201
    and kradar_detection_v1_1.py:154). Only 'rdr' coincides. Values are
    zero-padded STRINGS, because they come from a split('_') of the label
    header -- K-Radar uses them for path building, never arithmetic.

    Defaults are a real frame captured from the Alvis sensor dumps.
    """
    item = {"meta": {
        "seq": "1",
        "idx": dict(rdr=rdr, ldr64=ldr64, camf=camf,
                    ldr128=ldr128, camr=camr, tstamp=tstamp),
    }}
    item["rdr_sparse"] = _make_rdr_sparse(200, 5)
    item["rdr_polar_3d"] = _make_rdr_polar_3d()
    item["ldr64"] = _make_ldr64(500)
    item["front0"] = _make_camera_img()
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
    # pc100p is NOT set to None by frame deletion (it is untouched)
    _check("R1 del interval: pc100p NOT modified",
           d["pc100p"] is not None,
           f"type={type(d['pc100p']).__name__}")

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
    # pc100p is NOT set to None (untouched by frame deletion)
    _check("R1 del random p=1: pc100p NOT modified",
           d["pc100p"] is not None,
           f"type={type(d['pc100p']).__name__}")

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
    """R3: partial loss zeroes out a fraction of rows (shape preserved)."""
    rng = np.random.default_rng(204)

    # fraction=0.5 -> exactly int(N*0.5) rows = all-zero, count unchanged
    d = _make_dict_item(with_rdr_polar=False, with_pc100p=False)
    arr_orig = d["rdr_sparse"].copy()
    n_orig = len(arr_orig)
    d = radar_loss_partial(d, {"fraction": 0.5}, rng)
    n_after = len(d["rdr_sparse"])
    n_zeroed = int(np.sum(np.all(d["rdr_sparse"] == 0, axis=1)))
    expected_zeroed = int(n_orig * 0.5)
    _check("R3 partial: fraction=0.5 -> row count unchanged",
           n_after == n_orig,
           f"n_after={n_after}, n_orig={n_orig}")
    _check("R3 partial: fraction=0.5 -> exactly int(N*frac) rows zeroed",
           n_zeroed == expected_zeroed,
           f"zeroed={n_zeroed}, expected={expected_zeroed}")
    # Non-selected rows are untouched
    zeroed_indices = np.where(np.all(d["rdr_sparse"] == 0, axis=1))[0]
    kept_mask = np.ones(n_orig, dtype=bool)
    kept_mask[zeroed_indices] = False
    _check("R3 partial: non-selected rows unchanged",
           np.allclose(d["rdr_sparse"][kept_mask], arr_orig[kept_mask]),
           "non-selected rows differed from original")

    # fraction=0.0 -> no change
    d2 = _make_dict_item(with_rdr_polar=False, with_pc100p=False)
    n2 = len(d2["rdr_sparse"])
    d2 = radar_loss_partial(d2, {"fraction": 0.0}, rng)
    _check("R3 partial: fraction=0.0 -> count unchanged",
           len(d2["rdr_sparse"]) == n2)
    _check("R3 partial: fraction=0.0 -> no zeroed rows",
           np.all(d2["rdr_sparse"] != 0))

    # fraction=1.0 -> all rows zeroed, array NOT None (distinct from loss_complete)
    d3 = _make_dict_item(with_rdr_polar=False, with_pc100p=False)
    n3 = len(d3["rdr_sparse"])
    d3 = radar_loss_partial(d3, {"fraction": 1.0}, rng)
    _check("R3 partial: fraction=1.0 -> all rows zeroed, not None",
           d3["rdr_sparse"] is not None and len(d3["rdr_sparse"]) == n3
           and np.all(d3["rdr_sparse"] == 0))


def test_radar_loss_complete():
    """R4: complete loss sets all radar keys to None."""
    rng = np.random.default_rng(205)
    d = _make_dict_item()
    d = radar_loss_complete(d, {}, rng)
    _check("R4 complete: rdr_sparse is None",
           d["rdr_sparse"] is None)
    _check("R4 complete: rdr_polar_3d is None",
           d["rdr_polar_3d"] is None)
    # pc100p is NOT set to None by loss_complete (it is untouched)
    _check("R4 complete: pc100p NOT modified",
           d["pc100p"] is not None,
           f"type={type(d['pc100p']).__name__}")


def test_radar_loss_partial_vs_complete_distinction():
    """Partial (fraction=1.0) = all rows zeroed; complete = None."""
    rng = np.random.default_rng(206)
    d_partial = _make_dict_item(with_rdr_polar=False, with_pc100p=False)
    d_complete = _make_dict_item(with_rdr_polar=False, with_pc100p=False)

    d_partial = radar_loss_partial(d_partial, {"fraction": 1.0}, rng)
    d_complete = radar_loss_complete(d_complete, {}, rng)

    _check("R3 vs R4: partial -> all-zero array, not None",
           isinstance(d_partial["rdr_sparse"], np.ndarray)
           and len(d_partial["rdr_sparse"]) > 0
           and np.all(d_partial["rdr_sparse"] == 0))
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
    """L3: partial LiDAR loss zeroes out a fraction of points (shape preserved)."""
    rng = np.random.default_rng(209)
    d = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                         with_camera=False)
    arr_orig = d["ldr64"].copy()
    n_orig = len(arr_orig)
    d = lidar_loss_partial(d, {"fraction": 0.25}, rng)
    n_after = len(d["ldr64"])
    n_zeroed = int(np.sum(np.all(d["ldr64"] == 0, axis=1)))
    expected_zeroed = int(n_orig * 0.25)
    _check("L3 partial: fraction=0.25 -> row count unchanged",
           n_after == n_orig,
           f"n_after={n_after}, n_orig={n_orig}")
    _check("L3 partial: fraction=0.25 -> exactly int(N*frac) rows zeroed",
           n_zeroed == expected_zeroed,
           f"zeroed={n_zeroed}, expected={expected_zeroed}")
    # Non-selected rows untouched
    zeroed_idx = np.where(np.all(d["ldr64"] == 0, axis=1))[0]
    kept = np.ones(n_orig, dtype=bool)
    kept[zeroed_idx] = False
    _check("L3 partial: non-selected rows unchanged",
           np.allclose(d["ldr64"][kept], arr_orig[kept]),
           "non-selected rows differed from original")

    d2 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_camera=False)
    d2 = lidar_loss_partial(d2, {"fraction": 0.0}, rng)
    _check("L3 partial: fraction=0 -> no change",
           len(d2["ldr64"]) == 500)
    _check("L3 partial: fraction=0 -> no zeroed rows",
           np.all(d2["ldr64"] != 0))

    d3 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_camera=False)
    n3 = len(d3["ldr64"])
    d3 = lidar_loss_partial(d3, {"fraction": 1.0}, rng)
    _check("L3 partial: fraction=1 -> all rows zeroed, not None",
           d3["ldr64"] is not None and len(d3["ldr64"]) == n3
           and np.all(d3["ldr64"] == 0))


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
    """C3: partial camera loss via flatten-permute-zero (AI-MSF-Benchmark port)."""
    rng = np.random.default_rng(213)
    d = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                         with_ldr64=False)

    # fraction=0.0 -> no change
    H, W = 128, 256
    d["front0"] = _make_camera_img(H, W)
    old = d["front0"].clone()
    d2 = camera_loss_partial(d, {"fraction": 0.0}, rng)
    _check("C3 partial fraction=0: near-unchanged (round-trip tolerance)",
           torch.allclose(d2["front0"], old, atol=0.02))

    # fraction=0.3 -> exactly floor(H*W*3*0.3) scalar values zeroed, shape unchanged
    d3 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_ldr64=False)
    # Use a mid-gray image (no natural zeros) for exact count check
    raw = np.full((H, W, 3), 100, dtype=np.uint8)
    img_f = raw.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_f = (img_f - mean) / std
    d3["front0"] = torch.from_numpy(img_f.transpose(2, 0, 1))
    d3 = camera_loss_partial(d3, {"fraction": 0.3}, rng)
    # Check shape unchanged
    _check("C3 partial: shape unchanged",
           d3["front0"].shape == (3, H, W),
           f"shape={d3['front0'].shape}")
    # Check exact count zeroed in uint8 space
    img_np = d3["front0"].cpu().numpy().transpose(1, 2, 0)  # (H,W,3)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_uint8 = np.clip((img_np * std + mean) * 255.0, 0, 255).astype(np.uint8)
    n_zeroed = int(np.sum(img_uint8 == 0))
    expected_zeroed = int(H * W * 3 * 0.3)
    _check("C3 partial fraction=0.3: exact scalar count zeroed",
           n_zeroed == expected_zeroed,
           f"zeroed={n_zeroed}, expected={expected_zeroed}")

    # fraction=1.0 -> all scalar values zeroed
    d4 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_ldr64=False)
    d4["front0"] = _make_camera_img(H, W)
    d4 = camera_loss_partial(d4, {"fraction": 1.0}, rng)
    img_np4 = d4["front0"].cpu().numpy().transpose(1, 2, 0)
    img_uint8_4 = np.clip((img_np4 * std + mean) * 255.0, 0, 255).astype(np.uint8)
    _check("C3 partial fraction=1.0: all scalar values zeroed",
           np.all(img_uint8_4 == 0),
           f"non_zero={(img_uint8_4 != 0).sum()}")
    _check("C3 partial fraction=1.0: tensor NOT None (distinct from loss_complete)",
           d4["front0"] is not None)
    _check("C3 partial fraction=1.0: both cameras processed",
           "front1" in d4 and d4["front0"].shape == d4["front1"].shape)


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


def test_camera_loss_partial_cross_channel_independence():
    """C3: channel-independent dropout — R/G/B of the same pixel are NOT always zeroed together."""
    rng = np.random.default_rng(215)
    H, W = 32, 32
    d = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                         with_ldr64=False)
    # Create an image where every pixel has all 3 channels != 0
    raw = np.full((H, W, 3), 100, dtype=np.uint8)  # mid-gray
    img_f = raw.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_f = (img_f - mean) / std
    d["front0"] = torch.from_numpy(img_f.transpose(2, 0, 1))

    # Apply loss_partial with moderate fraction so some but not all channels get zeroed
    d = camera_loss_partial(d, {"fraction": 0.4}, rng)

    # Examine in uint8 space
    img_np = d["front0"].cpu().numpy().transpose(1, 2, 0)
    img_uint8 = np.clip((img_np * std + mean) * 255.0, 0, 255).astype(np.uint8)

    # Count per-pixel how many channels are zero
    zeroed_channels_per_pixel = np.sum(img_uint8 == 0, axis=2)

    has_partial_zero = np.any((zeroed_channels_per_pixel > 0) & (zeroed_channels_per_pixel < 3))
    has_full_zero = np.any(zeroed_channels_per_pixel == 3)
    _check("C3 cross-channel: some pixels have 1 or 2 channels zeroed (independent dropout)",
           has_partial_zero,
           f"pixels with 1 zeroed: {np.sum(zeroed_channels_per_pixel == 1)}, "
           f"2 zeroed: {np.sum(zeroed_channels_per_pixel == 2)}, "
           f"3 zeroed: {np.sum(zeroed_channels_per_pixel == 3)}")

    # At fraction=0.4, at least some pixels should also have all 3 channels intact
    has_unchanged = np.any(zeroed_channels_per_pixel == 0)
    _check("C3 cross-channel: some pixels fully intact",
           has_unchanged,
           f"unchanged pixels: {np.sum(zeroed_channels_per_pixel == 0)}")


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
    _check("INJ basic: ldr64 count preserved, rows zeroed",
           len(d["ldr64"]) == 500 and np.any(np.all(d["ldr64"] == 0, axis=1)),
           f"count={len(d['ldr64'])}, zeroed={int(np.sum(np.all(d['ldr64'] == 0, axis=1)))}")
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


def test_kradar_meta_idx_resolution():
    """_get_modality_idx resolves against the real K-Radar meta['idx'] keys.

    Regression for the bug where lidar/camera resolved to None because the
    lookup used the injector's modality names rather than the dataset's
    ('ldr64', 'camf'). Both the short names the effect functions pass and the
    long names used in configs must resolve.
    """
    get_idx = _noise_mod._get_modality_idx
    d = _make_kradar_dict_item()
    for name, expected in [("rdr", 182), ("ldr", 150), ("cam", 449),
                           ("radar", 182), ("lidar", 150), ("camera", 449)]:
        _check(f"KRADAR meta idx: '{name}' resolves to {expected}",
               get_idx(d, name) == expected,
               f"got {get_idx(d, name)!r}")

    # Zero-padded strings must be coerced, since frame_deletion does modulo.
    _check("KRADAR meta idx: value is int, not str",
           isinstance(get_idx(d, "lidar"), int))
    # Absent / unparseable indices degrade to None rather than raising.
    _check("KRADAR meta idx: missing key -> None",
           get_idx({"meta": {"idx": {}}}, "lidar") is None)
    _check("KRADAR meta idx: non-numeric -> None",
           get_idx({"meta": {"idx": {"ldr64": "abc"}}}, "lidar") is None)


def test_kradar_frame_deletion_fires_all_modalities():
    """Deterministic frame_deletion fires for every modality on real meta keys.

    This is the test that would have caught the original bug: with the old
    lookup, lidar and camera silently no-opped here while radar worked, and
    the run looked healthy.
    """
    params = {"mode": "deterministic", "interval": 2}
    rng = np.random.default_rng(0)

    # ldr64 '00150' -> 150, even -> deleted. camf '00449' -> 449, odd -> kept.
    d = _make_kradar_dict_item(rdr="00182", ldr64="00150", camf="00449")
    lidar_frame_deletion(d, params, rng)
    _check("KRADAR frame_del: lidar FIRES on even ldr64 index (150)",
           d["ldr64"] is None)

    d2 = _make_kradar_dict_item(rdr="00182", ldr64="00150", camf="00449")
    radar_frame_deletion(d2, params, rng)
    _check("KRADAR frame_del: radar fires on even rdr index (182)",
           d2["rdr_sparse"] is None)

    d3 = _make_kradar_dict_item(camf="00448")  # even -> camera blackout
    before = d3["front0"].clone()
    camera_frame_deletion(d3, params, rng)
    _check("KRADAR frame_del: camera FIRES on even camf index (448)",
           not torch.equal(d3["front0"], before))

    # Odd indices must be left alone -- proves the gate is real, not always-on.
    d4 = _make_kradar_dict_item(rdr="00183", ldr64="00151", camf="00449")
    cam_before = d4["front0"].clone()
    radar_frame_deletion(d4, params, rng)
    lidar_frame_deletion(d4, params, rng)
    camera_frame_deletion(d4, params, rng)
    _check("KRADAR frame_del: odd indices leave radar untouched",
           d4["rdr_sparse"] is not None)
    _check("KRADAR frame_del: odd indices leave lidar untouched",
           d4["ldr64"] is not None)
    _check("KRADAR frame_del: odd indices leave camera untouched",
           torch.equal(d4["front0"], cam_before))


def test_kradar_injector_end_to_end():
    """Through NoiseInjector, on real meta keys, across the 6 smoke-run frames.

    interval=2 over the actual ldr64 indices of the Alvis 6-frame scope
    (150,151,152,122,123,124) must delete exactly the 4 even ones.
    """
    config = EffectConfig(
        seed=42,
        lidar=[Effect("frame_deletion", p=1.0,
                      params={"mode": "deterministic", "interval": 2})],
    )
    injector = NoiseInjector(config)
    real_ldr_idx = ["00150", "00151", "00152", "00122", "00123", "00124"]
    deleted = []
    for i, ldr in enumerate(real_ldr_idx):
        d = injector(_make_kradar_dict_item(ldr64=ldr), frame_index=i)
        deleted.append(d["ldr64"] is None)
    _check("KRADAR injector: 4 of 6 smoke frames deleted",
           sum(deleted) == 4, f"deleted={deleted}")
    _check("KRADAR injector: exactly the even-index frames deleted",
           deleted == [True, False, True, True, False, True],
           f"got {deleted}")


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


def test_injector_should_skip_then_call_random_mode_consistency():
    """Regression for the RNG-desync bug (frame-deletion-verification-report.md §3.3).

    Reproduces the report's exact scenario: random-mode radar frame_deletion,
    p=0.2, seed=7, 30 frames, should_skip(i, "radar") called immediately before
    injector(d, frame_index=i) for every i (the design doc's documented
    "pre-check before loading" usage pattern). Before the fix, should_skip()
    and __call__() each drew independently from the same shared per-effect
    sub_rng, producing 5/30 mismatches in this exact configuration. After the
    fix (should_skip() decision cache, consumed by __call__()), there must be
    zero mismatches for every frame.
    """
    config = EffectConfig(
        seed=7,
        radar=[Effect("frame_deletion", p=1.0, params={"mode": "random", "p": 0.2})],
    )
    injector = NoiseInjector(config)

    mismatches = 0
    for i in range(30):
        predicted_skip = injector.should_skip(i, "radar")
        d = _make_dict_item(with_ldr64=False, with_camera=False, rdr_frame_idx=i)
        d = injector(d, frame_index=i)
        actually_deleted = d["rdr_sparse"] is None
        if predicted_skip != actually_deleted:
            mismatches += 1

    _check("INJ should_skip+call random-mode consistency: 0/30 mismatches",
           mismatches == 0, detail=f"got {mismatches}/30 mismatches")


def test_injector_should_skip_cache_does_not_leak_across_frame_indices():
    """should_skip()'s cached decision for frame N must not be reused for frame M."""
    config = EffectConfig(
        seed=3,
        radar=[Effect("frame_deletion", p=1.0,
                       params={"mode": "deterministic", "index_list": [5]})],
    )
    injector = NoiseInjector(config)

    _check("INJ skip-cache: should_skip(5) True",
           injector.should_skip(5, "radar") is True)
    _check("INJ skip-cache: should_skip(6) False (independent frame)",
           injector.should_skip(6, "radar") is False)

    d5 = _make_dict_item(with_ldr64=False, with_camera=False, rdr_frame_idx=5)
    d5 = injector(d5, frame_index=5)
    _check("INJ skip-cache: __call__ frame 5 deletes (matches cached decision)",
           d5["rdr_sparse"] is None)

    d6 = _make_dict_item(with_ldr64=False, with_camera=False, rdr_frame_idx=6)
    d6 = injector(d6, frame_index=6)
    _check("INJ skip-cache: __call__ frame 6 does not delete (matches cached decision)",
           d6["rdr_sparse"] is not None)


def test_injector_call_only_unaffected_by_skip_cache():
    """__call__() callers who never call should_skip() get unchanged behaviour.

    The skip cache is only ever populated by should_skip(); a caller that only
    ever uses __call__() must see identical draws/decisions with and without
    the fix in place (the fix must be a strict no-op on this path).
    """
    config = EffectConfig(
        seed=11,
        radar=[Effect("frame_deletion", p=1.0, params={"mode": "random", "p": 0.3})],
    )
    injector_a = NoiseInjector(config)
    injector_b = NoiseInjector(config)

    results_a, results_b = [], []
    for i in range(20):
        da = _make_dict_item(with_ldr64=False, with_camera=False, rdr_frame_idx=i)
        da = injector_a(da, frame_index=i)
        results_a.append(da["rdr_sparse"] is None)

        db = _make_dict_item(with_ldr64=False, with_camera=False, rdr_frame_idx=i)
        db = injector_b(db, frame_index=i)
        results_b.append(db["rdr_sparse"] is None)

    _check("INJ call-only path: two fresh injectors (same seed, never call should_skip) agree",
           results_a == results_b)
    _check("INJ call-only path: skip cache stays empty",
           len(injector_a._skip_cache) == 0)


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
    test_camera_loss_partial_cross_channel_independence()

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
    test_injector_should_skip_then_call_random_mode_consistency()
    test_injector_should_skip_cache_does_not_leak_across_frame_indices()
    test_injector_call_only_unaffected_by_skip_cache()
    test_injector_legacy_effect_rejected()
    test_injector_frame_deletion_index_list()
    test_injector_has_key_skips_none()

    # Real K-Radar meta['idx'] key naming (regression for the silent
    # frame_deletion no-op on lidar / camera)
    test_kradar_meta_idx_resolution()
    test_kradar_frame_deletion_fires_all_modalities()
    test_kradar_injector_end_to_end()

    success = _print_results()
    sys.exit(0 if success else 1)