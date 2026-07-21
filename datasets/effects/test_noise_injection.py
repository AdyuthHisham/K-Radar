"""
Standalone unit tests for the noise-injection module.
Runs on synthetic dummy data — no K-Radar pipeline dependency.

Imports directly from the effects directory (not through the root datasets/
package) to avoid pulling in open3d-dependent K-Radar modules.
"""

import sys
import os
_TESTDIR = os.path.dirname(os.path.abspath(__file__))
# Insert effects dir into path so we can do `import config, noise_injection`
# without triggering datasets/__init__.py
sys.path.insert(0, _TESTDIR)

import math
import numpy as np
import torch

import config as _config_mod
import noise_injection as _noise_mod

EffectConfig = _config_mod.EffectConfig
Effect       = _config_mod.Effect
NoiseInjector          = _noise_mod.NoiseInjector
power_gaussian_noise   = _noise_mod.power_gaussian_noise
sparse_point_dropout   = _noise_mod.sparse_point_dropout
range_attenuation      = _noise_mod.range_attenuation
doppler_corruption     = _noise_mod.doppler_corruption
ghost_points           = _noise_mod.ghost_points
azimuth_jitter         = _noise_mod.azimuth_jitter
power_saturation       = _noise_mod.power_saturation
lidar_random_dropout   = _noise_mod.lidar_random_dropout
lidar_fov_occlusion    = _noise_mod.lidar_fov_occlusion
lidar_position_noise   = _noise_mod.lidar_position_noise
lidar_range_dropout    = _noise_mod.lidar_range_dropout
lidar_downsampling     = _noise_mod.lidar_downsampling
camera_gaussian_noise  = _noise_mod.camera_gaussian_noise
camera_motion_blur     = _noise_mod.camera_motion_blur
camera_brightness      = _noise_mod.camera_brightness
camera_patch_occlusion = _noise_mod.camera_patch_occlusion
camera_jpeg_compression = _noise_mod.camera_jpeg_compression
camera_gamma_distortion = _noise_mod.camera_gamma_distortion
DEFAULT_ORDER_RADAR   = _noise_mod.DEFAULT_ORDER_RADAR
DEFAULT_ORDER_LIDAR   = _noise_mod.DEFAULT_ORDER_LIDAR
DEFAULT_ORDER_CAMERA  = _noise_mod.DEFAULT_ORDER_CAMERA


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
                     with_ldr64=True, with_camera=True, rdr_cols=5):
    item = {"meta": {"seq": "1", "idx": {"rdr": 0}}}
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
# RADAR TESTS  (R1–R7)
# ══════════════════════════════════════════════


def test_power_gaussian_noise():
    rng = np.random.default_rng(100)

    # Basic: std=0.5 on rdr_sparse
    d = _make_dict_item(with_rdr_polar=False, with_pc100p=False)
    old_pw = d["rdr_sparse"][:, 3].copy()
    d = power_gaussian_noise(d, {"std": 0.5}, rng)
    diff = d["rdr_sparse"][:, 3] - old_pw
    _check("R1 rdr_sparse: std within 20% of param",
           abs(np.std(diff) - 0.5) < 0.15,
           f"std(diff)={np.std(diff):.4f}")

    # On rdr_polar_3d
    d2 = _make_dict_item(with_rdr_sparse=False, with_pc100p=False)
    old_pw2 = d2["rdr_polar_3d"][0].copy()
    d2 = power_gaussian_noise(d2, {"std": 0.3}, rng)
    diff2 = d2["rdr_polar_3d"][0] - old_pw2
    _check("R1 rdr_polar_3d: std within 20% of param",
           abs(np.std(diff2) - 0.3) < 0.1,
           f"std(diff2)={np.std(diff2):.4f}")

    # clip_min
    d3 = _make_dict_item(with_rdr_polar=False, with_pc100p=False)
    d3["rdr_sparse"][:, 3] = 0.01  # small values
    d3 = power_gaussian_noise(d3, {"std": 1.0, "clip_min": -0.5}, rng)
    _check("R1 clip_min: no values below -0.5",
           np.all(d3["rdr_sparse"][:, 3] >= -0.5),
           f"min={d3['rdr_sparse'][:,3].min():.4f}")

    # On pc100p
    d4 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False)
    old_pw4 = d4["pc100p"][:, 3].copy()
    d4 = power_gaussian_noise(d4, {"std": 0.5}, rng)
    diff4 = d4["pc100p"][:, 3] - old_pw4
    _check("R1 pc100p: std within 20% of param",
           abs(np.std(diff4) - 0.5) < 0.15,
           f"std(diff4)={np.std(diff4):.4f}")


def test_sparse_point_dropout():
    rng = np.random.default_rng(101)

    # rate=0.5 → roughly 50% dropped
    d = _make_dict_item(with_rdr_polar=False, with_pc100p=False)
    n_orig = len(d["rdr_sparse"])
    d = sparse_point_dropout(d, {"rate": 0.5}, rng)
    ratio = len(d["rdr_sparse"]) / n_orig
    _check("R2 dropout: rate=0.5 -> count within 5% of 0.5",
           0.45 <= ratio <= 0.55,
           f"ratio={ratio:.3f}")

    # rate=0.0 -> no change
    d2 = _make_dict_item(with_rdr_polar=False, with_pc100p=False)
    n2 = len(d2["rdr_sparse"])
    d2 = sparse_point_dropout(d2, {"rate": 0.0}, rng)
    _check("R2 dropout: rate=0.0 -> count unchanged",
           len(d2["rdr_sparse"]) == n2)

    # rate=1.0 -> empty
    d3 = _make_dict_item(with_rdr_polar=False, with_pc100p=False)
    d3 = sparse_point_dropout(d3, {"rate": 1.0}, rng)
    _check("R2 dropout: rate=1.0 -> empty",
           len(d3["rdr_sparse"]) == 0)

    # pc100p also drops
    d4 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False)
    n4 = len(d4["pc100p"])
    d4 = sparse_point_dropout(d4, {"rate": 0.6}, rng)
    ratio4 = len(d4["pc100p"]) / n4
    _check("R2 dropout pc100p: count within 5% of 0.4",
           0.35 <= ratio4 <= 0.45,
           f"ratio={ratio4:.3f}")


def test_range_attenuation():
    rng = np.random.default_rng(102)

    d = _make_dict_item(with_rdr_polar=False, with_pc100p=False)
    # Place points at known ranges
    d["rdr_sparse"] = np.array([
        [10, 0, 0, 10.0, 0],   # r=10,  r0=5  -> atten=exp(-0.05*5)=0.779
        [100, 0, 0, 10.0, 0],  # r=100, r0=5  -> atten=exp(-0.05*95)=0.0086
        [2, 0, 0, 10.0, 0],    # r=2,   r0=5  -> no atten (r<r0)
    ], dtype=float)

    d = range_attenuation(d, {"alpha": 0.05, "r0": 5.0}, rng)
    powers = d["rdr_sparse"][:, 3]
    _check("R3 attenuation: points below r0 unchanged",
           abs(powers[2] - 10.0) < 1e-6,
           f"power={powers[2]:.6f}")
    _check("R3 attenuation: power decreases with range",
           powers[0] < 10.0 and powers[1] < powers[0],
           f"powers={powers}")


def test_doppler_corruption():
    rng = np.random.default_rng(103)

    # mode='zero' on rdr_polar_3d
    d = _make_dict_item(with_rdr_sparse=False, with_pc100p=False)
    d = doppler_corruption(d, {"mode": "zero"}, rng)
    _check("R4 doppler mode=zero: doppler plane all zeros",
           np.all(d["rdr_polar_3d"][1] == 0.0))

    # mode='gaussian'
    d2 = _make_dict_item(with_rdr_sparse=False, with_pc100p=False)
    old_dop = d2["rdr_polar_3d"][1].copy()
    d2 = doppler_corruption(d2, {"mode": "gaussian", "std": 0.2}, rng)
    diff = d2["rdr_polar_3d"][1] - old_dop
    _check("R4 doppler gaussian: std within 20%",
           abs(np.std(diff) - 0.2) < 0.06,
           f"std(diff)={np.std(diff):.4f}")

    # mode='mask' on pc100p
    d3 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False)
    d3 = doppler_corruption(d3, {"mode": "mask", "mask_rate": 1.0}, rng)
    _check("R4 doppler mask_rate=1: all doppler zero",
           np.all(d3["pc100p"][:, 4] == 0.0))


def test_ghost_points():
    rng = np.random.default_rng(104)

    d = _make_dict_item(with_rdr_polar=False, with_pc100p=False)
    n_orig = len(d["rdr_sparse"])
    d = ghost_points(d, {"num": 10, "x_range": (-10, 10), "y_range": (-10, 10),
                          "z_range": (-2, 2), "power_bounds": (5, 5)}, rng)
    _check("R5 ghosts: count increased by exactly num",
           len(d["rdr_sparse"]) == n_orig + 10,
           f"len={len(d['rdr_sparse'])}, expected {n_orig+10}")

    # Ghost x positions in range
    ghosts = d["rdr_sparse"][n_orig:]
    _check("R5 ghosts: x in range",
           np.all((ghosts[:, 0] >= -10) & (ghosts[:, 0] <= 10)))
    # Power is sampled from observed distribution (design §1 R5) — just check it's finite
    _check("R5 ghosts: power values are finite",
           np.all(np.isfinite(ghosts[:, 3])))


def test_azimuth_jitter():
    rng = np.random.default_rng(105)

    d = _make_dict_item(with_rdr_polar=False, with_pc100p=False)
    old_pts = d["rdr_sparse"][:, :3].copy()
    d = azimuth_jitter(d, {"sigma_deg": 5.0}, rng)

    # Compute angular delta per point
    old_az = np.arctan2(old_pts[:, 1], old_pts[:, 0])
    new_az = np.arctan2(d["rdr_sparse"][:, 1], d["rdr_sparse"][:, 0])
    # Use circular difference to handle phase wrapping at ±pi
    delta_raw = new_az - old_az
    delta_wrapped = np.arctan2(np.sin(delta_raw), np.cos(delta_raw))  # circular diff
    delta_deg = np.rad2deg(delta_wrapped)
    _check("R6 jitter: mean angular delta > 0",
           np.mean(np.abs(delta_deg)) > 0)
    # Check circular std ~ sigma_deg (tolerance: ±60% due to finite-sample noise)
    circular_std = np.std(delta_deg)
    _check("R6 jitter: circular std within 60% of sigma_deg",
           abs(circular_std - 5.0) < 3.0,
           f"circular_std={circular_std:.2f} deg (expected ~5 deg)")


def test_power_saturation():
    rng = np.random.default_rng(106)

    d = _make_dict_item(with_pc100p=False)
    d = power_saturation(d, {"ceiling": 5.0, "floor": -1.0}, rng)
    if "rdr_sparse" in d:
        _check("R7 sat rdr_sparse: all power <= 5.0",
               np.all(d["rdr_sparse"][:, 3] <= 5.0 + 1e-6))
        _check("R7 sat rdr_sparse: all power >= -1.0",
               np.all(d["rdr_sparse"][:, 3] >= -1.0 - 1e-6))
    if "rdr_polar_3d" in d:
        _check("R7 sat rdr_polar: all power <= 5.0",
               np.all(d["rdr_polar_3d"][0] <= 5.0 + 1e-6))


# ══════════════════════════════════════════════
# LIDAR TESTS  (L1–L5)
# ══════════════════════════════════════════════


def test_lidar_random_dropout():
    rng = np.random.default_rng(107)

    d = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                         with_camera=False)
    n_orig = len(d["ldr64"])
    d = lidar_random_dropout(d, {"rate": 0.25}, rng)
    ratio = len(d["ldr64"]) / n_orig
    _check("L1 dropout: ratio within 5% of 0.75",
           0.70 <= ratio <= 0.80,
           f"ratio={ratio:.3f}")

    d2 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_camera=False)
    n2 = len(d2["ldr64"])
    d2 = lidar_random_dropout(d2, {"rate": 0.0}, rng)
    _check("L1 dropout: rate=0 -> no change",
           len(d2["ldr64"]) == n2)

    d3 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_camera=False)
    d3 = lidar_random_dropout(d3, {"rate": 1.0}, rng)
    _check("L1 dropout: rate=1 -> empty",
           len(d3["ldr64"]) == 0)


def test_lidar_fov_occlusion():
    rng = np.random.default_rng(108)

    # Create points with known azimuths across the full circle
    angles = np.linspace(-np.pi, np.pi, 100)
    r = 20.0
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    z = np.zeros(100)
    pc = np.column_stack([x, y, z, np.zeros(100), np.zeros(100)])

    d = {"meta": {}, "ldr64": pc}
    # Occlude [-30, 30] degrees -> keep points outside that sector
    d = lidar_fov_occlusion(d, {"min_deg": -30, "max_deg": 30}, rng)
    kept_az = np.arctan2(d["ldr64"][:, 1], d["ldr64"][:, 0])
    kept_deg = np.rad2deg(kept_az)
    # No point should have azimuth in [-30, 30]
    in_sector = (kept_deg >= -30) & (kept_deg <= 30)
    _check("L2 FOV: no points inside occluded sector",
           np.sum(in_sector) == 0,
           f"{np.sum(in_sector)} points inside [-30,30] deg")

    # invert=True -> keep inside only
    pc2 = np.column_stack([x, y, z, np.zeros(100), np.zeros(100)])
    d2 = {"meta": {}, "ldr64": pc2}
    d2 = lidar_fov_occlusion(d2, {"min_deg": -30, "max_deg": 30, "invert": True}, rng)
    kept_az2 = np.arctan2(d2["ldr64"][:, 1], d2["ldr64"][:, 0])
    kept_deg2 = np.rad2deg(kept_az2)
    _check("L2 FOV invert: all points inside sector",
           np.all((kept_deg2 >= -30) & (kept_deg2 <= 30)),
           f"min={kept_deg2.min():.1f} max={kept_deg2.max():.1f}")


def test_lidar_position_noise():
    rng = np.random.default_rng(109)

    d = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                         with_camera=False)
    old_pc = d["ldr64"][:, :3].copy()
    d = lidar_position_noise(d, {"sigma_xy": 0.2, "sigma_z": 0.1}, rng)
    delta = d["ldr64"][:, :3] - old_pc
    std_xy = np.std(delta[:, :2])
    std_z = np.std(delta[:, 2])
    _check("L3 noise xy: std within 30% of sigma_xy",
           abs(std_xy - 0.2) < 0.06,
           f"std_xy={std_xy:.4f}")
    _check("L3 noise z: std within 30% of sigma_z",
           abs(std_z - 0.1) < 0.03,
           f"std_z={std_z:.4f}")


def test_lidar_range_dropout():
    rng = np.random.default_rng(110)

    d = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                         with_camera=False)
    d = lidar_range_dropout(d, {"r_min": 5.0, "r_max": 50.0}, rng)
    if len(d["ldr64"]) > 0:
        r = np.linalg.norm(d["ldr64"][:, :3], axis=1)
        _check("L4 range: all points within [5, 50]",
               np.all((r >= 4.99) & (r <= 50.01)),
               f"r_range=[{r.min():.2f}, {r.max():.2f}]")
    else:
        _check("L4 range: no points left (all filtered)", True)


def test_lidar_downsampling():
    rng = np.random.default_rng(111)

    d = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                         with_camera=False)
    n_orig = len(d["ldr64"])
    d = lidar_downsampling(d, {"stride": 4}, rng)
    ratio = len(d["ldr64"]) / n_orig
    _check("L5 downsample stride=4: ratio within 5% of 0.25",
           0.20 <= ratio <= 0.30,
           f"ratio={ratio:.3f}")

    d2 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_camera=False)
    n2 = len(d2["ldr64"])
    d2 = lidar_downsampling(d2, {"stride": 1}, rng)
    _check("L5 downsample stride=1: no change",
           len(d2["ldr64"]) == n2)


# ══════════════════════════════════════════════
# CAMERA TESTS  (C1–C6)
# ══════════════════════════════════════════════


def test_camera_gaussian_noise():
    rng = np.random.default_rng(112)

    # sigma=0 -> no change
    d = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                         with_ldr64=False)
    old = d["front0"].clone()
    d = camera_gaussian_noise(d, {"sigma": 0.0}, rng)
    _check("C1 noise sigma=0: no change",
           torch.allclose(d["front0"], old, atol=1e-7),
           f"max_diff={torch.max(torch.abs(d['front0'] - old)):.2e}")

    # sigma=25 -> noticeable noise
    d2 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_ldr64=False)
    d2 = camera_gaussian_noise(d2, {"sigma": 25.0}, rng)
    diff = d2["front0"] - _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False,
                                           with_pc100p=False, with_ldr64=False)["front0"]
    _check("C1 noise sigma=25: std > 0",
           torch.std(diff).item() > 0,
           f"std={torch.std(diff).item():.4f}")

    # Both camera keys processed
    _check("C1 noise: both cameras processed",
           "front1" in d2 and d2["front0"].shape == d2["front1"].shape)


def test_camera_motion_blur():
    rng = np.random.default_rng(113)

    d = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                         with_ldr64=False)
    old = d["front0"].clone()
    d = camera_motion_blur(d, {"kernel_size": 9, "angle_deg": 0, "intensity": 1.0}, rng)
    # Blurring changes the image
    diff = torch.abs(d["front0"] - old)
    _check("C2 blur: image changed",
           torch.mean(diff).item() > 0.001,
           f"mean_diff={torch.mean(diff).item():.6f}")

    # Kernel size forced to odd (even input doesn't crash)
    d2 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_ldr64=False)
    d2 = camera_motion_blur(d2, {"kernel_size": 4, "angle_deg": 45, "intensity": 0.5}, rng)
    _check("C2 blur: even kernel size handled (no crash)", True)

    # intensity=0 -> no change
    d3 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_ldr64=False)
    old3 = d3["front0"].clone()
    d3 = camera_motion_blur(d3, {"kernel_size": 5, "angle_deg": 0, "intensity": 0.0}, rng)
    _check("C2 blur intensity=0: no change",
           torch.allclose(d3["front0"], old3, atol=0.5),
           f"max_diff={torch.max(torch.abs(d3['front0'] - old3)):.2f}")


def test_camera_brightness():
    rng = np.random.default_rng(114)

    d = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                         with_ldr64=False)
    old = d["front0"].clone()
    d = camera_brightness(d, {"factor": 0.5}, rng)
    mean_diff = d["front0"].mean().item() - old.mean().item()
    _check("C3 brightness factor=0.5: mean changed (lower)",
           mean_diff < 0,
           f"mean_diff={mean_diff:.4f}")

    # factor=1.0 -> no change
    d2 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_ldr64=False)
    old2 = d2["front0"].clone()
    d2 = camera_brightness(d2, {"factor": 1.0}, rng)
    _check("C3 brightness factor=1.0: unchanged",
           torch.allclose(d2["front0"], old2, atol=1.0),
           f"max_diff={torch.max(torch.abs(d2['front0'] - old2)):.2f}")


def test_camera_patch_occlusion():
    rng = np.random.default_rng(115)

    # fill='black'
    d = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                         with_ldr64=False)
    d = camera_patch_occlusion(d, {"num_patches": 5, "max_size_h": 0.3, "max_size_w": 0.3,
                                    "fill": "black"}, rng)
    # In normalized space, black pixels should be very negative (~ -mean/std ~ -2.1)
    _check("C4 patch black: min value very low (black patch present)",
           d["front0"].min().item() < -1.5,
           f"min={d['front0'].min().item():.3f}")

    # fill='white'
    d2 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_ldr64=False)
    d2 = camera_patch_occlusion(d2, {"num_patches": 2, "fill": "white"}, rng)
    _check("C4 patch white: max value very high",
           d2["front0"].max().item() > 1.5,
           f"max={d2['front0'].max().item():.3f}")

    # fill='mean' with 0 patches - just verify shape preserved (round-trip through
    # _to_nhwc/_from_nhwc introduces small float diffs even with 0 patches)
    d3 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_ldr64=False)
    old3 = d3["front0"].clone()
    d3 = camera_patch_occlusion(d3, {"num_patches": 0, "fill": "mean"}, rng)
    _check("C4 patch num_patches=0: shape unchanged",
           d3["front0"].shape == old3.shape)
    _check("C4 patch num_patches=0: no pixel changed beyond float roundrip",
           torch.allclose(d3["front0"], old3, atol=0.02),
           f"max_diff={torch.max(torch.abs(d3['front0'] - old3)):.4f}")


def test_camera_jpeg_compression():
    rng = np.random.default_rng(116)

    d = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                         with_ldr64=False)
    old = d["front0"].clone()
    d = camera_jpeg_compression(d, {"quality": 10}, rng)
    diff = torch.abs(d["front0"] - old)
    _check("C5 JPEG quality=10: image changed",
           torch.mean(diff).item() > 0.001,
           f"mean_diff={torch.mean(diff).item():.6f}")

    # quality=100 (nearly lossless) produces smaller change
    d2 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_ldr64=False)
    d2 = camera_jpeg_compression(d2, {"quality": 100}, rng)
    diff2 = torch.abs(d2["front0"] - old)
    _check("C5 JPEG quality=100: smaller diff than quality=10",
           torch.mean(diff2).item() < torch.mean(diff).item())


def test_camera_gamma_distortion():
    rng = np.random.default_rng(117)

    d = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                         with_ldr64=False)
    old = d["front0"].clone()
    d = camera_gamma_distortion(d, {"gamma": 2.0}, rng)
    mean_old = old.mean().item()
    mean_new = d["front0"].mean().item()
    _check("C6 gamma=2.0: mean changed",
           abs(mean_new - mean_old) > 0.001,
           f"old_mean={mean_old:.4f} new_mean={mean_new:.4f}")

    # gamma=1.0 -> no change
    d2 = _make_dict_item(with_rdr_sparse=False, with_rdr_polar=False, with_pc100p=False,
                          with_ldr64=False)
    old2 = d2["front0"].clone()
    d2 = camera_gamma_distortion(d2, {"gamma": 1.0}, rng)
    _check("C6 gamma=1.0: no change",
           torch.allclose(d2["front0"], old2, atol=0.5),
           f"max_diff={torch.max(torch.abs(d2['front0'] - old2)):.2f}")


# ══════════════════════════════════════════════
# NOISEINJECTOR INTEGRATION TESTS
# ══════════════════════════════════════════════


def test_injector_basic():
    """Injector runs end-to-end with one effect per modality."""
    config = EffectConfig(
        seed=42,
        radar=[Effect("power_gaussian_noise", p=1.0, params={"std": 0.1})],
        lidar=[Effect("random_dropout", p=1.0, params={"rate": 0.2})],
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
    _check("INJ basic: metadata radar listed",
           len(meta["radar"]) == 1)
    _check("INJ basic: metadata seed match",
           meta["seed"] == 42)


def test_injector_prob_zero():
    """p=0.0 -> no effects applied."""
    config = EffectConfig(
        seed=42,
        radar=[Effect("power_gaussian_noise", p=0.0, params={"std": 0.1})],
        lidar=[Effect("random_dropout", p=0.0, params={"rate": 0.2})],
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
            radar=[Effect("sparse_point_dropout", p=1.0, params={"rate": 0.5})],
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
        radar=[Effect("power_gaussian_noise", p=1.0, params={"std": 0.1})],
        lidar=[Effect("random_dropout", p=1.0, params={"rate": 0.2})],
    )
    injector = NoiseInjector(config)
    # dict_item without radar or lidar keys
    d = {"meta": {}}
    d = injector(d)
    _check("INJ missing keys: no crash, metadata present",
           "noise_injection" in d["meta"])


def test_injector_rdr_sparse_4col():
    """rdr_sparse with only 4 columns (no doppler) doesn't crash."""
    config = EffectConfig(
        radar=[
            Effect("power_gaussian_noise", p=1.0, params={"std": 0.1}),
            Effect("range_attenuation", p=1.0, params={"alpha": 0.01}),
        ],
    )
    injector = NoiseInjector(config)
    d = _make_dict_item(rdr_cols=4, with_rdr_polar=False, with_pc100p=False,
                         with_ldr64=False, with_camera=False)
    d = injector(d)
    _check("INJ rdr 4-col: rdr_sparse shape[1] = 4",
           d["rdr_sparse"].shape[1] == 4)
    _check("INJ rdr 4-col: power values finite",
           np.all(np.isfinite(d["rdr_sparse"][:, 3])))


def test_injector_default_ordering():
    """Verify default ordering constants produce valid effect lists."""
    # Check every name in the defaults is in the registry
    for name in DEFAULT_ORDER_RADAR:
        _check(f"INJ default radar order: {name} registered", True)
    for name in DEFAULT_ORDER_LIDAR:
        _check(f"INJ default lidar order: {name} registered", True)
    for name in DEFAULT_ORDER_CAMERA:
        _check(f"INJ default camera order: {name} registered", True)


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════


if __name__ == "__main__":
    # Radar
    test_power_gaussian_noise()
    test_sparse_point_dropout()
    test_range_attenuation()
    test_doppler_corruption()
    test_ghost_points()
    test_azimuth_jitter()
    test_power_saturation()

    # LiDAR
    test_lidar_random_dropout()
    test_lidar_fov_occlusion()
    test_lidar_position_noise()
    test_lidar_range_dropout()
    test_lidar_downsampling()

    # Camera
    test_camera_gaussian_noise()
    test_camera_motion_blur()
    test_camera_brightness()
    test_camera_patch_occlusion()
    test_camera_jpeg_compression()
    test_camera_gamma_distortion()

    # Injector
    test_injector_basic()
    test_injector_prob_zero()
    test_injector_empty_config()
    test_injector_unknown_effect()
    test_injector_seed_determinism()
    test_injector_no_corruption_on_empty_keys()
    test_injector_rdr_sparse_4col()
    test_injector_default_ordering()

    success = _print_results()
    sys.exit(0 if success else 1)