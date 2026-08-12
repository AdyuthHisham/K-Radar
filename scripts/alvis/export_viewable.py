#!/usr/bin/env python3
"""Turn the per-condition sensor dumps into a viewable bundle.

The dumps store point clouds as .npy, which no point-cloud viewer opens
directly, and they duplicate the clean copy inside every condition (~1.2 GB
total). This writes a deduplicated bundle:

    viewable/
      clean/                 ldr64/rdr_sparse as .pcd + front0 .png (once)
      <condition>/           the corrupted sensor(s) for that condition
      README.txt

Point clouds become binary PCD v0.7 (x y z + intensity/power), which
CloudCompare, Open3D, PCL and Meshlab all read. Written by hand rather than via
open3d so this runs anywhere, in or out of the container.

Only the modality each condition actually corrupts is exported (plus the camera
PNGs, which are small) -- exporting all three for all conditions would be mostly
identical copies of the clean data. Pass --all to override.

Usage:
    python scripts/alvis/export_viewable.py --outputs outputs/alvis_smoke
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

import numpy as np

# Which modality each condition corrupts, inferred from the condition name.
_MODALITY_PREFIX = {"radar": "rdr_sparse", "lidar": "ldr64", "camera": "front0"}

# Column meaning: both arrays carry xyz in 0..2; col 3 is power (radar) or
# intensity (lidar). Verified against the dumps -- ldr64 is (N, 9), rdr_sparse
# is (N, 4).
_FIELD4 = {"rdr_sparse": "power", "ldr64": "intensity"}


def write_pcd(path: str, xyz_i: np.ndarray, field4: str = "intensity") -> None:
    """Write an (N, 4) float array as a binary PCD v0.7 file."""
    pts = np.ascontiguousarray(xyz_i.astype(np.float32))
    n = len(pts)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS x y z {field4}\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(pts.tobytes())


def _npy_to_pcd(src: str, dst: str, key: str) -> str | None:
    arr = np.load(src)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    if arr.shape[1] >= 4:
        out = arr[:, [0, 1, 2, 3]]
    else:  # xyz only -- pad the 4th channel
        out = np.column_stack([arr[:, :3], np.zeros(len(arr), dtype=arr.dtype)])
    write_pcd(dst, out, _FIELD4.get(key, "intensity"))
    return f"{len(arr)} pts"


def export_condition(dump_dir: str, out_dir: str, keys: list[str], clean: bool) -> int:
    os.makedirs(out_dir, exist_ok=True)
    suffix = "_clean" if clean else ""
    written = 0
    for key in keys:
        if key == "front0":
            for src in sorted(glob.glob(os.path.join(dump_dir, f"*_front0{suffix}.png"))):
                stem = os.path.basename(src).replace(f"_front0{suffix}.png", "")
                shutil.copy2(src, os.path.join(out_dir, f"{stem}_front0.png"))
                written += 1
        else:
            for src in sorted(glob.glob(os.path.join(dump_dir, f"*_{key}{suffix}.npy"))):
                stem = os.path.basename(src).replace(f"_{key}{suffix}.npy", "")
                dst = os.path.join(out_dir, f"{stem}_{key}.pcd")
                if _npy_to_pcd(src, dst, key):
                    written += 1
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", default="outputs/alvis_smoke")
    ap.add_argument("--dest", default=None, help="default: <outputs>/viewable")
    ap.add_argument("--all", action="store_true",
                    help="export all three modalities for every condition")
    args = ap.parse_args()

    dest = args.dest or os.path.join(args.outputs, "viewable")
    os.makedirs(dest, exist_ok=True)

    conditions = sorted(
        os.path.basename(os.path.dirname(p))
        for p in glob.glob(os.path.join(args.outputs, "*", "sensor_dump"))
    )
    if not conditions:
        print(f"No sensor_dump directories under {args.outputs}")
        return

    # Clean reference, taken once -- the *_clean copies are byte-identical
    # across conditions. Prefer a condition with a full 6-frame dump.
    ref = max(conditions,
              key=lambda c: len(glob.glob(os.path.join(args.outputs, c, "sensor_dump", "*_meta.txt"))))
    ref_dump = os.path.join(args.outputs, ref, "sensor_dump")
    n = export_condition(ref_dump, os.path.join(dest, "clean"),
                         ["rdr_sparse", "ldr64", "front0"], clean=True)
    print(f"clean/                        {n:3d} file(s)   [reference: {ref}]")

    for cond in conditions:
        dump = os.path.join(args.outputs, cond, "sensor_dump")
        if args.all:
            keys = ["rdr_sparse", "ldr64", "front0"]
        else:
            modality = cond.split("_")[0]
            keys = [_MODALITY_PREFIX.get(modality, "ldr64")]
            if "front0" not in keys:
                keys.append("front0")
        n = export_condition(dump, os.path.join(dest, cond), keys, clean=False)
        print(f"{cond:30s}{n:3d} file(s)")

    with open(os.path.join(dest, "README.txt"), "w") as f:
        f.write(
            "ASF corruption study -- viewable sensor data\n"
            "===========================================\n\n"
            "clean/      the uncorrupted frames (identical for every condition)\n"
            "<cond>/     the corrupted sensor for that condition\n\n"
            "Files are named <frameindex>_seq<N>_<sensor>.<ext>, so the same\n"
            "stem in clean/ and <cond>/ is the same frame before and after.\n\n"
            ".pcd  binary PCD v0.7, fields: x y z intensity|power.\n"
            "      Open in CloudCompare / Meshlab / PCL, or:\n"
            "          import open3d as o3d\n"
            "          o3d.visualization.draw_geometries([o3d.io.read_point_cloud('f.pcd')])\n"
            "      or with numpy, skipping the 11-line ASCII header:\n"
            "          a = np.fromfile('f.pcd', dtype=np.float32,\n"
            "                          offset=open('f.pcd','rb').read().index(b'DATA binary\\n')+12)\n"
            "          a = a.reshape(-1, 4)\n\n"
            ".png  camera frames, already denormalized to viewable 8-bit RGB.\n\n"
            "Note: in the *_zeros conditions every point is at the origin, so the\n"
            "cloud renders as a single blob at (0,0,0) -- that is the corruption,\n"
            "not a broken export.\n"
        )
    print(f"\nBundle at {dest}")


if __name__ == "__main__":
    main()
