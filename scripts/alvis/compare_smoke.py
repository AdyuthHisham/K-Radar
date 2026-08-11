#!/usr/bin/env python3
"""Compare clean vs. noisy ASF predictions from the Alvis smoke run.

AP on a 6-frame split (see asf_smoke.sbatch) is noise — the actual signal is
whether noisy predictions differ from clean, and by how much. This reads the
KITTI-format prediction files written under each condition's
outputs/alvis_smoke/<name>/exp_*/test_kitti/<epoch>/<conf_thr>/all/preds/ and
reports, per frame: box count and mean score for each condition, plus for
matched boxes (same class, nearest center) the mean center displacement of
each noisy condition vs. clean.

No GPU / model deps — stdlib + a directory of text files. Run on a login node.

Usage (from the K-Radar repo root, or point --outputs elsewhere):
    python scripts/alvis/compare_smoke.py \
        --outputs outputs/alvis_smoke \
        --conditions clean noisy_shape noisy_blackout
"""
from __future__ import annotations

import argparse
import glob
import math
import os
from collections import defaultdict


def _find_preds_dir(condition_root: str) -> str | None:
    matches = glob.glob(os.path.join(condition_root, "exp_*", "test_kitti", "*", "*", "all", "preds"))
    if not matches:
        return None
    # If multiple epochs/thresholds exist (re-run without cleanup), take the newest.
    return max(matches, key=os.path.getmtime)


def _parse_kitti_pred(path: str) -> list[dict]:
    """Parse one KITTI-format prediction file.

    Columns: class truncated occluded alpha bbox(4) dims(3) loc(3) rotation_y [score]
    """
    boxes = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 15:
                continue
            cls = parts[0]
            loc = tuple(float(x) for x in parts[11:14])  # x, y, z (camera frame)
            score = float(parts[15]) if len(parts) > 15 else None
            boxes.append({"cls": cls, "loc": loc, "score": score})
    return boxes


def _load_condition(root: str, name: str) -> dict[str, list[dict]] | None:
    condition_root = os.path.join(root, name)
    preds_dir = _find_preds_dir(condition_root)
    if preds_dir is None:
        print(f"[{name}] no preds dir found under {condition_root} — job likely did not complete")
        return None
    frames = {}
    for path in sorted(glob.glob(os.path.join(preds_dir, "*.txt"))):
        frame_id = os.path.splitext(os.path.basename(path))[0]
        frames[frame_id] = _parse_kitti_pred(path)
    print(f"[{name}] {len(frames)} frame(s) in {preds_dir}")
    return frames


def _match_and_displace(clean_boxes: list[dict], other_boxes: list[dict]) -> list[float]:
    """Greedy nearest-center matching within class; returns per-match displacement (m)."""
    remaining = list(other_boxes)
    displacements = []
    for cb in clean_boxes:
        candidates = [b for b in remaining if b["cls"] == cb["cls"]]
        if not candidates:
            continue
        best = min(
            candidates,
            key=lambda b: math.dist(cb["loc"], b["loc"]),
        )
        displacements.append(math.dist(cb["loc"], best["loc"]))
        remaining.remove(best)
    return displacements


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", default="outputs/alvis_smoke")
    ap.add_argument("--baseline", default="clean")
    ap.add_argument("--conditions", nargs="+", default=["clean", "noisy_shape", "noisy_blackout"])
    args = ap.parse_args()

    data = {name: _load_condition(args.outputs, name) for name in args.conditions}
    baseline = data.get(args.baseline)
    if baseline is None:
        print(f"\nBaseline condition '{args.baseline}' has no predictions — nothing to compare against.")
        return

    print("\n=== per-condition per-frame summary ===")
    for name, frames in data.items():
        if frames is None:
            continue
        for frame_id, boxes in sorted(frames.items()):
            scores = [b["score"] for b in boxes if b["score"] is not None]
            mean_score = sum(scores) / len(scores) if scores else float("nan")
            print(f"  [{name:16s}] frame {frame_id}: {len(boxes)} box(es), mean score {mean_score:.3f}")

    print(f"\n=== displacement vs. baseline '{args.baseline}' ===")
    for name, frames in data.items():
        if name == args.baseline or frames is None:
            continue
        all_disp = []
        for frame_id, clean_boxes in sorted(baseline.items()):
            other_boxes = frames.get(frame_id, [])
            disp = _match_and_displace(clean_boxes, other_boxes)
            all_disp.extend(disp)
            n_clean, n_other = len(clean_boxes), len(other_boxes)
            disp_str = f"{sum(disp) / len(disp):.3f} m" if disp else "n/a (no matches)"
            print(
                f"  [{name:16s}] frame {frame_id}: clean={n_clean} boxes, "
                f"this={n_other} boxes, mean matched displacement={disp_str}"
            )
        if all_disp:
            print(
                f"  [{name:16s}] OVERALL mean displacement over {len(all_disp)} matched box(es): "
                f"{sum(all_disp) / len(all_disp):.3f} m"
            )
            if sum(all_disp) / len(all_disp) < 1e-6:
                print(
                    f"  [{name:16s}] WARNING: zero displacement — noise injection may not have "
                    f"fired. Check the job log for the '[noise-injection] Loaded config' banner."
                )
        else:
            print(f"  [{name:16s}] no matched boxes across any frame")


if __name__ == "__main__":
    main()
