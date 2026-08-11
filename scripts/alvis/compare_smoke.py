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

    The pipeline writes a single placeholder row ``dummy -1 -1 0 0 ...`` for a
    frame with no detections above the confidence threshold, so the KITTI
    evaluator still has a file to read. Those rows are not detections and are
    dropped here — counting them as boxes would report an empty frame as
    "1 box, score 0.000".
    """
    boxes = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 15:
                continue
            cls = parts[0]
            if cls == "dummy":
                continue
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


def _match_and_displace(
    clean_boxes: list[dict], other_boxes: list[dict], match_radius: float
) -> tuple[list[float], int]:
    """Greedy nearest-center matching within class, gated by a distance cutoff.

    Without a cutoff, a heavily corrupted run pairs boxes that have nothing to
    do with each other and reports the distance between unrelated objects as
    "displacement". Candidates beyond ``match_radius`` metres are treated as
    unmatched instead -- i.e. the clean detection was lost, not moved.

    Returns (displacements, n_unmatched_clean).
    """
    remaining = list(other_boxes)
    displacements = []
    unmatched = 0
    for cb in clean_boxes:
        candidates = [b for b in remaining if b["cls"] == cb["cls"]]
        if not candidates:
            unmatched += 1
            continue
        best = min(candidates, key=lambda b: math.dist(cb["loc"], b["loc"]))
        dist = math.dist(cb["loc"], best["loc"])
        if dist > match_radius:
            unmatched += 1
            continue
        displacements.append(dist)
        remaining.remove(best)
    return displacements, unmatched


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", default="outputs/alvis_smoke")
    ap.add_argument("--baseline", default="clean")
    ap.add_argument("--conditions", nargs="+", default=["clean", "noisy_shape", "noisy_blackout"])
    ap.add_argument("--match-radius", type=float, default=5.0,
                    help="Max centre distance (m) for a noisy box to count as the same "
                         "detection as a clean one; beyond this the clean box is counted "
                         "as lost rather than displaced.")
    ap.add_argument("--csv", default=None, help="Optional path to write the summary table as CSV.")
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
        total = 0
        for frame_id, boxes in sorted(frames.items()):
            total += len(boxes)
            scores = [b["score"] for b in boxes if b["score"] is not None]
            if not boxes:
                print(f"  [{name:16s}] frame {frame_id}: NO DETECTIONS")
                continue
            mean_score = sum(scores) / len(scores) if scores else float("nan")
            by_cls = defaultdict(int)
            for b in boxes:
                by_cls[b["cls"]] += 1
            cls_str = ", ".join(f"{c}x{n}" for c, n in sorted(by_cls.items()))
            print(
                f"  [{name:16s}] frame {frame_id}: {len(boxes)} box(es) [{cls_str}], "
                f"mean score {mean_score:.3f}"
            )
        print(f"  [{name:16s}] TOTAL detections across {len(frames)} frame(s): {total}")

    rows: list[tuple] = []
    print(f"\n=== displacement vs. baseline '{args.baseline}' ===")
    for name, frames in data.items():
        if name == args.baseline or frames is None:
            continue
        all_disp = []
        total_unmatched = 0
        for frame_id, clean_boxes in sorted(baseline.items()):
            other_boxes = frames.get(frame_id, [])
            disp, unmatched = _match_and_displace(clean_boxes, other_boxes, args.match_radius)
            all_disp.extend(disp)
            total_unmatched += unmatched
            n_clean, n_other = len(clean_boxes), len(other_boxes)
            disp_str = f"{sum(disp) / len(disp):.3f} m" if disp else "n/a (no matches)"
            print(
                f"  [{name:16s}] frame {frame_id}: clean={n_clean} boxes, "
                f"this={n_other} boxes, matched={len(disp)}, lost={unmatched}, "
                f"mean displacement={disp_str}"
            )
        n_clean_total = sum(len(b) for b in baseline.values())
        n_this_total = sum(len(b) for b in frames.values())
        print(
            f"  [{name:16s}] detections: clean={n_clean_total} -> this={n_this_total} "
            f"({n_this_total - n_clean_total:+d})"
        )
        mean_disp = sum(all_disp) / len(all_disp) if all_disp else float("nan")
        rows.append((name, n_clean_total, n_this_total, n_this_total - n_clean_total,
                     len(all_disp), total_unmatched, mean_disp))
        if all_disp:
            print(
                f"  [{name:16s}] OVERALL mean displacement over {len(all_disp)} matched box(es): "
                f"{sum(all_disp) / len(all_disp):.3f} m"
            )
            if sum(all_disp) / len(all_disp) < 1e-6 and n_this_total == n_clean_total:
                print(
                    f"  [{name:16s}] WARNING: zero displacement and identical box count — noise "
                    f"injection may not have fired. Check the job log for the "
                    f"'[noise-injection] Loaded config' banner."
                )
        else:
            print(
                f"  [{name:16s}] no matched boxes across any frame "
                f"(detections suppressed rather than displaced)"
            )


    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["condition", "clean_detections", "detections", "delta",
                        "matched_boxes", "lost_boxes", "mean_displacement_m"])
            for r in rows:
                w.writerow([r[0], r[1], r[2], r[3], r[4], r[5],
                            "" if r[6] != r[6] else f"{r[6]:.3f}"])
        print(f"\nWrote summary table to {args.csv}")


if __name__ == "__main__":
    main()
