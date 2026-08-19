#!/usr/bin/env python3
"""Merge every KITTI-format prediction/GT .txt file across the whole sweep
into two flat text files, instead of ~1.1M individual files.

Why: Alvis enforces a per-project FILE COUNT quota, not just a byte-size
quota. Two flat files eliminates that problem for this data entirely.

Only reads the `all/preds` and `all/gts` subdirs (not the road/time/weather
condition duplicates -- pipeline_detection_v1_0.py writes the same
prediction line 4x, once under `all/` and once more under whichever
road/time/weather condition subdir the frame belongs to; `all/` alone is a
complete, non-redundant set).

Output 1: outputs/all_preds.txt
    One line per predicted object, prefixed with an id tag:
        <sequence>_<frame>_<condition> <original KITTI pred line>

Output 2: outputs/all_gts.txt
    GT does not vary by corruption condition -- corrupting sensor inputs
    doesn't move the underlying objects, so every condition's `all/gts/`
    copy for a given (sequence, frame) is identical, just reformatted from
    repos/K-Radar/**/info_label_rev2/*.txt by the eval pipeline. Deduplicated
    to one line per (sequence, frame), first occurrence wins:
        <sequence>_<frame> <original KITTI gt line>

Perf note: the outputs/ tree lives on CephFS -- glob'ing ~280K nested paths
single-threaded is dominated by MDS round-trip latency, not CPU. Each of the
58 alvis_seq* directories is independent, so a thread pool globs+reads them
concurrently (GIL releases during I/O syscalls); only the main thread ever
writes to the two output files, so there's no interleaving/corruption risk
from concurrent writers -- workers just glob+read and hand back in-memory
line lists, one sequence's worth at a time.

Dedup note: some (sequence, condition) pairs have more than one exp_* run
dir (an 08-17 run and a later 08-18 re-run both succeeded in ~128 cases,
confirmed via `find outputs/alvis_seq*/*/exp_*  -maxdepth 1 -type d` per
condition dir). Only the lexicographically-latest exp_* dir per condition is
read -- exp_* names embed a YYMMDD_HHMMSS timestamp, so string-max ==
most-recent -- otherwise those frames' predictions get double-counted.

Usage:
    python scripts/alvis/merge_predictions_text.py \\
        --outputs-dir outputs \\
        --preds-out outputs/all_preds.txt \\
        --gts-out outputs/all_gts.txt \\
        --workers 32
"""
from __future__ import annotations

import argparse
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def process_sequence(seq_dir_path: str) -> tuple[str, list[str], dict[tuple[str, str], list[str]]]:
    """Glob+read every pred file under one alvis_seq* dir. Returns
    (sequence, pred_lines, gt_lines_by_frame) -- gt_lines_by_frame dedups
    within this sequence; cross-sequence dedup is a no-op since sequence is
    part of the key."""
    sequence = os.path.basename(seq_dir_path)[len("alvis_seq"):]

    pred_lines: list[str] = []
    gt_lines_by_frame: dict[tuple[str, str], list[str]] = {}

    for cond_dir in sorted(glob.glob(os.path.join(seq_dir_path, "*"))):
        if not os.path.isdir(cond_dir):
            continue
        condition = os.path.basename(cond_dir)
        exp_dirs = sorted(glob.glob(os.path.join(cond_dir, "exp_*")))
        if not exp_dirs:
            continue
        latest_exp_dir = exp_dirs[-1]  # exp_YYMMDD_HHMMSS_... -- string-max == most recent

        pattern = os.path.join(latest_exp_dir, "test_kitti", "*", "*", "all", "preds", "*.txt")
        for preds_path in glob.glob(pattern):
            frame = os.path.splitext(os.path.basename(preds_path))[0]
            tag = f"{sequence}_{frame}_{condition}"

            with open(preds_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        pred_lines.append(f"{tag} {line}\n")

            gt_key = (sequence, frame)
            if gt_key not in gt_lines_by_frame:
                gt_path = preds_path.replace("/preds/", "/gts/")
                if os.path.exists(gt_path):
                    gt_tag = f"{sequence}_{frame}"
                    lines = []
                    with open(gt_path) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                lines.append(f"{gt_tag} {line}\n")
                    gt_lines_by_frame[gt_key] = lines

    return sequence, pred_lines, gt_lines_by_frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--preds-out", default="outputs/all_preds.txt")
    ap.add_argument("--gts-out", default="outputs/all_gts.txt")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--force", action="store_true",
                     help="Overwrite existing outputs even if the new run produced fewer lines.")
    args = ap.parse_args()

    seq_dirs = sorted(glob.glob(os.path.join(args.outputs_dir, "alvis_seq*")))
    print(f"Found {len(seq_dirs)} sequence directories, dispatching to {args.workers} workers", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.preds_out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.gts_out)), exist_ok=True)

    # Safety: once a sequence's raw test_kitti/ is deleted post-merge (to reclaim
    # Alvis's file-count quota), a from-scratch rerun can only ever see the
    # *remaining* raw data -- it can silently regress already-consolidated
    # sequences back to 0 lines. Write to temp paths and refuse to clobber an
    # existing output with something smaller, unless --force is passed.
    preds_tmp = args.preds_out + ".tmp"
    gts_tmp = args.gts_out + ".tmp"

    n_files = 0
    n_pred_lines = 0
    n_gt_entries = 0
    n_gt_lines = 0
    n_seq_done = 0

    with open(preds_tmp, "w") as preds_out, open(gts_tmp, "w") as gts_out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_sequence, d): d for d in seq_dirs}
            for fut in as_completed(futures):
                sequence, pred_lines, gt_lines_by_frame = fut.result()

                preds_out.writelines(pred_lines)
                n_pred_lines += len(pred_lines)

                for lines in gt_lines_by_frame.values():
                    gts_out.writelines(lines)
                    n_gt_lines += len(lines)
                n_gt_entries += len(gt_lines_by_frame)

                n_seq_done += 1
                print(f"...seq{sequence} done ({n_seq_done}/{len(seq_dirs)}): "
                      f"{len(pred_lines)} pred lines, {len(gt_lines_by_frame)} gt frames", flush=True)

    print(f"Done: {n_pred_lines} pred lines, {n_gt_entries} unique (sequence,frame) GT entries, "
          f"{n_gt_lines} gt lines")

    old_pred_lines = sum(1 for _ in open(args.preds_out)) if os.path.exists(args.preds_out) else 0
    old_gt_lines = sum(1 for _ in open(args.gts_out)) if os.path.exists(args.gts_out) else 0

    if not args.force and (n_pred_lines < old_pred_lines or n_gt_lines < old_gt_lines):
        print(f"REFUSING TO OVERWRITE: new pred_lines={n_pred_lines} (old={old_pred_lines}), "
              f"new gt_lines={n_gt_lines} (old={old_gt_lines}). This run would regress data, likely "
              f"because some sequences' raw test_kitti/ was already deleted post-merge. "
              f"New output left at {preds_tmp} / {gts_tmp} for manual inspection/merge -- "
              f"pass --force to overwrite anyway.")
        raise SystemExit(1)

    os.replace(preds_tmp, args.preds_out)
    os.replace(gts_tmp, args.gts_out)
    print(f"Wrote {args.preds_out} ({os.path.getsize(args.preds_out) / 1e6:.1f} MB)")
    print(f"Wrote {args.gts_out} ({os.path.getsize(args.gts_out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
