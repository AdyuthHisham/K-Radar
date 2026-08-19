#!/usr/bin/env python3
"""Convert every KITTI-format prediction/GT file across the whole sweep into
ONE parquet file, instead of ~1.1M individual .txt files.

Why: Alvis enforces a per-project FILE COUNT quota, not just a byte-size
quota (confirmed: writes failed with "Disk quota exceeded" while the
filesystem itself was 26% full). A single columnar file with one row per
predicted/GT object eliminates that problem entirely and is far more useful
for analysis than 1.1M tiny text files -- no reparsing needed.

Only reads the `all/preds` and `all/gts` subdirs (not the road/time/weather
condition duplicates -- pipeline_detection_v1_0.py writes the same
prediction line 4x, once under `all/` and once more under whichever
road/time/weather condition subdir the frame belongs to; `all/` alone is a
complete, non-redundant set).

Columns: sequence, condition (corruption name, or "original" for the
baseline run), frame (KITTI frame id, e.g. "000042"), source ("pred" or
"gt"), cls, height, width, length, loc_x, loc_y, loc_z, rotation_y, score
(NULL for GT rows, which have no score field).

Schema reference: utils/util_pipeline.py:dict_datum_to_kitti (see that
docstring for the field-order rationale -- native K-Radar sensor-frame
values are written into KITTI's field slots, not a real camera-frame
transform; loc_x/loc_y/loc_z here are exactly what's on disk, not renamed).

Usage:
    python scripts/alvis/predictions_to_parquet.py \\
        --outputs-dir outputs --out outputs/all_predictions.parquet
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import polars as pl

# cls trunc occ alpha bbox_l bbox_t bbox_r bbox_b h w l loc_x loc_y loc_z rot_y [score]
#   0    1   2    3     4      5      6      7    8 9 10  11    12    13    14    15
# 7 dummy fields (trunc..bbox_b) between cls and h -- verified against a real
# pred line before running on the full sweep.
_LINE_RE = re.compile(r"^(\S+)\s+(?:\S+\s+){7}(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)(?:\s+(\S+))?\s*$")


def parse_kitti_file(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("dummy"):
                continue
            m = _LINE_RE.match(line)
            if not m:
                continue
            cls, h, w, l, x, y, z, rot, score = m.groups()
            rows.append({
                "cls": cls, "height": float(h), "width": float(w), "length": float(l),
                "loc_x": float(x), "loc_y": float(y), "loc_z": float(z),
                "rotation_y": float(rot),
                "score": float(score) if score is not None else None,
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--out", default="outputs/all_predictions.parquet")
    args = ap.parse_args()

    pattern = os.path.join(args.outputs_dir, "alvis_seq*", "*", "exp_*",
                            "test_kitti", "*", "*", "all", "preds", "*.txt")
    all_records = []
    n_files = 0
    n_seq_cond_pairs = set()

    for preds_path in sorted(glob.glob(pattern)):
        # .../alvis_seq{N}/{condition}/exp_.../test_kitti/{epoch}/{conf}/all/preds/{frame}.txt
        parts = preds_path.split(os.sep)
        seq_dir = next(p for p in parts if p.startswith("alvis_seq"))
        sequence = seq_dir[len("alvis_seq"):]
        cond_idx = parts.index(seq_dir) + 1
        condition = parts[cond_idx]
        frame = os.path.splitext(os.path.basename(preds_path))[0]

        n_seq_cond_pairs.add((sequence, condition))
        n_files += 1

        for source, src_path in (("pred", preds_path), ("gt", preds_path.replace("/preds/", "/gts/"))):
            if not os.path.exists(src_path):
                continue
            for row in parse_kitti_file(src_path):
                row.update(sequence=sequence, condition=condition, frame=frame, source=source)
                all_records.append(row)

        if n_files % 20000 == 0:
            print(f"...{n_files} pred files processed, {len(all_records)} rows so far")

    print(f"Total: {n_files} prediction files across {len(n_seq_cond_pairs)} (sequence, condition) pairs, "
          f"{len(all_records)} object rows")

    df = pl.DataFrame(all_records).select(
        "sequence", "condition", "frame", "source", "cls",
        "height", "width", "length", "loc_x", "loc_y", "loc_z", "rotation_y", "score",
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.write_parquet(args.out)
    print(f"Wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB, {len(df)} rows)")


if __name__ == "__main__":
    main()
