#!/usr/bin/env python3
"""Ingestion layer: turn the sweep's raw text output into one canonical
prediction-row-grain polars.DataFrame that every downstream metric
(RR/mRR, CRI, stratified AP, TP-error decomposition, GT-object transition
analysis) consumes, instead of each metric script re-parsing/re-matching on
its own. See docs/vAIlt/06_Results/robustness-metric-implementation-plan.md
for the design rationale.

Two matching backends, auto-selected by file presence (§2.5 of the plan):

  1. trk_id path (preferred): reconstruct_object_ids.py's output --
     outputs/all_gts_with_trkid.txt, outputs/all_preds_with_trkid.txt,
     outputs/all_fn.txt. Object identity comes from K-Radar's own persistent
     trk_id (rejoined from the UWIPL-refined label tree) plus IoU-based
     pred<->GT matching using the repo's own bev_box_overlap -- the same
     metric complete_results.txt's AP numbers use. Requires running
     reconstruct_object_ids.py on Alvis first (needs torch/open3d/cv2 + the
     real dataset); NOT present as of this writing, so this path is
     implemented but unvalidated against real data -- see plan §1.3/§2.5.

  2. Coordinate/IoU fallback (this repo's own matcher): parses
     outputs/all_preds.txt / all_gts.txt directly and matches each frame's
     predictions against that frame's GT using
     gt_transition_analysis.match_frame_pairs (BEV IoU, class-conditioned,
     greedy highest-score-first). This is the only path actually testable
     on this machine right now (both files are present locally) -- used for
     local dev/iteration and as a cross-check once the trk_id path exists.

Output grain: one row per (sequence, condition, frame, prediction), plus one
synthesized row per unmatched GT object (is_fn=True; prediction-specific
columns null). Distance/weather enrichment from kradar_test_split_full.parquet
is NOT yet wired in here -- that requires the separate tesseract_idx ->
kitti_frame_number mapping (plan §1.3), which is a different problem from
object identity and still open at the time of writing.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Optional

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gt_transition_analysis import match_frame_pairs  # noqa: E402

# cls trunc occ alpha bbox_l bbox_t bbox_r bbox_b h w l loc_x loc_y loc_z rot_y [score]
#   0    1   2    3     4      5      6      7    8 9 10  11    12    13    14    15
_LINE_RE = re.compile(
    r"^(\S+)\s+(?:\S+\s+){7}(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)(?:\s+(\S+))?\s*$"
)


def _parse_kitti_line(line: str) -> Optional[dict]:
    m = _LINE_RE.match(line)
    if not m:
        return None
    cls, h, w, l, x, y, z, rot, score = m.groups()
    return {
        "cls": cls, "height": float(h), "width": float(w), "length": float(l),
        "loc_x": float(x), "loc_y": float(y), "loc_z": float(z),
        "rotation_y": float(rot), "score": float(score) if score is not None else None,
    }


def _parse_preds_txt(path: str) -> pl.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tag, _, rest = line.partition(" ")
            seq, frame, condition = tag.split("_", 2)
            parsed = _parse_kitti_line(rest)
            if parsed is None:
                continue
            parsed.update(sequence=seq, frame=frame, condition=condition)
            rows.append(parsed)
    if not rows:
        return pl.DataFrame(schema={
            "sequence": pl.Utf8, "frame": pl.Utf8, "condition": pl.Utf8, "cls": pl.Utf8,
            "height": pl.Float64, "width": pl.Float64, "length": pl.Float64,
            "loc_x": pl.Float64, "loc_y": pl.Float64, "loc_z": pl.Float64,
            "rotation_y": pl.Float64, "score": pl.Float64,
        })
    return pl.DataFrame(rows)


def _parse_gts_txt(path: str) -> pl.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tag, _, rest = line.partition(" ")
            seq, frame = tag.split("_", 1)
            parsed = _parse_kitti_line(rest)
            if parsed is None:
                continue
            parsed.pop("score", None)
            parsed.update(sequence=seq, frame=frame)
            rows.append(parsed)
    if not rows:
        return pl.DataFrame(schema={
            "sequence": pl.Utf8, "frame": pl.Utf8, "cls": pl.Utf8,
            "height": pl.Float64, "width": pl.Float64, "length": pl.Float64,
            "loc_x": pl.Float64, "loc_y": pl.Float64, "loc_z": pl.Float64,
            "rotation_y": pl.Float64,
        })
    return pl.DataFrame(rows)


def _match_via_coordinates(preds: pl.DataFrame, gts: pl.DataFrame, iou_threshold: float) -> pl.DataFrame:
    """Fallback backend: per-(sequence, frame, condition) BEV IoU matching
    against that frame's raw all_gts.txt rows (no cross-condition dedup --
    GT content is identical across conditions by construction, so matching
    condition-by-condition against its own GT copy gives the same result as
    matching against a canonical copy, just recomputed per condition).
    """
    preds = preds.with_row_index("pred_row_id")
    gts = gts.with_row_index("gt_row_id")

    out_rows: list[dict] = []
    gt_groups = {(s, f): df for (s, f), df in gts.group_by(["sequence", "frame"], maintain_order=True)}

    for (seq, frame, cond), pred_frame in preds.group_by(["sequence", "frame", "condition"], maintain_order=True):
        gt_frame = gt_groups.get((seq, frame))
        if gt_frame is None or len(gt_frame) == 0:
            gt_frame = gts.clear()

        pairs = match_frame_pairs(gt_frame, pred_frame, iou_threshold)
        matched_gt_idx = {gi for gi, _, _ in pairs}
        matched_pred_idx = {pi for _, pi, _ in pairs}
        iou_by_pred = {pi: iou for _, pi, iou in pairs}
        gt_row_id_by_local_idx = {i: int(gt_frame["gt_row_id"][i]) for i in range(len(gt_frame))}

        for i in range(len(pred_frame)):
            row = {c: pred_frame[c][i] for c in pred_frame.columns}
            is_tp = i in matched_pred_idx
            row["is_tp"] = is_tp
            row["is_fp"] = not is_tp
            row["is_fn"] = False
            row["iou"] = iou_by_pred.get(i)
            if is_tp:
                gt_local = next(gi for gi, pi, _ in pairs if pi == i)
                row["matched_gt_row_id"] = gt_row_id_by_local_idx[gt_local]
                row["gt_height"] = gt_frame["height"][gt_local]
                row["gt_width"] = gt_frame["width"][gt_local]
                row["gt_length"] = gt_frame["length"][gt_local]
                row["gt_loc_x"] = gt_frame["loc_x"][gt_local]
                row["gt_loc_y"] = gt_frame["loc_y"][gt_local]
                row["gt_loc_z"] = gt_frame["loc_z"][gt_local]
                row["gt_rotation_y"] = gt_frame["rotation_y"][gt_local]
            else:
                for c in ("matched_gt_row_id", "gt_height", "gt_width", "gt_length",
                          "gt_loc_x", "gt_loc_y", "gt_loc_z", "gt_rotation_y"):
                    row[c] = None
            out_rows.append(row)

        for gi in range(len(gt_frame)):
            if gi in matched_gt_idx:
                continue
            fn_row = {c: None for c in pred_frame.columns if c not in ("sequence", "frame", "condition")}
            fn_row.update(sequence=seq, frame=frame, condition=cond)
            fn_row["is_tp"] = False
            fn_row["is_fp"] = False
            fn_row["is_fn"] = True
            fn_row["iou"] = None
            fn_row["matched_gt_row_id"] = int(gt_frame["gt_row_id"][gi])
            fn_row["cls"] = gt_frame["cls"][gi]
            fn_row["gt_height"] = gt_frame["height"][gi]
            fn_row["gt_width"] = gt_frame["width"][gi]
            fn_row["gt_length"] = gt_frame["length"][gi]
            fn_row["gt_loc_x"] = gt_frame["loc_x"][gi]
            fn_row["gt_loc_y"] = gt_frame["loc_y"][gi]
            fn_row["gt_loc_z"] = gt_frame["loc_z"][gi]
            fn_row["gt_rotation_y"] = gt_frame["rotation_y"][gi]
            out_rows.append(fn_row)

    return pl.DataFrame(out_rows)


def _load_via_trkid(gts_path: str, preds_path: str, fn_path: str) -> pl.DataFrame:
    """Preferred backend: parse reconstruct_object_ids.py's output.

    outputs/all_gts_with_trkid.txt columns:
        seq_frame cls_kw trk_id ambiguous h w l loc_x loc_y loc_z rot_y
    outputs/all_preds_with_trkid.txt columns:
        seq_frame_condition cls_kw trk_id is_fp h w l loc_x loc_y loc_z rot_y score
    outputs/all_fn.txt columns:
        seq_frame_condition gt_line_idx=<idx>
    """
    gt_rows = []
    with open(gts_path) as f:
        for line_idx_in_file, line in enumerate(f):
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            seq, frame = tag.split("_", 1)
            gt_rows.append({
                "sequence": seq, "frame": frame, "cls": parts[1],
                "trk_id": int(parts[2]) if parts[2] != "None" else None,
                "ambiguous": bool(int(parts[3])),
                "height": float(parts[4]), "width": float(parts[5]), "length": float(parts[6]),
                "loc_x": float(parts[7]), "loc_y": float(parts[8]), "loc_z": float(parts[9]),
                "rotation_y": float(parts[10]),
            })
    gt_df = pl.DataFrame(gt_rows)

    pred_rows = []
    with open(preds_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            seq, frame, condition = tag.split("_", 2)
            pred_rows.append({
                "sequence": seq, "frame": frame, "condition": condition, "cls": parts[1],
                "trk_id": int(parts[2]) if parts[2] != "None" else None,
                "is_fp": bool(int(parts[3])),
                "height": float(parts[4]), "width": float(parts[5]), "length": float(parts[6]),
                "loc_x": float(parts[7]), "loc_y": float(parts[8]), "loc_z": float(parts[9]),
                "rotation_y": float(parts[10]), "score": float(parts[11]),
            })
    pred_df = pl.DataFrame(pred_rows)

    # Join preds -> matched GT box on (sequence, frame, trk_id) where trk_id
    # is not null and is_fp is False (FP preds have no matched GT by
    # construction -- reconstruct_object_ids.py's greedy_match_frame only
    # sets trk_id on a match).
    gt_for_join = gt_df.rename({
        "trk_id": "trk_id", "height": "gt_height", "width": "gt_width", "length": "gt_length",
        "loc_x": "gt_loc_x", "loc_y": "gt_loc_y", "loc_z": "gt_loc_z", "rotation_y": "gt_rotation_y",
    }).select("sequence", "frame", "trk_id", "gt_height", "gt_width", "gt_length",
              "gt_loc_x", "gt_loc_y", "gt_loc_z", "gt_rotation_y")

    pred_df = pred_df.with_columns(
        (~pl.col("is_fp") & pl.col("trk_id").is_not_null()).alias("is_tp"),
    ).with_columns(
        pl.col("is_fp").alias("is_fp"),
        pl.lit(False).alias("is_fn"),
    )
    pred_df = pred_df.join(gt_for_join, on=["sequence", "frame", "trk_id"], how="left")

    # FN rows: outputs/all_fn.txt references gt_line_idx into all_gts.txt's
    # per-(seq,frame) block, NOT this trk_id file's row order -- but since
    # both are written in the same per-frame order as all_gts.txt (see
    # reconstruct_object_ids.py's parse_gts_file), the Nth gt row per
    # (seq, frame) in gt_df corresponds to gt_line_idx==N here too.
    gt_df = gt_df.with_columns(pl.int_range(pl.len()).over(["sequence", "frame"]).alias("gt_line_idx"))

    fn_rows = []
    with open(fn_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            tag, kv = parts[0], parts[1]
            seq, frame, condition = tag.split("_", 2)
            gt_line_idx = int(kv.split("=")[1])
            fn_rows.append({"sequence": seq, "frame": frame, "condition": condition, "gt_line_idx": gt_line_idx})
    fn_df = pl.DataFrame(fn_rows) if fn_rows else pl.DataFrame(
        schema={"sequence": pl.Utf8, "frame": pl.Utf8, "condition": pl.Utf8, "gt_line_idx": pl.Int64})

    fn_df = fn_df.join(
        gt_df.select("sequence", "frame", "gt_line_idx", "cls", "trk_id",
                      "height", "width", "length", "loc_x", "loc_y", "loc_z", "rotation_y"),
        on=["sequence", "frame", "gt_line_idx"], how="left",
    ).rename({
        "height": "gt_height", "width": "gt_width", "length": "gt_length",
        "loc_x": "gt_loc_x", "loc_y": "gt_loc_y", "loc_z": "gt_loc_z", "rotation_y": "gt_rotation_y",
    }).with_columns(
        pl.lit(False).alias("is_tp"), pl.lit(False).alias("is_fp"), pl.lit(True).alias("is_fn"),
        pl.lit(None, dtype=pl.Float64).alias("height"), pl.lit(None, dtype=pl.Float64).alias("width"),
        pl.lit(None, dtype=pl.Float64).alias("length"), pl.lit(None, dtype=pl.Float64).alias("loc_x"),
        pl.lit(None, dtype=pl.Float64).alias("loc_y"), pl.lit(None, dtype=pl.Float64).alias("loc_z"),
        pl.lit(None, dtype=pl.Float64).alias("rotation_y"), pl.lit(None, dtype=pl.Float64).alias("score"),
    ).drop("gt_line_idx")

    common_cols = ["sequence", "frame", "condition", "cls", "trk_id",
                    "height", "width", "length", "loc_x", "loc_y", "loc_z", "rotation_y", "score",
                    "is_tp", "is_fp", "is_fn", "gt_height", "gt_width", "gt_length",
                    "gt_loc_x", "gt_loc_y", "gt_loc_z", "gt_rotation_y"]
    pred_df = pred_df.select(common_cols)
    fn_df = fn_df.select(common_cols)
    return pl.concat([pred_df, fn_df], how="vertical_relaxed")


def load_prediction_rows(
    preds_txt: str = "outputs/all_preds.txt",
    gts_txt: str = "outputs/all_gts.txt",
    trkid_gts_txt: str = "outputs/all_gts_with_trkid.txt",
    trkid_preds_txt: str = "outputs/all_preds_with_trkid.txt",
    fn_txt: str = "outputs/all_fn.txt",
    iou_threshold: float = 0.5,
    force_backend: Optional[str] = None,
) -> pl.DataFrame:
    """Canonical entry point: one row per (sequence, condition, frame,
    prediction) plus synthesized FN rows, with TP/FP/FN status and (where
    matched) both the predicted and matched-GT box for TP-error decomposition.

    Backend selection: trk_id path if all three of trkid_gts_txt/
    trkid_preds_txt/fn_txt exist, else the coordinate/IoU fallback on
    preds_txt/gts_txt. Pass force_backend="trkid"|"coordinate" to override.
    """
    use_trkid = (
        force_backend == "trkid"
        or (force_backend is None
            and os.path.exists(trkid_gts_txt) and os.path.exists(trkid_preds_txt) and os.path.exists(fn_txt))
    )
    if use_trkid:
        return _load_via_trkid(trkid_gts_txt, trkid_preds_txt, fn_txt)

    preds = _parse_preds_txt(preds_txt)
    gts = _parse_gts_txt(gts_txt)
    return _match_via_coordinates(preds, gts, iou_threshold)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--force-backend", choices=["trkid", "coordinate"], default=None)
    ap.add_argument("--out", default=None, help="optional parquet path to write the row table to")
    args = ap.parse_args()

    df = load_prediction_rows(
        preds_txt=os.path.join(args.outputs_dir, "all_preds.txt"),
        gts_txt=os.path.join(args.outputs_dir, "all_gts.txt"),
        trkid_gts_txt=os.path.join(args.outputs_dir, "all_gts_with_trkid.txt"),
        trkid_preds_txt=os.path.join(args.outputs_dir, "all_preds_with_trkid.txt"),
        fn_txt=os.path.join(args.outputs_dir, "all_fn.txt"),
        iou_threshold=args.iou_threshold,
        force_backend=args.force_backend,
    )
    print(f"Loaded {len(df)} rows: {df['is_tp'].sum()} TP, {df['is_fp'].sum()} FP, {df['is_fn'].sum()} FN")
    if args.out:
        df.write_parquet(args.out)
        print(f"Wrote {args.out}")
