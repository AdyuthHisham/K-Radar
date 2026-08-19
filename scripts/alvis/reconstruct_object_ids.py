#!/usr/bin/env python3
"""Reconstruct persistent object identity for every GT and predicted box in
outputs/all_gts.txt / outputs/all_preds.txt.

Background: dict_datum_to_kitti() (utils/util_pipeline.py) discards the
per-object trk_id when serializing to KITTI-format text -- see
`cls_name, cls_idx, (xc, yc, zc, rz, xl, yl, zl), _ = label`. The trk_id
still exists upstream: KRadarFusion_v1_0.get_label() (datasets/
kradar_fusion_v1_0.py) parses it straight from column 2 of each revised
label line (tools/revise_label/kradar_revised_label_v2_0/
KRadar_refined_label_by_UWIPL/{seq}/{radar}_{ldr64}.txt) and collate_fn
carries it through under the name `trk_id` -- i.e. this codebase's own
convention already treats it as a persistent cross-frame identity, not a
throwaway per-frame index.

Two-stage read-only reconstruction, both keyed by (seq, idx_datum) where
idx_datum is the positional index into KRadarFusion_v1_0(cfg, split='test')
.list_dict_item -- NOT the raw label filename. That indexing depends on
os.listdir() sort order plus the cfg's split/remove_0_obj/consider_cls
filters, so it is reproduced by instantiating the real dataset class
read-only (never trains, never touches configs/data loaders) rather than
reimplemented by hand.

Stage A -- GT rejoin (near-exact, low ambiguity):
    all_gts.txt's rounded KITTI columns (h, w, l, loc_x, loc_y, loc_z,
    rot_y) are exactly dict_item['meta']['label']'s (x, y, z, th, l, w, h)
    per object, calib-corrected and ROI-filtered, rounded to 2 decimals and
    column-permuted by dict_datum_to_kitti(). Matching is float-tolerance
    equality on that same permutation, not IoU -- these come from the same
    source values, so ambiguity only arises if two same-class objects in
    one frame round to identical values (detected and flagged, not
    silently resolved).

Stage B -- pred matching (genuine spatial matching, real ambiguity):
    KITTI-format lines are already the K-Radar repo's own "camera format"
    (x, y, z, l, h, w, ry) used by utils/kitti_eval/eval.py's
    bev_box_overlap/d3_box_overlap -- so the merged text files can be fed
    into that IoU code directly with no native-frame corner geometry
    needed. Greedy per-class, per-frame, highest-score-first assignment,
    at every threshold in cfg.VAL.LIST_VAL_IOU (this repo's own AP
    threshold set, so results are consistent with the paper's own
    evaluation instead of an arbitrarily chosen threshold). Unmatched
    preds are kept and flagged FP; unmatched GT are kept and flagged FN --
    never dropped, since silently dropping either would bias any
    downstream per-object corruption-sensitivity analysis.

Parallelism: Stage A does one dataset pass per sequence (all_gts.txt is
already deduplicated across the 35 corruption conditions, so there is no
redundant work to remove there). Stage B is embarrassingly parallel across
sequences -- a process pool handles one sequence at a time, and within a
sequence every condition's preds for a given frame reuse that frame's GT
box array once instead of re-deriving it 35x.

This must run where KRadarFusion_v1_0's dependencies (torch, open3d, cv2)
and the real dataset paths are available, i.e. on Alvis via `module load`
or a container, submitted with sbatch -- never on the login node, never
with a host-level pip/uv/conda install.

Usage:
    python scripts/alvis/reconstruct_object_ids.py \\
        --outputs-dir outputs \\
        --configs-dir configs \\
        --gts-out outputs/all_gts_with_trkid.txt \\
        --preds-out outputs/all_preds_with_trkid.txt \\
        --ambiguous-out outputs/all_gts_ambiguous.txt \\
        --workers 16

    Dry run on one sequence before the full sweep:
        python scripts/alvis/reconstruct_object_ids.py --seqs 32 --workers 1 \\
            --gts-out outputs/seq32_gts_with_trkid.txt \\
            --preds-out outputs/seq32_preds_with_trkid.txt
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np

sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from utils.util_config import cfg, cfg_from_yaml_file  # noqa: E402
from utils.kitti_eval.eval import bev_box_overlap  # noqa: E402
from datasets.kradar_fusion_v1_0 import KRadarFusion_v1_0  # noqa: E402

ROUND_TOL = 5e-3  # dict_datum_to_kitti rounds to 2 decimals; allow half-ULP-ish slack


@dataclass
class GtLine:
    seq: str
    frame: str
    line_idx: int  # position within this (seq, frame)'s block in all_gts.txt
    cls_kw: str
    h: float
    w: float
    l: float
    loc_x: float
    loc_y: float
    loc_z: float
    rot_y: float
    trk_id: int | None = None
    ambiguous: bool = False


@dataclass
class PredLine:
    seq: str
    frame: str
    condition: str
    line_idx: int
    cls_kw: str
    h: float
    w: float
    l: float
    loc_x: float
    loc_y: float
    loc_z: float
    rot_y: float
    score: float
    trk_id: int | None = None
    matched_gt_line_idx: int | None = None
    is_fp: bool = False


def parse_gts_file(path: str) -> dict[tuple[str, str], list[GtLine]]:
    by_frame: dict[tuple[str, str], list[GtLine]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            seq, frame = tag.split("_", 1)
            key = (seq, frame)
            idx = len(by_frame[key])
            by_frame[key].append(GtLine(
                seq=seq, frame=frame, line_idx=idx, cls_kw=parts[1],
                h=float(parts[9]), w=float(parts[10]), l=float(parts[11]),
                loc_x=float(parts[12]), loc_y=float(parts[13]), loc_z=float(parts[14]),
                rot_y=float(parts[15]),
            ))
    return by_frame


def parse_preds_file(path: str) -> dict[str, dict[tuple[str, str], list[PredLine]]]:
    """Returns {condition: {(seq, frame): [PredLine, ...]}}"""
    by_cond: dict[str, dict[tuple[str, str], list[PredLine]]] = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[1] == "dummy":
                continue
            tag = parts[0]
            seq, frame, condition = tag.split("_", 2)
            key = (seq, frame)
            idx = len(by_cond[condition][key])
            by_cond[condition][key].append(PredLine(
                seq=seq, frame=frame, condition=condition, line_idx=idx, cls_kw=parts[1],
                h=float(parts[9]), w=float(parts[10]), l=float(parts[11]),
                loc_x=float(parts[12]), loc_y=float(parts[13]), loc_z=float(parts[14]),
                rot_y=float(parts[15]), score=float(parts[16]),
            ))
    return by_cond


def stage_a_rejoin_sequence(
    seq: str,
    frames: dict[str, list[GtLine]],
    configs_dir: str,
    class_val_keyword: dict[str, str],
) -> tuple[dict[str, list[GtLine]], list[GtLine], list[tuple[str, str]], list[tuple[str, str, str]]]:
    """Read-only: instantiate KRadarFusion_v1_0(split='test') for this
    sequence's cfg and match dataset trk_id back onto each GtLine by
    float-tolerance equality on the same rounding dict_datum_to_kitti used.

    As a byproduct, also returns the (seq, tesseract_idx, kitti_frame_number)
    mapping for every idx_datum in this sequence -- this is the frame-identity
    join the enrichment plan (docs/vAIlt/06_Results/robustness-metric-implementation-plan.md
    S1.3) needs to attach kradar_test_split_full.parquet's distance/weather
    columns (keyed by tesseract_idx) onto all_gts.txt/all_preds.txt (keyed by
    kitti_frame_number). Free to compute here since this loop already walks
    dataset.list_dict_item with idx_datum in hand -- no separate replay script
    needed, no replication risk (see docstring at top of this file).
    """
    cfg_path = osp.join(configs_dir, f"ASF_v2_0_seq{seq}_alvis.yml")
    if not osp.exists(cfg_path):
        raise FileNotFoundError(f"No per-sequence cfg for seq {seq}: {cfg_path}")

    local_cfg = cfg_from_yaml_file(cfg_path, cfg)
    dataset = KRadarFusion_v1_0(local_cfg, split="test")

    ambiguous_keys: list[tuple[str, str]] = []
    unmatched_gt_lines: list[GtLine] = []
    frame_index_map: list[tuple[str, str, str]] = []  # (seq, tesseract_idx, kitti_frame_number)

    for idx_datum, dict_item in enumerate(dataset.list_dict_item):
        frame_tag = str(idx_datum).zfill(6)
        tesseract_idx = dict_item["meta"]["idx"]["rdr"]
        frame_index_map.append((seq, tesseract_idx, frame_tag))

        gt_lines = frames.get(frame_tag)
        if gt_lines is None:
            continue  # frame had zero objects -> dict_datum_to_kitti wrote nothing

        dataset_objs = dict_item["meta"]["label"]  # [(cls_name, (x,y,z,th,l,w,h), trk_id, avail), ...]
        candidates = []
        for cls_name, (x, y, z, th, l, w, h), trk_id, _avail in dataset_objs:
            cls_kw = class_val_keyword.get(cls_name)
            if cls_kw is None:
                continue
            candidates.append(dict(
                cls_kw=cls_kw, trk_id=trk_id,
                h=round(h, 2), w=round(w, 2), l=round(l, 2),
                loc_x=round(y, 2), loc_y=round(z, 2), loc_z=round(x, 2),
                rot_y=round(th, 2),
            ))

        used = [False] * len(candidates)
        for gl in gt_lines:
            matches = [
                i for i, c in enumerate(candidates)
                if not used[i]
                and c["cls_kw"] == gl.cls_kw
                and abs(c["h"] - gl.h) < ROUND_TOL
                and abs(c["w"] - gl.w) < ROUND_TOL
                and abs(c["l"] - gl.l) < ROUND_TOL
                and abs(c["loc_x"] - gl.loc_x) < ROUND_TOL
                and abs(c["loc_y"] - gl.loc_y) < ROUND_TOL
                and abs(c["loc_z"] - gl.loc_z) < ROUND_TOL
                and abs(c["rot_y"] - gl.rot_y) < ROUND_TOL
            ]
            if len(matches) == 1:
                i = matches[0]
                gl.trk_id = candidates[i]["trk_id"]
                used[i] = True
            elif len(matches) > 1:
                gl.ambiguous = True
                ambiguous_keys.append((seq, frame_tag))
                i = matches[0]
                gl.trk_id = candidates[i]["trk_id"]
                used[i] = True
            else:
                unmatched_gt_lines.append(gl)

    return frames, unmatched_gt_lines, ambiguous_keys, frame_index_map


def greedy_match_frame(
    gt_lines: list[GtLine],
    pred_lines: list[PredLine],
    iou_thresh: float,
) -> None:
    """Per-class, highest-score-first greedy assignment. Mutates pred_lines
    (trk_id / matched_gt_line_idx / is_fp) in place. Unmatched GT are simply
    left with matched_gt_line_idx unset by any pred -- caller derives FN by
    set difference.
    """
    by_cls: dict[str, list[int]] = defaultdict(list)
    for i, gl in enumerate(gt_lines):
        by_cls[gl.cls_kw].append(i)

    pred_by_cls: dict[str, list[int]] = defaultdict(list)
    for i, pl in enumerate(pred_lines):
        pred_by_cls[pl.cls_kw].append(i)

    for cls_kw, pred_idxs in pred_by_cls.items():
        gt_idxs = by_cls.get(cls_kw, [])
        if not gt_idxs:
            for pi in pred_idxs:
                pred_lines[pi].is_fp = True
            continue

        gt_boxes = np.array([
            [gt_lines[gi].loc_x, gt_lines[gi].loc_y, gt_lines[gi].loc_z,
             gt_lines[gi].l, gt_lines[gi].h, gt_lines[gi].w, gt_lines[gi].rot_y]
            for gi in gt_idxs
        ], dtype=np.float64)
        pred_boxes = np.array([
            [pred_lines[pi].loc_x, pred_lines[pi].loc_y, pred_lines[pi].loc_z,
             pred_lines[pi].l, pred_lines[pi].h, pred_lines[pi].w, pred_lines[pi].rot_y]
            for pi in pred_idxs
        ], dtype=np.float64)

        # bev_box_overlap (unlike d3_box_overlap) does NOT slice out the BEV
        # columns itself -- it expects pre-sliced [cx, cy, w, l, angle] rows.
        # Camera-format convention here is z_axis=1 (loc_y is the height
        # axis, h is the height extent) matching d3_box_overlap's own
        # default, so drop columns 1 (loc_y) and 4 (h): [loc_x, loc_z, l, w, rot_y].
        bev_axes = [0, 2, 3, 5, 6]
        overlaps = bev_box_overlap(pred_boxes[:, bev_axes], gt_boxes[:, bev_axes])  # [n_pred, n_gt]

        order = sorted(range(len(pred_idxs)), key=lambda k: pred_lines[pred_idxs[k]].score, reverse=True)
        gt_taken = [False] * len(gt_idxs)
        for k in order:
            pi = pred_idxs[k]
            row = overlaps[k]
            best_j, best_iou = -1, iou_thresh
            for j in range(len(gt_idxs)):
                if gt_taken[j]:
                    continue
                if row[j] > best_iou:
                    best_iou = row[j]
                    best_j = j
            if best_j >= 0:
                gt_taken[best_j] = True
                gi = gt_idxs[best_j]
                pred_lines[pi].trk_id = gt_lines[gi].trk_id
                pred_lines[pi].matched_gt_line_idx = gt_lines[gi].line_idx
            else:
                pred_lines[pi].is_fp = True


def stage_b_match_sequence(
    seq: str,
    gt_frames: dict[str, list[GtLine]],
    preds_by_cond: dict[str, dict[tuple[str, str], list[PredLine]]],
    iou_thresh: float,
) -> tuple[list[PredLine], list[tuple[str, str, str, int]]]:
    """FN entries returned as (seq, frame, condition, gt_line_idx)."""
    all_preds: list[PredLine] = []
    fn_entries: list[tuple[str, str, str, int]] = []

    for condition, frames in preds_by_cond.items():
        for frame, pred_lines in frames.items():
            gt_lines = gt_frames.get(frame, [])
            greedy_match_frame(gt_lines, pred_lines, iou_thresh)
            matched_gt_idxs = {pl.matched_gt_line_idx for pl in pred_lines if pl.matched_gt_line_idx is not None}
            for gl in gt_lines:
                if gl.line_idx not in matched_gt_idxs:
                    fn_entries.append((seq, frame, condition, gl.line_idx))
            all_preds.extend(pred_lines)

    return all_preds, fn_entries


def _stage_a_worker(args):
    seq, frames, configs_dir, class_val_keyword = args
    return seq, stage_a_rejoin_sequence(seq, frames, configs_dir, class_val_keyword)


def _stage_b_worker(args):
    seq, gt_frames, preds_by_cond, iou_thresh = args
    return seq, stage_b_match_sequence(seq, gt_frames, preds_by_cond, iou_thresh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--configs-dir", default="configs")
    ap.add_argument("--gts-in", default=None)
    ap.add_argument("--preds-in", default=None)
    ap.add_argument("--gts-out", default=None)
    ap.add_argument("--preds-out", default=None)
    ap.add_argument("--fn-out", default=None)
    ap.add_argument("--ambiguous-out", default=None)
    ap.add_argument("--frame-index-map-out", default=None,
                     help="(seq, tesseract_idx, kitti_frame_number) mapping, byproduct of Stage A -- "
                          "feeds the kradar_test_split_full.parquet distance/weather enrichment join.")
    ap.add_argument("--unmatched-gt-out", default=None,
                     help="GT lines that had no dataset trk_id match in Stage A (should be rare/zero).")
    ap.add_argument("--iou-thresh", type=float, default=0.5,
                     help="BEV IoU threshold for pred<->GT matching (repo's own VAL.LIST_VAL_IOU is [0.7, 0.5, 0.3])")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seqs", default=None,
                     help="Comma-separated sequence numbers to restrict processing to, e.g. '32,33' "
                          "for a cheap dry run before the full sweep. Default: all sequences present in the input files.")
    args = ap.parse_args()

    seq_filter = None
    if args.seqs:
        seq_filter = {s.strip() for s in args.seqs.split(",") if s.strip()}

    gts_in = args.gts_in or osp.join(args.outputs_dir, "all_gts.txt")
    preds_in = args.preds_in or osp.join(args.outputs_dir, "all_preds.txt")
    gts_out = args.gts_out or osp.join(args.outputs_dir, "all_gts_with_trkid.txt")
    preds_out = args.preds_out or osp.join(args.outputs_dir, "all_preds_with_trkid.txt")
    fn_out = args.fn_out or osp.join(args.outputs_dir, "all_fn.txt")
    ambiguous_out = args.ambiguous_out or osp.join(args.outputs_dir, "all_gts_ambiguous.txt")
    frame_index_map_out = args.frame_index_map_out or osp.join(args.outputs_dir, "frame_index_map.txt")
    unmatched_gt_out = args.unmatched_gt_out or osp.join(args.outputs_dir, "all_gts_unmatched.txt")

    class_val_keyword = {
        "Sedan": "sed", "Bus or Truck": "bus", "Motorcycle": "mot",
        "Bicycle": "bic", "Bicycle Group": "big",
        "Pedestrian": "ped", "Pedestrian Group": "peg",
    }

    print(f"args.seqs={args.seqs!r} gts_in={gts_in} preds_in={preds_in}", flush=True)
    print("Parsing all_gts.txt / all_preds.txt ...", flush=True)
    gt_by_frame = parse_gts_file(gts_in)
    preds_all = parse_preds_file(preds_in)
    n_pred_lines = sum(len(v) for frames in preds_all.values() for v in frames.values())
    print(f"Parsed {len(gt_by_frame)} GT frames, {n_pred_lines} pred lines across {len(preds_all)} conditions.", flush=True)
    if not gt_by_frame or not preds_all:
        print("WARNING: input file(s) parsed empty -- likely read mid-write by a concurrent merge job. "
              "Re-run once the writer (merge_predictions_text.py) is not actively rewriting outputs/all_gts.txt "
              "/ all_preds.txt, or snapshot them to a private copy first.", flush=True)

    if seq_filter is not None:
        gt_by_frame = {(seq, frame): lines for (seq, frame), lines in gt_by_frame.items() if seq in seq_filter}
        preds_all = {
            condition: {(seq, frame): lines for (seq, frame), lines in frames.items() if seq in seq_filter}
            for condition, frames in preds_all.items()
        }
        print(f"--seqs filter active: restricting to sequences {sorted(seq_filter)}", flush=True)

    gt_by_seq: dict[str, dict[str, list[GtLine]]] = defaultdict(dict)
    for (seq, frame), lines in gt_by_frame.items():
        gt_by_seq[seq][frame] = lines

    preds_by_seq: dict[str, dict[str, dict[str, list[PredLine]]]] = defaultdict(lambda: defaultdict(dict))
    for condition, frames in preds_all.items():
        for (seq, frame), lines in frames.items():
            preds_by_seq[seq][condition][frame] = lines

    # --- Stage A: GT trk_id rejoin, parallel across sequences ---
    print(f"Stage A: GT rejoin across {len(gt_by_seq)} sequences ({args.workers} workers) ...", flush=True)
    all_unmatched_gt: list[GtLine] = []
    all_ambiguous: list[tuple[str, str]] = []
    all_frame_index_map: list[tuple[str, str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(_stage_a_worker, (seq, frames, args.configs_dir, class_val_keyword))
            for seq, frames in gt_by_seq.items()
        ]
        for fut in as_completed(futures):
            seq, (mutated_frames, unmatched, ambiguous, frame_index_map) = fut.result()
            gt_by_seq[seq] = mutated_frames  # trk_id/ambiguous mutations happened in a worker process; propagate back
            all_unmatched_gt.extend(unmatched)
            all_ambiguous.extend(ambiguous)
            all_frame_index_map.extend(frame_index_map)
            print(f"  seq {seq}: {len(unmatched)} unmatched, {len(ambiguous)} ambiguous frames, "
                  f"{len(frame_index_map)} frame-index rows", flush=True)

    with open(gts_out, "w") as f:
        for seq, frames in gt_by_seq.items():
            for frame, lines in frames.items():
                for gl in sorted(lines, key=lambda g: g.line_idx):
                    f.write(f"{gl.seq}_{gl.frame} {gl.cls_kw} {gl.trk_id} {int(gl.ambiguous)} "
                            f"{gl.h} {gl.w} {gl.l} {gl.loc_x} {gl.loc_y} {gl.loc_z} {gl.rot_y}\n")

    with open(ambiguous_out, "w") as f:
        for seq, frame in all_ambiguous:
            f.write(f"{seq}_{frame}\n")

    with open(frame_index_map_out, "w") as f:
        for seq, tesseract_idx, kitti_frame in sorted(all_frame_index_map, key=lambda r: (int(r[0]), r[2])):
            f.write(f"{seq} {tesseract_idx} {kitti_frame}\n")
    print(f"Frame-index map: {len(all_frame_index_map)} rows -> {frame_index_map_out}", flush=True)

    with open(unmatched_gt_out, "w") as f:
        for gl in all_unmatched_gt:
            f.write(f"{gl.seq}_{gl.frame} {gl.cls_kw} {gl.h} {gl.w} {gl.l} "
                    f"{gl.loc_x} {gl.loc_y} {gl.loc_z} {gl.rot_y}\n")

    if all_unmatched_gt:
        print(f"WARNING: {len(all_unmatched_gt)} GT lines had no dataset trk_id match "
              f"(rounding tolerance {ROUND_TOL}) -- investigate before trusting Stage B. "
              f"Written to {unmatched_gt_out}", flush=True)

    # --- Stage B: pred<->GT IoU matching, parallel across sequences ---
    print(f"Stage B: pred matching at IoU>{args.iou_thresh} across {len(preds_by_seq)} sequences ...", flush=True)
    all_matched_preds: list[PredLine] = []
    all_fn: list[tuple[str, str, str, int]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(_stage_b_worker, (seq, gt_by_seq.get(seq, {}), dict(cond_frames), args.iou_thresh))
            for seq, cond_frames in preds_by_seq.items()
        ]
        for fut in as_completed(futures):
            seq, (preds, fn_entries) = fut.result()
            all_matched_preds.extend(preds)
            all_fn.extend(fn_entries)
            n_fp = sum(1 for p in preds if p.is_fp)
            print(f"  seq {seq}: {len(preds)} preds, {n_fp} FP, {len(fn_entries)} FN", flush=True)

    with open(preds_out, "w") as f:
        for pl in all_matched_preds:
            f.write(f"{pl.seq}_{pl.frame}_{pl.condition} {pl.cls_kw} {pl.trk_id} {int(pl.is_fp)} "
                    f"{pl.h} {pl.w} {pl.l} {pl.loc_x} {pl.loc_y} {pl.loc_z} {pl.rot_y} {pl.score}\n")

    with open(fn_out, "w") as f:
        for seq, frame, condition, gt_line_idx in all_fn:
            f.write(f"{seq}_{frame}_{condition} gt_line_idx={gt_line_idx}\n")

    n_fp_total = sum(1 for p in all_matched_preds if p.is_fp)
    print(f"Done. {len(all_matched_preds)} preds ({n_fp_total} FP), {len(all_fn)} FN, "
          f"{len(all_unmatched_gt)} unmatched GT, {len(all_ambiguous)} ambiguous GT frames.", flush=True)


if __name__ == "__main__":
    main()
