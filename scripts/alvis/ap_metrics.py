#!/usr/bin/env python3
"""41-point interpolated AP, computed directly on ingest.py's row-level
prediction table, parameterized by an arbitrary group-by key (class,
distance_bin, condition, ...). Reimplements the sampling/threshold logic in
utils/kitti_eval/eval_revised.py:get_thresholds + eval_class (K-Radar's own
official evaluator, which complete_results.txt is generated from) rather
than inventing a different AP formula, so results are comparable to
complete_results.txt's existing per-class numbers on the unstratified case.

Deliberately NOT reimplemented from eval_revised.py: difficulty
stratification (MIN_HEIGHT/MAX_OCCLUSION/MAX_TRUNCATION cutoffs on the 2D
image bbox) -- K-Radar's KITTI writer hardcodes every GT bbox to a dummy
'50 50 150 150' box (see util_pipeline.py:339 / plan §2.4), so every object
has identical 2D height and passes every difficulty's MIN_HEIGHT cutoff
identically; there is no real difficulty signal to stratify by. This
reimplementation is therefore equivalent to K-Radar's difficulty=0 (easiest)
pass with no ignored/DontCare handling, which is the only difficulty level
that receives every object -- confirmed by reading clean_data's
MAX_OCCLUSION=[0,1,2]/MAX_TRUNCATION=[0.15,0.3,0.5] gates, all of which are
trivially satisfied when occluded=0/truncated=0.00 for every object (also
hardcoded, per plan §2.4).

Usage as a library:
    from ap_metrics import compute_ap, compute_ap_grouped
    ap = compute_ap(rows_for_one_group)  # rows_for_one_group: polars.DataFrame
                                          # with is_tp/is_fp/is_fn/score for ONE
                                          # (class, condition, ...) cell already
    aps = compute_ap_grouped(row_table, group_by=["condition", "cls", "distance_bin"])
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import polars as pl

N_SAMPLE_PTS = 41


def _get_thresholds(tp_scores: np.ndarray, num_gt: int, num_sample_pts: int = N_SAMPLE_PTS) -> list[float]:
    """Verbatim port of eval_revised.py:get_thresholds (K-Radar's own
    evaluator) -- picks up to num_sample_pts score thresholds spaced evenly
    in recall, from the sorted scores of TRUE POSITIVES ONLY (not all
    predictions)."""
    if num_gt == 0:
        return []
    scores = np.sort(tp_scores)[::-1]
    current_recall = 0.0
    thresholds: list[float] = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        r_recall = (i + 2) / num_gt if i < (len(scores) - 1) else l_recall
        if ((r_recall - current_recall) < (current_recall - l_recall)) and (i < (len(scores) - 1)):
            continue
        thresholds.append(float(score))
        current_recall += 1.0 / (num_sample_pts - 1.0)
    return thresholds


def compute_ap(rows: pl.DataFrame, num_sample_pts: int = N_SAMPLE_PTS) -> dict:
    """AP for one already-filtered group (one class, one condition, one
    distance bin, whatever cells the caller wants held fixed). `rows` must
    have is_tp/is_fp/is_fn (bool) and score (float, null for FN rows) --
    ingest.py's row-table schema.

    Returns {"ap": float|None, "num_gt": int, "num_tp": int, "num_fp": int,
    "num_thresholds": int}. ap is None if num_gt==0 (undefined, not zero --
    matches complete_results.txt's own convention of only reporting AP for
    classes with GT present).
    """
    num_gt = int(rows["is_tp"].sum()) + int(rows["is_fn"].sum())
    if num_gt == 0:
        return {"ap": None, "num_gt": 0, "num_tp": 0, "num_fp": int(rows["is_fp"].sum()), "num_thresholds": 0}

    dets = rows.filter(pl.col("is_tp") | pl.col("is_fp")).sort("score", descending=True)
    det_scores = dets["score"].fill_null(0.0).to_numpy()
    det_is_tp = dets["is_tp"].to_numpy()

    tp_scores = det_scores[det_is_tp]
    thresholds = _get_thresholds(tp_scores, num_gt, num_sample_pts)
    if not thresholds:
        return {"ap": 0.0, "num_gt": num_gt, "num_tp": 0, "num_fp": int(rows["is_fp"].sum()), "num_thresholds": 0}

    # Precision/recall at each threshold: sweep score >= t over ALL
    # detections (TP+FP), same as eval_class's compute_statistics inner loop.
    order = np.argsort(-det_scores)
    sorted_scores = det_scores[order]
    sorted_is_tp = det_is_tp[order]
    cum_tp = np.cumsum(sorted_is_tp)
    cum_fp = np.cumsum(~sorted_is_tp)

    precisions = np.zeros(len(thresholds))
    for k, t in enumerate(thresholds):
        idx = np.searchsorted(-sorted_scores, -t, side="right") - 1
        if idx < 0:
            precisions[k] = 0.0
            continue
        tp_at_t, fp_at_t = cum_tp[idx], cum_fp[idx]
        precisions[k] = tp_at_t / (tp_at_t + fp_at_t) if (tp_at_t + fp_at_t) > 0 else 0.0

    # Monotonic envelope (precision can only look "better" moving toward
    # lower recall / higher score threshold) -- standard KITTI post-processing.
    for k in range(len(precisions) - 2, -1, -1):
        precisions[k] = max(precisions[k], precisions[k + 1])

    ap = float(np.sum(precisions) / num_sample_pts * 100.0)
    return {
        "ap": ap, "num_gt": num_gt, "num_tp": int(cum_tp[-1]), "num_fp": int(cum_fp[-1]),
        "num_thresholds": len(thresholds),
    }


def compute_ap_grouped(row_table: pl.DataFrame, group_by: Sequence[str]) -> pl.DataFrame:
    """AP per group-by cell (e.g. ["condition", "cls"] or ["condition", "cls",
    "distance_bin"]). Groups with num_gt==0 for that cell are omitted (not
    zero-filled) -- matches compute_ap's None convention.
    """
    group_by = list(group_by)
    out_rows = []
    for key, group in row_table.group_by(group_by, maintain_order=True):
        key = key if isinstance(key, tuple) else (key,)
        result = compute_ap(group)
        if result["ap"] is None:
            continue
        row = dict(zip(group_by, key))
        row.update(result)
        out_rows.append(row)
    return pl.DataFrame(out_rows)
