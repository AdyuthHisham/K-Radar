#!/usr/bin/env python3
"""GT-object transition analysis: per-object churn/retention between a clean
(original) ASF run and a corrupted run, stratifiable by arbitrary keys.

Read-only diagnostic. Operates entirely on the existing prediction/GT output
already produced by predictions_to_parquet.py (outputs/all_predictions.parquet)
-- does not touch inference or training pipeline code.

Motivation (see docs/lit_survey/metrics/metrics_audit.md, Family 5 and the
frame_deletion-vs-loss_complete question): RR/mRR-style retention metrics are
aggregate-AP ratios and cannot say *which* objects a corruption cost, or
whether a corrupted run recovers detections the clean run missed (which a
pure AP-drop number cannot distinguish from "corruption cost fewer TPs than
it looks like, because it also fixed some FNs"). This module answers that at
the individual-GT-object level.

-----------------------------------------------------------------------------
Matching protocol (read this before trusting a match count)
-----------------------------------------------------------------------------
This module implements its own greedy, class-conditioned, single-IoU-threshold
BEV matching (see `match_frame`). It is NOT the official KITTI AP protocol in
utils/kitti_eval/eval_revised.py, which additionally handles ignored regions,
DontCare, per-difficulty height cutoffs, and a confidence-threshold sweep. Do
not compare this module's match counts directly against complete_results.txt's
AP numbers -- they answer different questions (per-object transition identity
vs. precision-recall-integrated accuracy) and will not numerically agree.

The BEV box construction *does* reuse the exact convention the official eval
already uses, to stay physically consistent with it:
utils/kitti_eval/kitti_common.py:get_label_anno reindexes the raw KITTI
dimensions columns (h, w, l) to (l, h, w), and
utils/kitti_eval/eval_revised.py:calculate_iou_partly's BEV branch (metric==1)
then takes bev_axes=[0, 2] of that reindexed array, i.e. (l, w) -- and
bev_axes=[0, 2] of the raw (loc_x, loc_y, loc_z) location, i.e. (loc_x, loc_z).
The resulting box vector is [loc_x, loc_z, length, width, rotation_y]. This
module reproduces that same vector directly from the (height, width, length,
loc_x, loc_y, loc_z, rotation_y) columns already in all_predictions.parquet,
rather than re-deriving the reindex from scratch, specifically to avoid
silently drifting from the convention the rest of this repo scores against.

Rotated-IoU implementation: utils/kitti_eval/rotate_iou_cpu.py, a pure NumPy
Sutherland-Hodgman polygon-clip implementation with the same call signature
and area convention as the GPU kernel it stands in for (see that file's
docstring) -- no numba/GPU dependency, safe to import anywhere.

-----------------------------------------------------------------------------
GT object identity across conditions
-----------------------------------------------------------------------------
K-Radar's raw label files carry no persistent per-object ID field. But
corrupting sensor input doesn't move or add/remove the underlying objects --
GT is identical across every condition for a given (sequence, frame) (see
merge_predictions_text.py's own note to this effect, and the identical
observation encoded in predictions_to_parquet.py's schema). This module takes
one condition's GT rows per (sequence, frame) as canonical -- preferring
"original", falling back to whichever condition has that (sequence, frame) if
"original" is absent -- and assigns `gt_obj_idx` by within-frame row order.
Every condition's predictions are matched against this single canonical GT
table, not against each condition's own (possibly duplicated) GT copy, so
`(sequence, frame, gt_obj_idx)` is a stable object key across the whole
sweep.

-----------------------------------------------------------------------------
Usage
-----------------------------------------------------------------------------
    python scripts/alvis/gt_transition_analysis.py \\
        --predictions outputs/all_predictions.parquet \\
        --clean-condition original \\
        --corrupted-condition camera_gaussian_noise_high \\
        --corrupted-condition radar_loss_partial_low \\
        --iou-threshold 0.5 \\
        --group-by effect --group-by severity --group-by cls \\
        --distance-bins 0 20 40 72 \\
        --out outputs/transition_report.json

As a library:

    from gt_transition_analysis import (
        build_match_table, build_transition_table, aggregate_transition_rates,
    )
    match_table = build_match_table(df, iou_threshold=0.5)
    transitions = build_transition_table(match_table, "original", "camera_gaussian_noise_high")
    rates = aggregate_transition_rates(transitions, group_by=["cls", "distance_bin"])
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Sequence

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "utils", "kitti_eval"))
from rotate_iou_cpu import rotate_iou_gpu_eval  # noqa: E402  (path set above)

# NOTE: ingest.py imports match_frame_pairs from this module (its coordinate
# fallback backend) -- importing ingest at module level here would be
# circular. load_prediction_rows is imported lazily inside main() instead,
# by which point both modules are fully loaded.

DEFAULT_IOU_THRESHOLD = 0.5

# Known severity tokens, used to split a condition string like
# "camera_gaussian_noise_high" into (modality, effect, severity). Conditions
# without a trailing severity token (e.g. "original", "loss_complete_zero")
# get severity=None -- see parse_condition's docstring.
_SEVERITY_TOKENS = {"low", "medium", "high"}
_MODALITY_TOKENS = {"camera", "radar", "lidar"}


def parse_condition(condition: str) -> dict[str, str | None]:
    """Split a condition string into (modality, effect, severity).

    Convention established by scripts/alvis/gen_single_effect_configs.py:
    "{modality}_{effect}_{severity}", e.g. "radar_noise_induced_shifts_medium".
    "original" (the clean baseline) and unconditional effects like
    "loss_complete_zero" don't carry a severity token; those become
    severity=None. Effect name is whatever sits between the modality prefix
    (if present) and the severity suffix (if present) -- not validated
    against a fixed taxonomy list, so an unrecognized naming pattern degrades
    to effect=condition, modality=None, severity=None rather than raising.
    """
    if condition == "original":
        return {"modality": None, "effect": "original", "severity": None}

    tokens = condition.split("_")
    severity = None
    if tokens and tokens[-1] in _SEVERITY_TOKENS:
        severity = tokens[-1]
        tokens = tokens[:-1]

    modality = None
    if tokens and tokens[0] in _MODALITY_TOKENS:
        modality = tokens[0]
        tokens = tokens[1:]

    effect = "_".join(tokens) if tokens else condition
    return {"modality": modality, "effect": effect, "severity": severity}


def _bev_box_array(df: pl.DataFrame) -> np.ndarray:
    """(N, 5) array [loc_x, loc_z, length, width, rotation_y], df's row order.

    See module docstring's "Matching protocol" section for why this exact
    column selection, not a different one.
    """
    if len(df) == 0:
        return np.zeros((0, 5), dtype=np.float64)
    return df.select("loc_x", "loc_z", "length", "width", "rotation_y").to_numpy()


def match_frame_pairs(
    gt: pl.DataFrame, pred: pl.DataFrame, iou_threshold: float,
) -> list[tuple[int, int, float]]:
    """Greedy one-to-one class-conditioned BEV IoU matching for one frame.

    Returns a list of (gt_idx, pred_idx, iou) triples -- gt_idx/pred_idx are
    positional indices into `gt`/`pred` (row order as passed in). Predictions
    are considered highest-score first; each prediction can match at most one
    GT box (the highest-IoU eligible one still unmatched), and each GT box
    can be matched at most once. Cross-class pairs are never eligible.
    Unmatched GT/pred rows are simply absent from the returned list -- caller
    derives FN (unmatched gt_idx) / FP (unmatched pred_idx) by set
    difference against range(len(gt))/range(len(pred)).
    """
    n_gt, n_pred = len(gt), len(pred)
    if n_gt == 0 or n_pred == 0:
        return []

    gt_boxes = _bev_box_array(gt)
    pred_boxes = _bev_box_array(pred)
    gt_cls = gt["cls"].to_numpy()
    pred_cls = pred["cls"].to_numpy()
    pred_score = pred["score"].fill_null(0.0).to_numpy()

    iou = rotate_iou_gpu_eval(gt_boxes, pred_boxes, criterion=-1)  # (n_gt, n_pred)
    class_ok = gt_cls[:, None] == pred_cls[None, :]
    iou = np.where(class_ok, iou, 0.0)

    matched_gt = np.zeros(n_gt, dtype=bool)
    pairs: list[tuple[int, int, float]] = []
    for p in np.argsort(-pred_score):
        eligible = np.where((iou[:, p] >= iou_threshold) & ~matched_gt)[0]
        if eligible.size == 0:
            continue
        best_gt = eligible[np.argmax(iou[eligible, p])]
        matched_gt[best_gt] = True
        pairs.append((int(best_gt), int(p), float(iou[best_gt, p])))
    return pairs


def match_frame(gt: pl.DataFrame, pred: pl.DataFrame, iou_threshold: float) -> np.ndarray:
    """Thin wrapper over match_frame_pairs for existing callers: returns an
    (len(gt),) int64 array, 1 = matched (TP), 0 = unmatched (FN), collapsing
    away the pred_idx/iou detail match_frame_pairs carries.
    """
    matched = np.zeros(len(gt), dtype=np.int64)
    for gt_idx, _pred_idx, _iou in match_frame_pairs(gt, pred, iou_threshold):
        matched[gt_idx] = 1
    return matched


def _canonical_gt(df: pl.DataFrame) -> pl.DataFrame:
    """One GT row per (sequence, frame, gt_obj_idx), from a single condition
    per frame (prefer "original"; else whichever condition has that frame
    first, by condition-name sort order, for determinism).

    See module docstring's "GT object identity across conditions" section.
    """
    gt = df.filter(pl.col("source") == "gt")

    # Pick one condition per (sequence, frame): "original" if it has GT rows
    # there, else the alphabetically-first condition that does (deterministic,
    # arbitrary otherwise -- content is identical across conditions by
    # construction, so which one wins doesn't affect correctness).
    chosen_condition = (
        gt.select("sequence", "frame", "condition")
        .unique()
        .with_columns(
            pl.when(pl.col("condition") == "original").then(0).otherwise(1).alias("_priority")
        )
        .sort(["sequence", "frame", "_priority", "condition"])
        .group_by(["sequence", "frame"], maintain_order=True)
        .first()
        .select("sequence", "frame", "condition")
    )

    canonical = gt.join(chosen_condition, on=["sequence", "frame", "condition"], how="inner")
    canonical = canonical.sort(["sequence", "frame"])
    canonical = canonical.with_columns(
        pl.int_range(pl.len()).over(["sequence", "frame"]).alias("gt_obj_idx")
    )
    return canonical


def build_match_table(df: pl.DataFrame, iou_threshold: float = DEFAULT_IOU_THRESHOLD) -> pl.DataFrame:
    """For every (sequence, frame, condition) in `df`, match that condition's
    predictions against the canonical per-(sequence, frame) GT table.

    Returns one row per (sequence, frame, condition, gt_obj_idx): the GT
    object's class, its location (for downstream distance-bin stratification),
    and `matched` (1 = TP, 0 = FN) under that condition.

    `df` must have the schema written by predictions_to_parquet.py: sequence,
    condition, frame, source ("pred"/"gt"), cls, height, width, length,
    loc_x, loc_y, loc_z, rotation_y, score.
    """
    canonical_gt = _canonical_gt(df)
    preds = df.filter(pl.col("source") == "pred")
    conditions = df.select("condition").unique().to_series().to_list()

    rows: list[dict[str, Any]] = []
    for (sequence, frame), gt_frame in canonical_gt.group_by(["sequence", "frame"], maintain_order=True):
        gt_frame = gt_frame.sort("gt_obj_idx")
        for condition in conditions:
            pred_frame = preds.filter(
                (pl.col("sequence") == sequence)
                & (pl.col("frame") == frame)
                & (pl.col("condition") == condition)
            )
            matched = match_frame(gt_frame, pred_frame, iou_threshold)
            for i in range(len(gt_frame)):
                rows.append({
                    "sequence": sequence, "frame": frame, "condition": condition,
                    "gt_obj_idx": int(gt_frame["gt_obj_idx"][i]),
                    "cls": gt_frame["cls"][i],
                    "loc_x": float(gt_frame["loc_x"][i]), "loc_z": float(gt_frame["loc_z"][i]),
                    "matched": int(matched[i]),
                })

    return pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={"sequence": pl.Utf8, "frame": pl.Utf8, "condition": pl.Utf8,
                "gt_obj_idx": pl.Int64, "cls": pl.Utf8, "loc_x": pl.Float64,
                "loc_z": pl.Float64, "matched": pl.Int64}
    )


def build_match_table_from_ingest(row_table: pl.DataFrame) -> pl.DataFrame:
    """Adapter: build the same (sequence, frame, condition, gt_obj_idx, cls,
    loc_x, loc_z, matched) schema build_match_table produces, but from
    ingest.py's load_prediction_rows() output instead of re-matching from
    scratch. See plan §2.5 -- ingest.py already does the matching (via
    either the trk_id backend or this module's own match_frame_pairs used
    as its coordinate fallback), so this just reshapes that result rather
    than duplicating the match.

    GT object identity: ingest.py's row table carries either `trk_id`
    (trk_id backend) or `matched_gt_row_id` (coordinate fallback -- a row
    id into the condition-invariant all_gts.txt, already stable across
    conditions for the same reason trk_id is) -- whichever is present is
    used as gt_obj_idx. FP rows (no matched GT) are dropped: transitions are
    about GT-object fates, and an FP has no GT object to attach a fate to.
    """
    id_col = "trk_id" if "trk_id" in row_table.columns else "matched_gt_row_id"

    tp = row_table.filter(pl.col("is_tp")).select(
        "sequence", "frame", "condition", pl.col(id_col).alias("gt_obj_idx"),
        "cls", pl.col("gt_loc_x").alias("loc_x"), pl.col("gt_loc_z").alias("loc_z"),
    ).with_columns(pl.lit(1).alias("matched"))

    fn = row_table.filter(pl.col("is_fn")).select(
        "sequence", "frame", "condition", pl.col(id_col).alias("gt_obj_idx"),
        "cls", pl.col("gt_loc_x").alias("loc_x"), pl.col("gt_loc_z").alias("loc_z"),
    ).with_columns(pl.lit(0).alias("matched"))

    return pl.concat([tp, fn], how="vertical_relaxed").filter(pl.col("gt_obj_idx").is_not_null())


def classify_transition(m_clean: int, m_corr: int) -> str:
    """The four-way classification, per the module spec:
        stable                    : M_clean=1, M_corr=1
        corruption_induced_miss   : M_clean=1, M_corr=0
        corruption_induced_recovery: M_clean=0, M_corr=1
        baseline_failure          : M_clean=0, M_corr=0
    """
    if m_clean == 1 and m_corr == 1:
        return "stable"
    if m_clean == 1 and m_corr == 0:
        return "corruption_induced_miss"
    if m_clean == 0 and m_corr == 1:
        return "corruption_induced_recovery"
    return "baseline_failure"


def build_transition_table(
    match_table: pl.DataFrame, clean_condition: str, corrupted_condition: str,
) -> pl.DataFrame:
    """Join one corrupted condition's match results against the clean
    condition's, per (sequence, frame, gt_obj_idx), and classify each GT
    object's transition.

    A GT object present under one condition but not the other (e.g. that
    condition's run never reached the eval step for that frame -- see
    compute_robustness.py's "no_predictions" handling for the parallel case
    in the AP-based tool) is reported separately as `missing_one_side`, NOT
    silently dropped and NOT forced into a transition category -- inspect it
    before trusting the transition counts for affected (sequence, frame)
    pairs.
    """
    clean = match_table.filter(pl.col("condition") == clean_condition).rename({"matched": "m_clean"})
    corr = match_table.filter(pl.col("condition") == corrupted_condition).rename({"matched": "m_corr"})

    joined = clean.join(
        corr, on=["sequence", "frame", "gt_obj_idx"], how="full", suffix="_corr",
    )

    missing_one_side = joined.filter(pl.col("m_clean").is_null() | pl.col("m_corr").is_null())
    if len(missing_one_side) > 0:
        print(
            f"WARNING: {len(missing_one_side)} GT object(s) present under only one of "
            f"'{clean_condition}' / '{corrupted_condition}' (missing eval output for that "
            f"condition on that frame) -- excluded from the transition table. Inspect "
            f"`missing_one_side` before trusting rates for affected frames.",
            file=sys.stderr,
        )
    joined = joined.filter(pl.col("m_clean").is_not_null() & pl.col("m_corr").is_not_null())

    # `full` join duplicates the coalesce-able key/attribute columns with a
    # "_corr" suffix; sequence/frame/gt_obj_idx are join keys (identical on
    # both sides for matched rows) and cls/loc_x/loc_z come from GT, which is
    # identical across conditions by construction (see _canonical_gt) -- so
    # the un-suffixed columns are authoritative and the _corr duplicates of
    # them are dropped, not the different-valued m_clean/m_corr pair.
    joined = joined.select(
        "sequence", "frame", "gt_obj_idx", "cls", "loc_x", "loc_z", "m_clean", "m_corr",
    )

    joined = joined.with_columns(
        pl.struct(["m_clean", "m_corr"])
        .map_elements(lambda s: classify_transition(s["m_clean"], s["m_corr"]), return_dtype=pl.Utf8)
        .alias("transition")
    )
    joined = joined.with_columns(
        pl.lit(clean_condition).alias("clean_condition"),
        pl.lit(corrupted_condition).alias("corrupted_condition"),
    )
    return joined


def add_distance_bin(df: pl.DataFrame, bin_edges: Sequence[float]) -> pl.DataFrame:
    """Add a `distance_bin` column from GT range (sqrt(loc_x^2 + loc_z^2), the
    same BEV axes the matching itself uses -- see module docstring), bucketed
    by `bin_edges` (e.g. [0, 20, 40, 72] -> "[0,20)", "[20,40)", "[40,72)";
    a range at or beyond the last edge, or NaN, becomes "out_of_range").
    """
    edges = list(bin_edges)
    labels = [f"[{edges[i]},{edges[i+1]})" for i in range(len(edges) - 1)]

    range_m = (pl.col("loc_x") ** 2 + pl.col("loc_z") ** 2).sqrt()
    expr = pl.lit("out_of_range")
    for lo, hi, label in zip(edges[:-1][::-1], edges[1:][::-1], labels[::-1]):
        expr = pl.when((range_m >= lo) & (range_m < hi)).then(pl.lit(label)).otherwise(expr)
    return df.with_columns(expr.alias("distance_bin"))


def add_condition_strata(df: pl.DataFrame, condition_col: str = "corrupted_condition") -> pl.DataFrame:
    """Add `modality`, `effect`, `severity` columns parsed from a condition
    column (see parse_condition)."""
    parsed = [parse_condition(c) for c in df[condition_col].to_list()]
    return df.with_columns(
        pl.Series("modality", [p["modality"] for p in parsed]),
        pl.Series("effect", [p["effect"] for p in parsed]),
        pl.Series("severity", [p["severity"] for p in parsed]),
    )


def aggregate_transition_rates(transitions: pl.DataFrame, group_by: Sequence[str]) -> pl.DataFrame:
    """Group `transitions` (the output of build_transition_table, optionally
    enriched with add_distance_bin / add_condition_strata / any other
    externally-joined stratification columns -- e.g. occlusion proxy bin or
    weather/time/environment, which are not derivable from the current
    all_predictions.parquet schema and must be left-joined in by the caller
    on (sequence, frame[, gt_obj_idx]) before calling this) by `group_by`,
    and compute the raw 2x2 counts plus derived rates per cell.

    Raises AssertionError if, for any cell, N_stable + N_miss + N_recover +
    N_fail does not equal that cell's total GT-object row count -- this
    should be structurally impossible given `transition` is computed by
    classify_transition on every row, but is checked explicitly per the
    module spec rather than assumed.
    """
    group_by = list(group_by)
    counted = (
        transitions.group_by(group_by + ["transition"], maintain_order=True)
        .agg(pl.len().alias("n"))
        .pivot(values="n", index=group_by, on="transition")
        .fill_null(0)
    )
    for col in ("stable", "corruption_induced_miss", "corruption_induced_recovery", "baseline_failure"):
        if col not in counted.columns:
            counted = counted.with_columns(pl.lit(0).alias(col))

    counted = counted.rename({
        "stable": "n_stable",
        "corruption_induced_miss": "n_miss",
        "corruption_induced_recovery": "n_recover",
        "baseline_failure": "n_fail",
    })

    total_by_cell = transitions.group_by(group_by, maintain_order=True).agg(pl.len().alias("total"))
    counted = counted.join(total_by_cell, on=group_by, how="left")

    computed_total = counted["n_stable"] + counted["n_miss"] + counted["n_recover"] + counted["n_fail"]
    mismatch = counted.filter(computed_total != pl.col("total"))
    assert len(mismatch) == 0, (
        f"Sanity check failed: N_stable+N_miss+N_recover+N_fail != total GT object count "
        f"for {len(mismatch)} cell(s):\n{mismatch}"
    )

    stable_plus_miss = pl.col("n_stable") + pl.col("n_miss")
    recover_plus_fail = pl.col("n_recover") + pl.col("n_fail")

    counted = counted.with_columns(
        pl.when(stable_plus_miss > 0).then(pl.col("n_miss") / stable_plus_miss).otherwise(None)
            .alias("corruption_induced_miss_rate"),
        pl.when(stable_plus_miss > 0).then(pl.col("n_stable") / stable_plus_miss).otherwise(None)
            .alias("stable_detection_rate"),
        pl.when(recover_plus_fail > 0).then(pl.col("n_recover") / recover_plus_fail).otherwise(None)
            .alias("corruption_induced_recovery_rate"),
        (pl.col("n_miss") + pl.col("n_recover")).alias("churn"),
    )
    counted = counted.with_columns(
        (pl.col("churn") / pl.col("total")).alias("churn_rate"),
        # Spec's literal formula: (N_stable+N_recover) / (N_stable+N_miss). Note the
        # denominator is NOT the same population as the numerator (it's the
        # clean-detected count, not the corrupted-detected count) -- this is not a
        # bounded-in-[0,1] retention fraction in general (it can exceed 1 when
        # recoveries outnumber the clean-miss population), by construction of the
        # formula as specified, not a bug in this implementation.
        pl.when(stable_plus_miss > 0)
            .then((pl.col("n_stable") + pl.col("n_recover")) / stable_plus_miss)
            .otherwise(None)
            .alias("net_retention"),
    )

    return counted.select(
        *group_by, "n_stable", "n_miss", "n_recover", "n_fail", "total",
        "corruption_induced_miss_rate", "stable_detection_rate",
        "corruption_induced_recovery_rate", "churn", "churn_rate", "net_retention",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs-dir", default="outputs",
                     help="directory containing all_preds.txt/all_gts.txt (and, if present, "
                          "all_*_with_trkid.txt/all_fn.txt, preferred automatically -- see ingest.py)")
    ap.add_argument("--clean-condition", default="original")
    ap.add_argument("--corrupted-condition", action="append", required=True, dest="corrupted_conditions")
    ap.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    ap.add_argument("--group-by", action="append", default=[],
                     help="stratification column(s); repeatable. Built-ins: cls, distance_bin "
                          "(requires --distance-bins), modality, effect, severity, corrupted_condition. "
                          "Any other column must already exist on the transition table (e.g. joined in "
                          "externally for occlusion/weather/time/environment).")
    ap.add_argument("--distance-bins", type=float, nargs="+", default=None,
                     help="bin edges in meters, e.g. 0 20 40 72; required if 'distance_bin' is in --group-by")
    ap.add_argument("--out", required=True, help="output path for the JSON report")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ingest import load_prediction_rows  # local import; see note near top of file

    row_table = load_prediction_rows(
        preds_txt=os.path.join(args.outputs_dir, "all_preds.txt"),
        gts_txt=os.path.join(args.outputs_dir, "all_gts.txt"),
        trkid_gts_txt=os.path.join(args.outputs_dir, "all_gts_with_trkid.txt"),
        trkid_preds_txt=os.path.join(args.outputs_dir, "all_preds_with_trkid.txt"),
        fn_txt=os.path.join(args.outputs_dir, "all_fn.txt"),
        iou_threshold=args.iou_threshold,
    )
    match_table = build_match_table_from_ingest(row_table)

    all_transitions = []
    for corrupted_condition in args.corrupted_conditions:
        t = build_transition_table(match_table, args.clean_condition, corrupted_condition)
        if "distance_bin" in args.group_by:
            if args.distance_bins is None:
                raise SystemExit("--distance-bins is required when 'distance_bin' is in --group-by")
            t = add_distance_bin(t, args.distance_bins)
        if any(k in args.group_by for k in ("modality", "effect", "severity")):
            t = add_condition_strata(t, condition_col="corrupted_condition")
        all_transitions.append(t)

    transitions = pl.concat(all_transitions, how="diagonal_relaxed")

    group_by = list(args.group_by) if args.group_by else ["corrupted_condition"]
    if "corrupted_condition" not in group_by:
        group_by = ["corrupted_condition"] + group_by
    rates = aggregate_transition_rates(transitions, group_by=group_by)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rates.write_json(args.out)
    print(f"Wrote {args.out} ({len(rates)} cells across {len(args.corrupted_conditions)} corrupted condition(s))")


if __name__ == "__main__":
    main()
