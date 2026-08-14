#!/usr/bin/env python3
"""Compute the Rb^c / mRb robustness metric from a pair (or set) of ASF runs.

Rb^c = P_s^c / P_original  (AP under corruption pattern c at severity s,
relative to the original/uncorrupted run's AP on the same metric), from
"Benchmarking Robustness of AI-Enabled Multi-sensor Fusion Systems" section
3.3 (docs/lit_survey/noise_addition/markdown/BenchmarkingRobustness.md).
mRb = mean(Rb^c) over the corruption patterns supplied.

Parses each run's complete_results.txt (written by
pipelines/pipeline_detection_v1_0.py:validate_kitti_conditional, repeated
blocks of the form:

    Conf thr: 0.3, Condition: all
    cls: sed
    iou: 0.7 0.5 0.3
    bev: v1 v2 v3
    3d  :v1 v2 v3

    (blank line separates blocks)

Note the asymmetric label spacing -- "bev: " has a space after the colon,
"3d  :" does not -- so this parser matches on the KEY prefix before the first
colon, not a literal "key: " string. No existing script in this repo parses
this format; complete_results.txt only has writers (pipeline_detection_v1_0.py,
main_fusion_eval_0.py), confirmed by repo-wide search.

A run whose complete_results.txt is missing entirely (the model aborted before
reaching the eval step -- expected for some conditions, e.g. radar/lidar
frame_deletion at high severity, loss_complete_zero) is represented explicitly
in the output as status "no_predictions", not silently dropped and not a
script-crashing error.

Usage:
    python scripts/alvis/compute_robustness.py \\
        --seq 46 \\
        --original-dir outputs/alvis_seq46/original \\
        --corrupted-dir outputs/alvis_seq46/camera_gaussian_noise_high \\
        --corrupted-dir outputs/alvis_seq46/radar_loss_partial_low \\
        --out outputs/alvis_seq46/robustness_report.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from typing import Any


def find_complete_results(run_dir: str) -> str | None:
    """Locate complete_results.txt inside a run directory.

    Layout: <run_dir>/exp_<timestamp>_.../test_kitti/<epoch>/<conf_thr>/complete_results.txt
    A run directory can accumulate multiple exp_* subdirs across retries (see
    the seq46 pilot's stale-directory issue this session) -- pick the most
    recently modified match.
    """
    if not os.path.isdir(run_dir):
        return None
    matches = glob.glob(os.path.join(run_dir, "exp_*", "test_kitti", "*", "*", "complete_results.txt"))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def parse_complete_results(path: str) -> list[dict[str, Any]]:
    """Parse complete_results.txt into a list of per-(conf_thr, condition, class) blocks."""
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    header_re = re.compile(r"^Conf thr:\s*([\d.]+),\s*Condition:\s*(\S+)\s*$")

    with open(path) as f:
        for raw_line in f:
            line = raw_line.rstrip("\n").rstrip()
            if not line:
                if current is not None:
                    blocks.append(current)
                    current = None
                continue
            m = header_re.match(line)
            if m:
                if current is not None:
                    blocks.append(current)
                current = {"conf_thr": float(m.group(1)), "condition": m.group(2)}
                continue
            if current is None:
                continue
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            if key == "cls":
                current["cls"] = value.strip()
            elif key == "iou":
                current["iou"] = [float(x) for x in value.split()]
            elif key == "bev":
                current["bev"] = [float(x) for x in value.split()]
            elif key == "3d":
                current["3d"] = [float(x) for x in value.split()]

    if current is not None:
        blocks.append(current)
    return blocks


def _rb(original_val: float, corrupted_val: float) -> float | None:
    if original_val == 0.0:
        return None
    return corrupted_val / original_val


def compute_condition_report(original_blocks: list[dict], corrupted_blocks: list[dict],
                              condition_name: str) -> dict[str, Any]:
    """Compute Rb^c rows for one corrupted condition against the original blocks."""
    original_by_key = {(b["conf_thr"], b["condition"], b["cls"]): b for b in original_blocks}

    rows = []
    for cb in corrupted_blocks:
        key = (cb["conf_thr"], cb["condition"], cb["cls"])
        ob = original_by_key.get(key)
        if ob is None:
            continue
        for view in ("bev", "3d"):
            o_vals, c_vals = ob.get(view), cb.get(view)
            o_ious, c_ious = ob.get("iou"), cb.get("iou")
            if not (o_vals and c_vals and o_ious and c_ious):
                continue
            for o_iou, o_val in zip(o_ious, o_vals):
                # Align by IoU value (not blind index) in case the two runs'
                # iou lists ever differ in order/count.
                if o_iou not in c_ious:
                    continue
                c_val = c_vals[c_ious.index(o_iou)]
                rb = _rb(o_val, c_val)
                rows.append({
                    "conf_thr": cb["conf_thr"], "condition": cb["condition"],
                    "cls": cb["cls"], "view": view, "iou": o_iou,
                    "original_ap": o_val, "corrupted_ap": c_val, "rb": rb,
                })

    rb_values = [r["rb"] for r in rows if r["rb"] is not None]
    return {
        "condition_name": condition_name,
        "status": "ok" if rows else "no_matching_rows",
        "rows": rows,
        "mean_rb": statistics.mean(rb_values) if rb_values else None,
        "min_rb": min(rb_values) if rb_values else None,
        "max_rb": max(rb_values) if rb_values else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True, help="sequence number, e.g. 46 or 1")
    ap.add_argument("--original-dir", required=True,
                     help="run directory for the original/uncorrupted baseline")
    ap.add_argument("--corrupted-dir", action="append", required=True,
                     help="run directory for a corrupted condition (repeatable)")
    ap.add_argument("--out", required=True, help="output path for the JSON report")
    args = ap.parse_args()

    original_results_path = find_complete_results(args.original_dir)
    if original_results_path is None:
        raise SystemExit(
            f"No complete_results.txt found under {args.original_dir} -- "
            "the original baseline run must succeed before computing robustness."
        )
    original_blocks = parse_complete_results(original_results_path)

    conditions = []
    for corrupted_dir in args.corrupted_dir:
        condition_name = os.path.basename(os.path.normpath(corrupted_dir))
        results_path = find_complete_results(corrupted_dir)
        if results_path is None:
            conditions.append({
                "condition_name": condition_name,
                "status": "no_predictions",
                "rows": [], "mean_rb": None, "min_rb": None, "max_rb": None,
                "note": "no complete_results.txt found -- the model likely aborted "
                        "before reaching the eval step for this condition (expected "
                        "for some high-severity blackout conditions, not necessarily "
                        "a driver bug)",
            })
            continue
        corrupted_blocks = parse_complete_results(results_path)
        conditions.append(compute_condition_report(original_blocks, corrupted_blocks, condition_name))

    overall_rb = [c["mean_rb"] for c in conditions if c["mean_rb"] is not None]
    report = {
        "sequence": args.seq,
        "original_dir": args.original_dir,
        "original_results_path": original_results_path,
        "n_conditions": len(conditions),
        "n_conditions_with_results": sum(1 for c in conditions if c["status"] == "ok"),
        "n_conditions_no_predictions": sum(1 for c in conditions if c["status"] == "no_predictions"),
        "mean_mRb": statistics.mean(overall_rb) if overall_rb else None,
        "conditions": conditions,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    md_path = os.path.splitext(args.out)[0] + ".md"
    _write_markdown(report, md_path)

    print(f"Wrote {args.out}")
    print(f"Wrote {md_path}")
    print(f"mean mRb across {report['n_conditions_with_results']}/{report['n_conditions']} "
          f"conditions with results: {report['mean_mRb']}")
    if report["n_conditions_no_predictions"]:
        print(f"{report['n_conditions_no_predictions']} condition(s) had no predictions "
              f"(model aborted) -- see report for names.")


def _write_markdown(report: dict, path: str) -> None:
    lines = [
        f"# Robustness report — sequence {report['sequence']}",
        "",
        f"Original run: `{report['original_dir']}`",
        "",
        f"Conditions evaluated: {report['n_conditions']} "
        f"({report['n_conditions_with_results']} with results, "
        f"{report['n_conditions_no_predictions']} with no predictions / model abort)",
        "",
        f"**mean mRb across conditions with results: "
        f"{report['mean_mRb']:.3f}**" if report["mean_mRb"] is not None
        else "**mean mRb: n/a (no conditions produced results)**",
        "",
        "| Condition | Status | mean Rb^c | min Rb^c | max Rb^c |",
        "|---|---|---|---|---|",
    ]
    for c in sorted(report["conditions"], key=lambda c: c["condition_name"]):
        mean_s = f"{c['mean_rb']:.3f}" if c["mean_rb"] is not None else "—"
        min_s = f"{c['min_rb']:.3f}" if c["min_rb"] is not None else "—"
        max_s = f"{c['max_rb']:.3f}" if c["max_rb"] is not None else "—"
        lines.append(f"| {c['condition_name']} | {c['status']} | {mean_s} | {min_s} | {max_s} |")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
