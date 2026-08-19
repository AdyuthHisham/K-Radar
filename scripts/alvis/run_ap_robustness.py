#!/usr/bin/env python3
"""RR/mRR + CRI computed directly from the unified all_preds.txt/all_gts.txt
table via ingest.py + ap_metrics.py, instead of complete_results.txt.

Why: complete_results.txt is written per-run under outputs/alvis_seq*/<condition>/,
but predictions get merged into the unified outputs/all_preds.txt /
all_gts.txt and the per-run directories are cleaned up afterward -- so
compute_robustness.py's directory-based approach has no "original" entry to
read for any sequence (confirmed empirically: no alvis_seq*/original/ dir
has an all/preds/ subdir or complete_results.txt, for any of the 10 sequences
with corrupted-condition results). This script computes AP the same way
(ap_metrics.compute_ap, validated bit-for-bit against complete_results.txt's
official KITTI AP on the unstratified case) but grouped by (sequence,
condition, cls) directly off the row-level ingestion table, so it needs
nothing but the two flat text files that actually exist.

Usage:
    python scripts/alvis/run_ap_robustness.py \\
        --outputs-dir outputs \\
        --out outputs/robustness_reports/ap_robustness.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingest import load_prediction_rows  # noqa: E402
from ap_metrics import compute_ap_grouped  # noqa: E402
from compute_robustness import compute_cri  # noqa: E402

_SEVERITY_TIERS = ("low", "medium", "high")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--force-backend", choices=["trkid", "coordinate"], default=None)
    args = ap.parse_args()

    rows = load_prediction_rows(
        preds_txt=os.path.join(args.outputs_dir, "all_preds.txt"),
        gts_txt=os.path.join(args.outputs_dir, "all_gts.txt"),
        trkid_gts_txt=os.path.join(args.outputs_dir, "all_gts_with_trkid.txt"),
        trkid_preds_txt=os.path.join(args.outputs_dir, "all_preds_with_trkid.txt"),
        fn_txt=os.path.join(args.outputs_dir, "all_fn.txt"),
        iou_threshold=args.iou_threshold,
        force_backend=args.force_backend,
    )
    print(f"Loaded {len(rows)} rows: {rows['is_tp'].sum()} TP, {rows['is_fp'].sum()} FP, {rows['is_fn'].sum()} FN",
          flush=True)

    ap_table = compute_ap_grouped(rows, ["sequence", "condition", "cls"])
    print(f"AP table: {len(ap_table)} (sequence, condition, cls) cells", flush=True)

    report: dict = {"sequences": {}}

    for seq in sorted(ap_table["sequence"].unique().to_list(), key=lambda s: int(s)):
        seq_ap = ap_table.filter(ap_table["sequence"] == seq)
        original = seq_ap.filter(seq_ap["condition"] == "original")
        if original.height == 0:
            report["sequences"][seq] = {"status": "no_original_ap", "note": "no 'original' condition rows for this sequence"}
            continue
        orig_ap_by_cls = dict(zip(original["cls"].to_list(), original["ap"].to_list()))

        conditions: dict = {}
        rb_by_effect: dict[str, dict[str, float]] = {}
        for cond in sorted(c for c in seq_ap["condition"].unique().to_list() if c != "original"):
            cond_ap = seq_ap.filter(seq_ap["condition"] == cond)
            rb_by_cls = {}
            for cls_name, cls_ap in zip(cond_ap["cls"].to_list(), cond_ap["ap"].to_list()):
                orig = orig_ap_by_cls.get(cls_name)
                if orig is not None and orig > 0:
                    rb_by_cls[cls_name] = cls_ap / orig
            if not rb_by_cls:
                continue
            mean_rb = sum(rb_by_cls.values()) / len(rb_by_cls)
            conditions[cond] = {"rb_by_cls": rb_by_cls, "mean_rb": mean_rb}

            tokens = cond.rsplit("_", 1)
            if len(tokens) == 2 and tokens[1] in _SEVERITY_TIERS:
                effect, severity = tokens
                rb_by_effect.setdefault(effect, {})[severity] = mean_rb

        overall_rb = [c["mean_rb"] for c in conditions.values()]
        mrb = sum(overall_rb) / len(overall_rb) if overall_rb else None

        cri_by_effect = {
            effect: compute_cri(sev_map)
            for effect, sev_map in rb_by_effect.items()
        }

        report["sequences"][seq] = {
            "status": "ok",
            "orig_ap_by_cls": orig_ap_by_cls,
            "conditions": conditions,
            "mRb": mrb,
            "cri_by_effect": cri_by_effect,
        }
        print(f"  seq {seq}: mRb={mrb}, {len(conditions)} conditions, {len(cri_by_effect)} effects with CRI", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
