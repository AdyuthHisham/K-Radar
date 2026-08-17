#!/usr/bin/env python3
"""Build ONE gallery.html sampling N random (sequence, frame) pairs from
across a multi-sequence sweep's full run tree, covering every corrupted
condition -- for spot-checking a large sweep without rendering every frame
of every sequence (see build_sequence_full_gallery.py for the per-sequence,
every-frame version).

Reuses render_original_frame/render_corrupted_frame/find_kitti_dirs/
find_sensor_dump from build_sequence_full_gallery.py unchanged.

Sampling is deterministic: sorted(sequences) x frame ids 000000..(frames_per_seq-1),
random.Random(seed).sample(pool, n_frames) -- same algorithm used to pick this
session's 20-frame sample from the 10-sequence x 10-frame sweep (seed 42).

Usage:
    python scripts/alvis/build_random_sample_gallery.py \\
        --sequences 2,7,8,9,15,16,18,41,48,55 \\
        --frames-per-seq 10 \\
        --n-frames 20 --seed 42 \\
        --run-root-template outputs/alvis_seq{seq} \\
        --out outputs/random_sample_gallery.html
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "datasets", "effects"))
sys.path.insert(0, os.path.join(_REPO, "scripts", "alvis"))

import pickle
from kitti_boxes import parse_kitti_boxes  # noqa: E402
from build_sequence_full_gallery import (  # noqa: E402
    find_kitti_dirs, find_sensor_dump, render_original_frame, render_corrupted_frame, parse_meta,
    load_condition_params,
)


def pick_sample(sequences: list[int], frames_per_seq: int, n_frames: int, seed: int) -> list[tuple[int, str]]:
    frame_ids = [f"{i:06d}" for i in range(frames_per_seq)]
    pool = [(s, f) for s in sorted(sequences) for f in frame_ids]
    rng = random.Random(seed)
    return sorted(rng.sample(pool, min(n_frames, len(pool))))


def load_calib(seq: int) -> dict:
    path = os.path.join(_REPO, "resources", "cam_calib", "T_params_seq", str(seq))
    with open(path, "rb") as f:
        return pickle.load(f)["front0"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequences", required=True, help="comma-separated sequence numbers")
    ap.add_argument("--frames-per-seq", type=int, default=10)
    ap.add_argument("--n-frames", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-root-template", default="outputs/alvis_seq{seq}")
    ap.add_argument("--raw-cam-dir-template", default=None,
                     help="default: data/{seq}/cam-front")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sequences = [int(s) for s in args.sequences.split(",")]
    sample = pick_sample(sequences, args.frames_per_seq, args.n_frames, args.seed)
    print(f"Sample ({len(sample)} frames): {sample}")

    calibs = {seq: load_calib(seq) for seq in {s for s, _ in sample}}

    out_dir = os.path.dirname(os.path.abspath(args.out))
    render_dir = os.path.join(out_dir, "_gallery_render_tmp_sample")
    os.makedirs(render_dir, exist_ok=True)

    # Per-sequence: original preds/gts dir, and a source dump dir for the
    # shared original-panel backups (same rationale as build_sequence_full_gallery.py --
    # the `original` run has no --dump-sensors dump of its own).
    per_seq_state: dict[int, dict] = {}
    for seq in {s for s, _ in sample}:
        run_root = args.run_root_template.format(seq=seq)
        original_dir = os.path.join(run_root, "original")
        original_kitti = find_kitti_dirs(original_dir)
        if original_kitti is None:
            raise SystemExit(f"No original-run KITTI results for sequence {seq} under {original_dir}")
        original_preds_dir, gts_dir = original_kitti

        condition_dirs = sorted(
            d for d in os.listdir(run_root)
            if os.path.isdir(os.path.join(run_root, d)) and d != "original" and not d.startswith("_")
        )
        # Candidate source dumps for this sequence's shared ORIGINAL-panel
        # backups (the `original` run has no --dump-sensors dump of its own).
        # A condition's sensor_dump/ directory can exist yet be empty or
        # missing specific frames if that condition's run aborted before
        # writing anything (e.g. radar/lidar frame_deletion at high severity
        # crashing the sparse-conv backbone on frame 0) -- don't just check
        # directory existence, verify per-frame below.
        source_dump_candidates = [
            d for cond in condition_dirs
            if (d := find_sensor_dump(os.path.join(run_root, cond))) is not None
        ]

        per_seq_state[seq] = {
            "run_root": run_root, "original_preds_dir": original_preds_dir, "gts_dir": gts_dir,
            "condition_dirs": condition_dirs, "source_dump_candidates": source_dump_candidates,
        }

    all_conditions = sorted({c for st in per_seq_state.values() for c in st["condition_dirs"]})

    raw_cam_dir_template = args.raw_cam_dir_template or os.path.join(_REPO, "data", "{seq}", "cam-front")

    REQUIRED_BACKUP_SUFFIXES = ("rdr_sparse_clean.npy", "ldr64_clean.npy", "meta.txt")

    def _find_verified_source_dump(seq: int, fid: str) -> str | None:
        stem = f"{fid}_seq{seq}"
        for dump in per_seq_state[seq]["source_dump_candidates"]:
            if all(os.path.exists(os.path.join(dump, f"{stem}_{suf}")) for suf in REQUIRED_BACKUP_SUFFIXES):
                return dump
        return None

    originals: dict[tuple[int, str], dict] = {}
    for seq, fid in sample:
        st = per_seq_state[seq]
        source_dump = _find_verified_source_dump(seq, fid)
        if source_dump is None:
            print(f"WARNING: no condition has a usable sensor_dump backup for seq{seq}/{fid} "
                  f"(checked {len(st['source_dump_candidates'])} candidates) -- skipping original render")
            continue
        stem = f"{fid}_seq{seq}"
        gt = parse_kitti_boxes(os.path.join(st["gts_dir"], f"{fid}.txt"))
        preds = parse_kitti_boxes(os.path.join(st["original_preds_dir"], f"{fid}.txt"))
        calib = calibs[seq]
        raw_cam_dir = raw_cam_dir_template.format(seq=seq)
        raw_cam_dir = raw_cam_dir if os.path.isdir(raw_cam_dir) else None
        key = f"seq{seq}_{fid}"
        originals[(seq, fid)] = render_original_frame(
            key, stem, source_dump, render_dir, preds, gt,
            calib["radar2image"], calib["img_aug_matrix"], raw_cam_dir)
        print(f"original seq{seq}/{fid}: gt={originals[(seq, fid)]['gt_count']} "
              f"pred={originals[(seq, fid)]['pred_count']}")

    conditions_data = []
    for cond in all_conditions:
        frames = {}
        for seq, fid in sample:
            st = per_seq_state[seq]
            if cond not in st["condition_dirs"]:
                continue
            cond_dir = os.path.join(st["run_root"], cond)
            dump = find_sensor_dump(cond_dir)
            kitti = find_kitti_dirs(cond_dir)
            if dump is None or kitti is None:
                continue
            preds_dir, _ = kitti
            stem = f"{fid}_seq{seq}"
            meta_path = os.path.join(dump, f"{stem}_meta.txt")
            if not os.path.exists(meta_path):
                continue
            meta = parse_meta(meta_path)
            gt = parse_kitti_boxes(os.path.join(st["gts_dir"], f"{fid}.txt"))
            preds = parse_kitti_boxes(os.path.join(preds_dir, f"{fid}.txt"))
            calib = calibs[seq]
            key = f"seq{seq}_{fid}"
            frames[(seq, fid)] = render_corrupted_frame(
                key, stem, dump, render_dir, cond, preds, gt, meta,
                calib["radar2image"], calib["img_aug_matrix"])
        status = "ok" if frames else "no_predictions"
        conditions_data.append({"name": cond, "status": status, "frames": frames})
        print(f"{cond}: {len(frames)}/{len(sample)} frame(s) rendered")

    html = _build_html(sample, originals, conditions_data)
    with open(args.out, "w") as f:
        f.write(html)

    shutil.rmtree(render_dir, ignore_errors=True)
    print(f"\nGallery written: {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


def _build_html(sample: list[tuple[int, str]], originals: dict, conditions_data: list[dict]) -> str:
    parts = [f'''<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Random Sample Gallery — {len(sample)} frames</title>
<style>
:root {{
  --bg: #0c0d11; --panel: #161822; --panel-2: #1c1f2c; --border: #2a2d3d;
  --text: #dcdce2; --text-dim: #8b8ea3; --accent: #e94560; --ok: #2d9d6f; --warn: #e8a33d;
  --mono: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--text); font-family: var(--sans); margin: 0; padding: 28px; line-height: 1.55; max-width: 1400px; margin: 0 auto; }}
h1 {{ font-family: var(--mono); font-size: 22px; margin: 0 0 6px; }}
h2 {{ font-family: var(--mono); font-size: 15px; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 8px; margin: 40px 0 16px; }}
.dek {{ color: var(--text-dim); font-size: 13px; margin: 0 0 20px; }}
.params {{ font-family: var(--mono); font-size: 11px; color: var(--warn); }}
td.params {{ font-size: 11px; }}
.badge {{ display: inline-block; font-size: 10px; padding: 2px 7px; border-radius: 3px; margin-left: 8px; font-family: var(--mono); }}
.badge-ok {{ background: rgba(45,157,111,0.18); color: var(--ok); }}
.badge-abort {{ background: rgba(233,69,96,0.18); color: var(--accent); }}
table {{ border-collapse: collapse; width: 100%; font-family: var(--mono); font-size: 12px; margin-bottom: 24px; }}
th, td {{ padding: 6px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
th {{ color: var(--text-dim); font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; font-size: 10px; }}
.frame {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; margin-bottom: 16px; }}
.frame-head {{ padding: 8px 14px; border-bottom: 1px solid var(--border); font-family: var(--mono); font-size: 11px; color: var(--text-dim); }}
.pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--border); }}
.pair figure {{ margin: 0; background: var(--panel); }}
.pair img {{ width: 100%; display: block; }}
.pair figcaption {{ font-family: var(--mono); font-size: 10px; text-align: center; padding: 6px; color: var(--text-dim); text-transform: uppercase; }}
.orig-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }}
.orig-grid figure {{ margin: 0; background: var(--panel); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }}
.orig-grid img {{ width: 100%; display: block; }}
.orig-grid figcaption {{ font-family: var(--mono); font-size: 10px; text-align: center; padding: 6px; color: var(--text-dim); }}
</style>
</head>
<body>
<h1>Random Sample Gallery &middot; {len(sample)} frames across the sweep</h1>
<p class="dek">Random (sequence, frame) sample from the 10-sequence x 10-frame x 30-condition sweep (seed 42). ORIGINAL panels shared across all condition sections below.</p>
<h2>Summary</h2>
<table>
<tr><th>Condition</th><th>Parameters</th><th>Status</th><th>Frames rendered</th></tr>
''']
    for c in conditions_data:
        badge = "badge-ok" if c["status"] == "ok" else "badge-abort"
        label = "ran" if c["status"] == "ok" else "no predictions"
        params = load_condition_params(c["name"])
        parts.append(f'<tr><td>{c["name"]}</td><td class="params">{params}</td>'
                     f'<td><span class="badge {badge}">{label}</span></td>'
                     f'<td>{len(c["frames"])}/{len(sample)}</td></tr>\n')
    parts.append("</table>\n")

    parts.append('<h2>Original baseline<span class="badge badge-ok">reference</span></h2>\n')
    parts.append('<div class="frames-original">\n')
    for seq, fid in sample:
        if (seq, fid) not in originals:
            continue
        o = originals[(seq, fid)]
        parts.append(f'<div class="frame"><div class="frame-head">SEQ {seq} &middot; FRAME {fid} &middot; '
                     f'GT {o["gt_count"]} &middot; PRED {o["pred_count"]}</div>\n'
                     f'<div class="orig-grid">\n'
                     f'<figure><img src="data:image/jpeg;base64,{o["radar"]}"><figcaption>radar</figcaption></figure>\n'
                     f'<figure><img src="data:image/jpeg;base64,{o["lidar"]}"><figcaption>lidar</figcaption></figure>\n'
                     f'<figure><img src="data:image/jpeg;base64,{o["camera"]}"><figcaption>camera front0</figcaption></figure>\n')
        if "cam_front1" in o:
            parts.append(f'<figure><img src="data:image/jpeg;base64,{o["cam_front1"]}">'
                         f'<figcaption>camera front1 (unused by ASF)</figcaption></figure>\n')
        parts.append('</div></div>\n')
    parts.append('</div>\n')

    for c in conditions_data:
        badge = ('<span class="badge badge-ok">ran</span>' if c["status"] == "ok"
                 else '<span class="badge badge-abort">no predictions</span>')
        parts.append(f'<h2>{c["name"]}{badge}</h2>\n')
        parts.append(f'<p class="dek params">{load_condition_params(c["name"])}</p>\n')
        if not c["frames"]:
            parts.append('<p class="dek">No predictions for this condition in the sampled frames.</p>\n')
            continue
        for seq, fid in sample:
            if (seq, fid) not in c["frames"] or (seq, fid) not in originals:
                continue
            cf = c["frames"][(seq, fid)]
            o = originals[(seq, fid)]
            parts.append(f'<div class="frame"><div class="frame-head">SEQ {seq} &middot; FRAME {fid} &middot; '
                         f'ORIGINAL PRED {o["pred_count"]} &middot; CORRUPTED PRED {cf["pred_count"]}</div>\n')
            for modality, label in (("radar", "RADAR"), ("lidar", "LIDAR"), ("camera", "CAMERA")):
                if modality not in cf:
                    continue
                changed = cf["changed"].get(
                    "rdr_sparse" if modality == "radar" else
                    "ldr64" if modality == "lidar" else "front0", None)
                corrupt_label = "corrupted" if changed else "original (identical input)"
                parts.append(f'<div class="pair">\n'
                             f'<figure><img src="data:image/jpeg;base64,{o[modality]}">'
                             f'<figcaption>{label} original</figcaption></figure>\n'
                             f'<figure><img src="data:image/jpeg;base64,{cf[modality]}">'
                             f'<figcaption>{label} {corrupt_label}</figcaption></figure>\n'
                             f'</div>\n')
            parts.append('</div>\n')

    parts.append("</body>\n</html>\n")
    return "".join(parts)


if __name__ == "__main__":
    main()
