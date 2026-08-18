#!/usr/bin/env python3
"""Generate a full-frame (no max_per_seq cap) ASF_v2_0_seq<N>_alvis.yml for
every sequence 1..58, from the ASF_v2_0_seq46_alvis.yml template (the only
functional difference between sequence configs is the portion block --
confirmed by direct diff during the sequence-1/sequence-46 pilot).

Overwrites any existing 10-frame-capped test configs (sequences 2, 7, 8, 9,
15, 16, 18, 41, 48, 55) with fresh full-frame versions -- per user decision
2026-08-18, the full run uses every sequence's complete (post-filter) frame
set, not the 10-sequence test's 10-frame cap. The old capped test *output*
data (outputs/alvis_seq*/*) is left untouched on disk; only the *config*
files are regenerated.

Usage:
    python scripts/alvis/gen_sequence_configs.py
"""
from __future__ import annotations

import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEMPLATE_PATH = os.path.join(REPO, "configs", "ASF_v2_0_seq46_alvis.yml")
N_SEQUENCES = 58


def main() -> None:
    with open(TEMPLATE_PATH) as f:
        template = f.read()

    # Template's header comment + portion block (see ASF_v2_0_seq46_alvis.yml)
    header_marker = "### ----- General ----- ###"
    body = template.split(header_marker, 1)[1]

    written = []
    for seq in range(1, N_SEQUENCES + 1):
        header = (
            f"# Sequence-{seq} original-vs-corrupted run on Alvis: full sequence\n"
            f"# (all frames surviving class/ROI/remove_0_obj filtering), no frame\n"
            f"# subsetting. Part of the full 58-sequence sweep. Identical to\n"
            f"# ASF_v2_0_seq46_alvis.yml except for the portion block.\n"
        )
        text = header + header_marker + body
        text = text.replace(
            "  # Scope: sequence 46 only, all frames (no max_per_seq/frame_select cap).\n  portion: ['46']",
            f"  # Scope: sequence {seq}, all frames (no max_per_seq/frame_select cap).\n  portion: ['{seq}']"
        )
        assert f"portion: ['{seq}']" in text, f"portion substitution failed for seq {seq}"
        assert "max_per_seq:" not in text, f"stale max_per_seq cap leaked into seq {seq} config"

        path = os.path.join(REPO, "configs", f"ASF_v2_0_seq{seq}_alvis.yml")
        with open(path, "w") as f:
            f.write(text)
        written.append(path)

    print(f"Wrote {len(written)} sequence configs")


if __name__ == "__main__":
    main()
