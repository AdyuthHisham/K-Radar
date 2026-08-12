# ASF per-sensor corruption study — Alvis

**Model:** ASF / A2Fusion (`model_10.pt`) · **Container:** `kradarV0.sif`
**Scope:** 6 frames — K-Radar sequences 1 and 2, the 3 densest frames each
(`configs/ASF_v2_0_smoke_alvis.yml`: `portion: ['1','2']`, `max_per_seq: 3`,
`frame_select: most_objects`) · **conf_thr:** 0.3 · **seed:** 42

Each condition applies **exactly one effect to exactly one modality**; all other
sensors are clean, and all conditions run the identical 6 frames. Blackout
policy is `empty` throughout — a genuinely dead input, with no model-side
compensation.

Taxonomy is the canonical **9 effects** (3 per modality) per
`docs/vAIlt/04_Design/Noise_Taxonomy.md`. `loss_complete` is excluded: it is
`frame_deletion` at `interval: 1` — the same operation, ungated. An earlier
sweep ran it and confirmed the equivalence (identical aborts for radar/LiDAR,
rate-only difference for camera).

> These numbers postdate the seeding fix below and are reproducible. Any figures
> from before 2026-08-12 are not comparable — they were generated with
> per-process-random seeds.

## Results

| Condition | Detections (clean=19) | Δ | Matched | Lost | Mean displacement | Status |
|---|---|---|---|---|---|---|
| **clean** | 19 | — | — | — | — | baseline |
| radar_noise_induced_shifts | 16 | −3 | 16 | 3 | 0.380 m | OK |
| radar_loss_partial (0.5) | 19 | +0 | 19 | 0 | 0.304 m | OK |
| radar_zeros (all-zero frame) | 20 | +1 | 19 | 0 | 0.315 m | OK |
| radar_frame_deletion | — | — | — | — | — | **spconv abort** |
| **lidar_gaussian_noise** | **7** | **−12** | 6 | 13 | 0.516 m | OK |
| lidar_loss_partial (0.5) | 16 | −3 | 16 | 3 | 0.130 m | OK |
| lidar_zeros (all-zero frame) | 9 | −10 | 8 | 11 | 0.438 m | OK |
| lidar_frame_deletion | — | — | — | — | — | **spconv abort** |
| camera_frame_deletion | 19 | +0 | 19 | 0 | 0.073 m | OK |
| camera_gaussian_noise (σ=40) | 18 | −1 | 18 | 1 | 0.127 m | OK |
| camera_loss_partial (0.3) | 18 | −1 | 18 | 1 | 0.106 m | OK |

Machine-readable: `corruption_summary.csv`. Per-frame corrupted sensor data
(`.npy` point clouds, `.png` images, clean copies, `_meta.txt`) is under each
condition's `sensor_dump/`.

## Sensitivity ranking

**LiDAR ≫ radar > camera.**

- **LiDAR dominates.** Modest positional noise (`sigma_xy=0.5`, `sigma_z=0.2`)
  destroys 12 of 19 detections. The mechanism is voxel-level: 0.5 m exceeds the
  0.4 m voxel size (`roi.voxel_size: [0.4, 0.4, 0.4]`), so points migrate to
  neighbouring voxels and the sparse-conv occupancy pattern is scrambled rather
  than blurred. Zeroing LiDAR entirely costs 10.
- **Radar is mid.** Coordinate shifts cost 3; zeroing half the detections costs
  nothing measurable (19 → 19).
- **Camera is nearly free.** Every camera corruption costs at most 1 detection
  of 19, including dropping every other frame entirely (0). At this operating
  point the camera branch contributes little the other two do not already
  supply — notable for a model billed as tri-modal, and the single most
  interesting result to confirm at larger scale.

Across all conditions the failure mode is **detections disappearing, not
drifting**: surviving boxes stay within ~0.5 m. Read displacement alongside the
lost count, never alone — a condition that deletes most detections can still
show small displacement.

## The architecture cannot ingest a dead radar or LiDAR

Both point-cloud `frame_deletion` conditions abort with zero predictions:

```
RuntimeError: N > 0 assert faild. CUDA kernel launch blocks must be positive, but got N= 0
  spconv/pytorch/ops.py:521  get_indice_pairs_implicit_gemm
  models/backbone_3d/spconv_backbone.py:149
```

Zero points → zero voxels → a zero-block kernel launch. The failure is fatal,
not graceful: the pipeline's handler then dies with `UnboundLocalError: local
variable 'dict_out' referenced before assignment`, killing the whole job rather
than skipping one frame.

1. **A dead point-cloud sensor is not representable at the data level.**
2. **`avail_feats` does not rescue it.** `A2Fusion.forward`
   (`a2_fusion.py:167-172`) does have an availability mask that skips a branch
   at eval time, but it sits *downstream* of both collate and the encoder, so a
   null sensor crashes long before fusion is reached.
3. **The `_zeros` conditions are the usable substitute** — a well-formed frame
   of all-zero readings (`loss_partial` at `fraction: 1.0`), still purely
   data-level, which survives the encoder.

Camera is exempt: a dead camera is a black image tensor, which is well-formed.

## Reproducibility

Verified by submitting one condition twice (`determinism_A` / `determinism_B`):
**injected sensor data is byte-identical across runs**, detection counts match,
and mean displacement is 0.000 m.

A residual float jitter of ~1e-5 remains in the predictions (e.g. a score of
`0.31614053` vs `0.31615087`), from non-associative float accumulation in the
GPU sparse-conv — reproduced on the *same* node, so it is not node variation.
It does not change detection counts or box positions, and matters only for a box
sitting exactly on the 0.3 confidence threshold.

## Bugs found and fixed during this study

**1. `frame_deletion` silently never fired for LiDAR or camera.**
`_get_modality_idx` resolved the frame index with a bare
`idx_dict.get(modality)`, but `meta['idx']` is keyed by the dataset's sensor
names — `{'rdr', 'ldr64', 'ldr128', 'camf', 'camr'}` (`kradar_fusion_v1_0.py:442`,
identical in `kradar_detection_v2_0.py:201` / `v1_1.py:154`). Only `'rdr'`
coincided, so the lookup returned `None` and `_frame_deletion_check` treats
`None` as "do not delete". Affected runs reported 6/6 predictions and looked
healthy while being unwitting clean runs. Caught by the per-frame sensor dumps
(`changed=False`), not by any test.

**2. Seeding was not reproducible across runs.** `NoiseInjector._resolve`
derived each effect's sub-seed with `hash((seed, modality, name, i))`. Since
`modality` and `name` are `str`, CPython's per-process hash salt made the
sub-seed different on every run — two executions of the same config with the
same `seed:` produced **different corruption**. This is why an earlier
regeneration of the sweep drifted despite unchanged configs. Replaced with a
blake2b-based `_stable_hash`.

Both bugs were invisible to the existing 120 tests, for the same underlying
reason: the tests exercise the injector against its own conventions and inside a
single process. The suite now carries K-Radar-accurate metadata fixtures and a
subprocess test that varies `PYTHONHASHSEED` — **140 tests, passing locally and
in-container on Python 3.8**. Reverting either fix makes the new tests fail.

Two measurement bugs were also fixed: the pipeline's `dummy` placeholder row was
being counted as a detection (so empty frames read as "1 box, score 0.000"), and
box matching had no distance gate (pairing unrelated boxes and reporting 7.488 m
of "displacement" where the truth was lost detections).

## Caveats

- **6 frames, 19 baseline detections.** These are detection-count deltas, not
  AP. The KITTI AP on this scope is meaningless and is not reported.
- **One severity per effect**, from `effect_parameters.yaml`. No severity ladder
  was swept, so "camera is unimportant" holds at σ=40, not in general.
- **Single seed.** Every condition uses `seed: 42`; no seed-variance estimate.
- Matching is greedy nearest-centre within class, gated at 5 m
  (`--match-radius`).

## Reproduce

```bash
cd $AM                                    # repos/K-Radar on Alvis
bash scripts/alvis/submit_smoke.sh clean
bash scripts/alvis/submit_smoke.sh sweep  # the 9 single-effect conditions
bash scripts/alvis/submit_smoke.sh radar_zeros radar_zeros.yml empty
bash scripts/alvis/submit_smoke.sh lidar_zeros lidar_zeros.yml empty

apptainer exec --bind $AM:/opt/K-Radar $SIF \
  python /opt/K-Radar/scripts/alvis/compare_smoke.py \
    --outputs /opt/K-Radar/outputs/alvis_smoke --conditions <names> \
    --csv /opt/K-Radar/outputs/alvis_smoke/corruption_summary.csv
```

Jobs are preemptible: a `CANCELLED` state with `00:00:00` elapsed and no log
means the scheduler reclaimed it before it started — just resubmit.
