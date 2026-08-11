# ASF per-sensor corruption study — Alvis smoke run

**Date:** 2026-08-11 · **Model:** ASF / A2Fusion (`model_10.pt`) · **Container:** `kradarV0.sif`
**Scope:** 6 frames — K-Radar sequences 1 and 2, the 3 densest frames each
(`configs/ASF_v2_0_smoke_alvis.yml`: `portion: ['1','2']`, `max_per_seq: 3`,
`frame_select: most_objects`) · **conf_thr:** 0.3 · **seed:** 42

Every condition applies **exactly one effect to exactly one modality**; all other
sensors are clean, and all conditions run the identical 6 frames. Blackout policy
is `empty` throughout — a genuinely dead input, with no model-side compensation.

## Headline

| Condition | Detections (clean=19) | Δ | Matched | Lost | Mean displacement | Status |
|---|---|---|---|---|---|---|
| **clean** | 19 | — | — | — | — | baseline |
| radar_noise_induced_shifts | 14 | −5 | 14 | 5 | 0.305 m | OK |
| radar_loss_partial (0.5) | 19 | +0 | 19 | 0 | 0.268 m | OK |
| radar_zeros (all-zero frame) | 20 | +1 | 19 | 0 | 0.315 m | OK |
| radar_frame_deletion | — | — | — | — | — | **spconv abort** |
| radar_loss_complete | — | — | — | — | — | **spconv abort** |
| lidar_gaussian_noise | 6 | **−13** | 5 | 14 | 0.723 m | OK |
| lidar_loss_partial (0.5) | 16 | −3 | 16 | 3 | 0.117 m | OK |
| lidar_zeros (all-zero frame) | 9 | **−10** | 8 | 11 | 0.438 m | OK |
| lidar_frame_deletion | — | — | — | — | — | **spconv abort** |
| lidar_loss_complete | — | — | — | — | — | **spconv abort** |
| camera_frame_deletion | 19 | +0 | 19 | 0 | 0.073 m | OK |
| camera_gaussian_noise (σ=40) | 18 | −1 | 18 | 1 | 0.127 m | OK |
| camera_loss_partial (0.3) | 19 | +0 | 19 | 0 | 0.119 m | OK |
| camera_loss_complete | 18 | −1 | 18 | 1 | 0.143 m | OK |

Machine-readable: `corruption_summary.csv`. Per-frame corrupted sensor data
(point clouds `.npy`, camera `.png`, plus clean copies and a `_meta.txt`) is under
each condition's `sensor_dump/`.

## Sensitivity ranking

**LiDAR ≫ radar > camera.** LiDAR is by far the dominant modality: modest
positional noise (`sigma_xy=0.5`, `sigma_z=0.2`) destroys 13 of 19 detections,
and a zeroed LiDAR frame costs 10. Radar corruption costs at most 5. Camera
corruption is nearly free — even a **fully dead camera loses a single detection
out of 19**.

Why LiDAR noise bites so hard: `sigma_xy=0.5 m` exceeds the 0.4 m voxel size
(`roi.voxel_size: [0.4, 0.4, 0.4]`), so every point is displaced roughly one
voxel or more and the sparse-conv occupancy pattern is scrambled — not merely
blurred. Displacement of surviving boxes stays small (0.723 m), so the failure
mode is **detections disappearing, not drifting**.

The camera result is the notable one for a "tri-modal" fusion model: at this
operating point the camera branch contributes almost nothing that the LiDAR and
radar branches do not already supply. Worth confirming on more than 6 frames
before treating it as general.

## The architecture cannot ingest a dead radar or LiDAR

4 of 14 conditions abort with zero predictions. All four are the true-blackout
cases on a point-cloud sensor:

```
RuntimeError: N > 0 assert faild. CUDA kernel launch blocks must be positive, but got N= 0
  spconv/pytorch/ops.py:521  get_indice_pairs_implicit_gemm
  models/backbone_3d/spconv_backbone.py:149
```

A zero-point cloud means zero voxels, so the sparse-conv kernel launches with
zero thread blocks and asserts. The failure is fatal, not graceful: the
pipeline's own handler then dies with `UnboundLocalError: local variable
'dict_out' referenced before assignment`, aborting the whole job rather than
skipping one frame.

Consequences worth recording:

1. **A dead point-cloud sensor is not representable at the data level.** Feeding
   nothing is not an option the stack supports.
2. **`avail_feats` does not help.** `A2Fusion.forward` (`a2_fusion.py:167-172`)
   does have an availability mask that skips a branch at eval time, but it sits
   *downstream* of both the collate and the encoder, so a null sensor crashes
   long before fusion is reached.
3. **The `_zeros` conditions are the usable substitute.** A well-formed frame of
   all-zero readings (`loss_partial` at `fraction: 1.0`) is still purely
   data-level, survives the encoder, and yields the measurable −10 (LiDAR) /
   +1 (radar) rows above.

Camera is exempt throughout: a dead camera is a black image tensor, which is
well-formed, so it runs end to end.

## Bug found and fixed during the run

`_get_modality_idx` (`datasets/effects/noise_injection.py`) resolved the frame
index with a bare `idx_dict.get(modality)`, but `dict_item['meta']['idx']` is
keyed by the dataset's sensor names — `{'rdr', 'ldr64', 'ldr128', 'camf',
'camr'}` — not the injector's. Only `'rdr'` coincided, so **deterministic
`frame_deletion` silently never fired for LiDAR or camera**: it returned
`frame_index=None`, and `_frame_deletion_check` treats `None` as "do not delete".

The affected runs reported 6/6 predictions and looked healthy while being
unwitting clean runs. It was caught by the per-frame sensor dumps
(`changed=False`, `blackout: []` on every frame), not by any test — the existing
120 unit tests pass both before and after, because they construct `meta['idx']`
with the injector's own naming.

Fixed by mapping both spellings (`rdr`/`radar`, `ldr`/`lidar`, `cam`/`camera`)
onto the dataset's candidate keys and coercing the zero-padded string indices
(`'00150'`) to int. `lidar_frame_deletion` correctly aborts after the fix,
matching `lidar_loss_complete`.

## Caveats

- **6 frames.** These are detection-count deltas, not AP. The KITTI AP written
  by the pipeline on this scope is meaningless and is not reported here.
- **One severity per effect**, taken from `effect_parameters.yaml`. No severity
  ladder was swept, so "camera is unimportant" holds at σ=40, not in general.
- **Matching is greedy nearest-centre within class, gated at 5 m**
  (`--match-radius`). Without the gate, `lidar_gaussian_noise` reported 7.488 m
  by pairing unrelated boxes; the gate reclassifies those as lost detections and
  gives 0.723 m over 5 genuine matches.
- Displacement is only computed over surviving matches, so a condition that
  deletes most detections can show a small displacement — read it alongside the
  lost count, never alone.

## Reproduce

```bash
cd $AM                                    # repos/K-Radar on Alvis
bash scripts/alvis/submit_smoke.sh clean
bash scripts/alvis/submit_smoke.sh sweep  # all 12 single-effect conditions
bash scripts/alvis/submit_smoke.sh radar_zeros radar_zeros.yml empty
bash scripts/alvis/submit_smoke.sh lidar_zeros lidar_zeros.yml empty

apptainer exec --bind $AM:/opt/K-Radar $SIF \
  python /opt/K-Radar/scripts/alvis/compare_smoke.py \
    --outputs /opt/K-Radar/outputs/alvis_smoke --conditions <names> \
    --csv /opt/K-Radar/outputs/alvis_smoke/corruption_summary.csv
```
