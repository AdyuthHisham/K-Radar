# v2 Regeneration — Byte-Identity Confirmation Log

Compares each re-rendered PNG's SHA-256 against the file that existed at that
path before this run (from the original `visualize_effects.py` pass), for every
effect NOT intentionally re-styled. `loss_partial` (radar/lidar) is intentionally
re-styled (old file replaced by the zero-out-aware render) and is excluded from
the match/mismatch check below.

| Modality | Effect | Frame | Old hash (prefix) | New hash (prefix) | Result |
|---|---|---|---|---|---|
| radar | clean | seq001_rdr033 | `11efec8bf14f` | `11efec8bf14f` | CONFIRMED IDENTICAL |
| radar | frame_deletion | seq001_rdr033 | `9d44acc32133` | `9d44acc32133` | CONFIRMED IDENTICAL |
| radar | frame_deletion_random | seq001_rdr033 | `347247b48841` | `347247b48841` | CONFIRMED IDENTICAL |
| radar | noise_induced_shifts | seq001_rdr033 | `ecc48c46c484` | `ecc48c46c484` | CONFIRMED IDENTICAL |
| radar | loss_partial | seq001_rdr033 | `73ba5fb8b2e4` | `73ba5fb8b2e4` | RE-STYLED (intentional) |
| radar | loss_complete | seq001_rdr033 | `6535112b01bf` | `6535112b01bf` | CONFIRMED IDENTICAL |
| lidar | clean | seq001_rdr033 | `89b58e023bb8` | `89b58e023bb8` | CONFIRMED IDENTICAL |
| lidar | frame_deletion | seq001_rdr033 | `9d44acc32133` | `9d44acc32133` | CONFIRMED IDENTICAL |
| lidar | frame_deletion_random | seq001_rdr033 | `bc380b536ca9` | `bc380b536ca9` | CONFIRMED IDENTICAL |
| lidar | gaussian_noise | seq001_rdr033 | `5590a564a90c` | `5590a564a90c` | CONFIRMED IDENTICAL |
| lidar | loss_partial | seq001_rdr033 | `dabb2d7a3983` | `dabb2d7a3983` | RE-STYLED (intentional) |
| lidar | loss_complete | seq001_rdr033 | `6535112b01bf` | `6535112b01bf` | CONFIRMED IDENTICAL |
| camera | clean | seq001_rdr033 | `0553c80b2bb3` | `0553c80b2bb3` | CONFIRMED IDENTICAL |
| camera | frame_deletion | seq001_rdr033 | `86f9fa1352aa` | `86f9fa1352aa` | CONFIRMED IDENTICAL |
| camera | frame_deletion_random | seq001_rdr033 | `8c9f3006f056` | `8c9f3006f056` | CONFIRMED IDENTICAL |
| camera | gaussian_noise | seq001_rdr033 | `9dc4206d9be5` | `9dc4206d9be5` | CONFIRMED IDENTICAL |
| camera | loss_partial | seq001_rdr033 | `1be9886eca32` | `1be9886eca32` | CONFIRMED IDENTICAL |
| camera | loss_complete | seq001_rdr033 | `bf34f64e2ae0` | `bf34f64e2ae0` | CONFIRMED IDENTICAL |
| radar | clean | seq021_rdr015 | `699a4f434c79` | `699a4f434c79` | CONFIRMED IDENTICAL |
| radar | frame_deletion | seq021_rdr015 | `324aeb7038d9` | `324aeb7038d9` | CONFIRMED IDENTICAL |
| radar | frame_deletion_random | seq021_rdr015 | `b68eaa34fca8` | `b68eaa34fca8` | CONFIRMED IDENTICAL |
| radar | noise_induced_shifts | seq021_rdr015 | `9617b14c29d8` | `9617b14c29d8` | CONFIRMED IDENTICAL |
| radar | loss_partial | seq021_rdr015 | `c208043f5a74` | `c208043f5a74` | RE-STYLED (intentional) |
| radar | loss_complete | seq021_rdr015 | `3a6637675a21` | `3a6637675a21` | CONFIRMED IDENTICAL |
| lidar | clean | seq021_rdr015 | `40d94830e6c5` | `40d94830e6c5` | CONFIRMED IDENTICAL |
| lidar | frame_deletion | seq021_rdr015 | `324aeb7038d9` | `324aeb7038d9` | CONFIRMED IDENTICAL |
| lidar | frame_deletion_random | seq021_rdr015 | `887b95eac62e` | `887b95eac62e` | CONFIRMED IDENTICAL |
| lidar | gaussian_noise | seq021_rdr015 | `f4c89d2bcec0` | `f4c89d2bcec0` | CONFIRMED IDENTICAL |
| lidar | loss_partial | seq021_rdr015 | `52337d019b81` | `52337d019b81` | RE-STYLED (intentional) |
| lidar | loss_complete | seq021_rdr015 | `3a6637675a21` | `3a6637675a21` | CONFIRMED IDENTICAL |
| camera | clean | seq021_rdr015 | `95114179daab` | `95114179daab` | CONFIRMED IDENTICAL |
| camera | frame_deletion | seq021_rdr015 | `53b53dd35661` | `53b53dd35661` | CONFIRMED IDENTICAL |
| camera | frame_deletion_random | seq021_rdr015 | `206437b7c74d` | `206437b7c74d` | CONFIRMED IDENTICAL |
| camera | gaussian_noise | seq021_rdr015 | `9caeafa2da5a` | `9caeafa2da5a` | CONFIRMED IDENTICAL |
| camera | loss_partial | seq021_rdr015 | `1c5981ce31ab` | `1c5981ce31ab` | CONFIRMED IDENTICAL |
| camera | loss_complete | seq021_rdr015 | `2ac3b7b3500a` | `2ac3b7b3500a` | CONFIRMED IDENTICAL |
| radar | clean | seq048_rdr035 | `6856e26b585a` | `6856e26b585a` | CONFIRMED IDENTICAL |
| radar | frame_deletion | seq048_rdr035 | `c32cc9bb768f` | `c32cc9bb768f` | CONFIRMED IDENTICAL |
| radar | frame_deletion_random | seq048_rdr035 | `124e1f537f06` | `124e1f537f06` | CONFIRMED IDENTICAL |
| radar | noise_induced_shifts | seq048_rdr035 | `7463e5a9a3e4` | `7463e5a9a3e4` | CONFIRMED IDENTICAL |
| radar | loss_partial | seq048_rdr035 | `0ab216aca02b` | `0ab216aca02b` | RE-STYLED (intentional) |
| radar | loss_complete | seq048_rdr035 | `441697a87655` | `441697a87655` | CONFIRMED IDENTICAL |
| lidar | clean | seq048_rdr035 | `f085db46bbfa` | `f085db46bbfa` | CONFIRMED IDENTICAL |
| lidar | frame_deletion | seq048_rdr035 | `c32cc9bb768f` | `c32cc9bb768f` | CONFIRMED IDENTICAL |
| lidar | frame_deletion_random | seq048_rdr035 | `c9d945584fc3` | `c9d945584fc3` | CONFIRMED IDENTICAL |
| lidar | gaussian_noise | seq048_rdr035 | `4d98bceebea3` | `4d98bceebea3` | CONFIRMED IDENTICAL |
| lidar | loss_partial | seq048_rdr035 | `04e67543b604` | `04e67543b604` | RE-STYLED (intentional) |
| lidar | loss_complete | seq048_rdr035 | `441697a87655` | `441697a87655` | CONFIRMED IDENTICAL |
| camera | clean | seq048_rdr035 | `5498a954434b` | `5498a954434b` | CONFIRMED IDENTICAL |
| camera | frame_deletion | seq048_rdr035 | `4ddc2d33ea56` | `4ddc2d33ea56` | CONFIRMED IDENTICAL |
| camera | frame_deletion_random | seq048_rdr035 | `3d03182b653d` | `3d03182b653d` | CONFIRMED IDENTICAL |
| camera | gaussian_noise | seq048_rdr035 | `c4bc5d670f8e` | `c4bc5d670f8e` | CONFIRMED IDENTICAL |
| camera | loss_partial | seq048_rdr035 | `33b40e78080b` | `33b40e78080b` | CONFIRMED IDENTICAL |
| camera | loss_complete | seq048_rdr035 | `eb0324bde603` | `eb0324bde603` | CONFIRMED IDENTICAL |
