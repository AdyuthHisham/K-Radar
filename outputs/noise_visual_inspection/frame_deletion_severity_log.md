# frame_deletion severity realized outcomes

`frame_deletion` is a binary per-frame effect (present/absent); severity here is
the random-mode `p` (probability of firing this specific frame), not a
continuous corruption strength. A single Bernoulli draw at a given `p` can
land on the less-likely outcome (e.g. p=0.9 not firing is a real ~10% event).
`low`/`high` search up to 25 alternate seeds to land on the statistically
expected outcome (not fired / fired); `medium` (p=0.5) takes the first draw
with no preference, since 50/50 has no expected side.

| Modality | Frame | Severity | p | Fired? | Seed attempts |
|---|---|---|---|---|---|
| radar | seq001_rdr033 | low | 0.1 | False | 1 |
| radar | seq001_rdr033 | medium | 0.5 | False | 1 |
| radar | seq001_rdr033 | high | 0.9 | True | 1 |
| lidar | seq001_rdr033 | low | 0.1 | False | 1 |
| lidar | seq001_rdr033 | medium | 0.5 | True | 1 |
| lidar | seq001_rdr033 | high | 0.9 | True | 1 |
| camera | seq001_rdr033 | low | 0.1 | False | 1 |
| camera | seq001_rdr033 | medium | 0.5 | True | 1 |
| camera | seq001_rdr033 | high | 0.9 | True | 1 |
| radar | seq021_rdr015 | low | 0.1 | False | 1 |
| radar | seq021_rdr015 | medium | 0.5 | False | 1 |
| radar | seq021_rdr015 | high | 0.9 | True | 1 |
| lidar | seq021_rdr015 | low | 0.1 | False | 1 |
| lidar | seq021_rdr015 | medium | 0.5 | False | 1 |
| lidar | seq021_rdr015 | high | 0.9 | True | 1 |
| camera | seq021_rdr015 | low | 0.1 | False | 1 |
| camera | seq021_rdr015 | medium | 0.5 | True | 1 |
| camera | seq021_rdr015 | high | 0.9 | True | 1 |
| radar | seq048_rdr035 | low | 0.1 | False | 1 |
| radar | seq048_rdr035 | medium | 0.5 | True | 1 |
| radar | seq048_rdr035 | high | 0.9 | True | 1 |
| lidar | seq048_rdr035 | low | 0.1 | False | 1 |
| lidar | seq048_rdr035 | medium | 0.5 | True | 1 |
| lidar | seq048_rdr035 | high | 0.9 | True | 1 |
| camera | seq048_rdr035 | low | 0.1 | False | 1 |
| camera | seq048_rdr035 | medium | 0.5 | True | 1 |
| camera | seq048_rdr035 | high | 0.9 | True | 2 |
