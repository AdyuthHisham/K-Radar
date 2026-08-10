# Before/After Row Counts — loss_partial zero-out (fraction=0.30)

| Frame/Modality | n_before | n_after | Row count preserved | Zeroed count | Expected zeroed | Actual zeroed frac |
|---|---|---|---|---|---|---|
| seq001_rdr033/radar | 150000 | 150000 | True | 45000 | 45000 | 0.3000 |
| seq001_rdr033/lidar | 60601 | 60601 | True | 18180 | 18180 | 0.3000 |
| seq001_rdr033/camera | 5529600 | 5529600 | True | 1671600 | 1658880 | 0.3023 |
| seq021_rdr015/radar | 150000 | 150000 | True | 45000 | 45000 | 0.3000 |
| seq021_rdr015/lidar | 73569 | 73569 | True | 22070 | 22070 | 0.3000 |
| seq021_rdr015/camera | 5529600 | 5529600 | True | 1659009 | 1658880 | 0.3000 |
| seq048_rdr035/radar | 150000 | 150000 | True | 45000 | 45000 | 0.3000 |
| seq048_rdr035/lidar | 48209 | 48209 | True | 14462 | 14462 | 0.3000 |
| seq048_rdr035/camera | 5529600 | 5529600 | True | 1659679 | 1658880 | 0.3001 |
