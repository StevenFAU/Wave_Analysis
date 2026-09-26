# Dataset Splitting

Code: `datasets/splits.py`; ADR 0005.

## Why random splits are invalid here

Consecutive frames, overlapping clips, and consecutive buoy records are strongly
autocorrelated, and the wave state changes over hours. A random frame split puts
near-identical neighbours in train and test, and the reported score then measures
memorisation. `check_leakage` demonstrates this: in the test suite a random 80/20
split of hourly samples is flagged with hundreds of train/test pairs less than
2 hours apart.

## Protocols

| Protocol | Function | Unit held out | Question |
|---|---|---|---|
| `iid_grouped` | `grouped_split(df, ["station_id", "day"])` | station-days (or events, sequences) | In-distribution skill with independent blocks |
| `cross_time` | `time_block_split(..., embargo="24h")` | later dates, with an embargo gap after each boundary | Robustness to drift, seasons, sensor ageing |
| `cross_station` | `holdout_groups_split(df, "station_id", test_groups=[...])` | whole stations/cameras | **Transfer to unseen cameras.** The primary benchmark |
| `cross_region` | `holdout_groups_split(df, "region", ...)` | coastlines / basins | Transfer across wave climates |
| `extreme` | `extreme_condition_split(df, "Hs", quantile=0.9, group_col="event")` | the upper tail, by whole event | Extrapolation to storms |
| LOGO | `leave_one_group_out(df, "station_id")` | one site per fold | Small numbers of sites |

Details:

- `grouped_split` fills partitions greedily by sample count after a seeded
  shuffle, which handles very unequal group sizes.
- `time_block_split` labels samples within the embargo after each boundary
  `excluded` (kept, not used). With `group_col` (e.g. storm id), a group that
  straddles a boundary is excluded whole.
- `extreme_condition_split` moves any *event* containing an extreme sample
  entirely to test, so a storm's ramp-up in training cannot reveal its peak.
- Explicit hold-out lists (not random draws) keep cross-station benchmarks
  stable and citable.

## Self-supervised pretraining data

Unlabelled video used for pretraining (e.g. V-JEPA-style) must **exclude test
stations and test periods** too. Otherwise the encoder has seen the test
distribution. Track pretraining corpora in their own manifest and audit them
against the test split with the same `check_leakage` group and time checks.

## Mandatory audit

Every manifest build runs `check_leakage`:

1. shared groups across train/val and test;
2. temporal neighbours closer than `min_gap` within each station;
3. near-duplicate frames (dHash Hamming distance ≤ 4) across train and test.

The result is written into the manifest's provenance. `scripts/build_ml_manifest.py`
exits non-zero on leakage.
