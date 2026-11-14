# Sea state from NDBC buoy cameras (`ndbc_buoycam_sea_state_v0`)

**Question.** From one hourly six-view strip of an NDBC buoy camera, can a
network tell the sea state (four WMO classes; the wave height itself in a
later version) on buoys it has never seen? Decided with Steven on 2026-11-14,
after the Currituck camera turned out to be judged against a buoy 9.2 km
away ([Currituck](currituck_sea_state.md)). Here **the camera and the wave
sensor are on the same buoy**: the label is measured where the image is
taken.

Config: `configs/experiments/ndbc_buoycam_sea_state_v0.yaml`. Pairing:
`scripts/buoycam_pairs.py`. Package and baselines:
`scripts/buoycam_package.py`. Building blocks:
`wave_analysis.datasets.buoycam`. Built on 2026-11-14 from commit `7b8a93c`
(`dirty: false`); package `data/processed/packages/ndbc_buoycam_sea_state_v0`
(2.9 GB, `SHA256SUMS` checked).

## Ground truth

The buoy's significant wave height, `WVHT` in NDBC's realtime spectral
summary (`.spec`). The buoy computes it on board from a 20-min record of its
own motion (accelerometers or a heave-pitch-roll package, depending on the
payload). NDBC does not transmit the raw motion series, only the spectra and
summaries ([NDBC](../datasets/ndbc.md#instruments)). One record's 90 %
interval is about −10 % to +15 % of H_m0 (Earle 2003).

- **Source files:** the realtime `.spec` files of every camera station,
  saved on the 1st and 15th of each month (45 days each) and once more on
  2026-11-14 15:33 UTC to cover the last two weeks. 377 snapshot files,
  331,178 records with WVHT at 79 stations; where snapshots overlap, the
  latest is kept (none of the overlapping values differed). These are
  realtime values, not NDBC's quality-controlled annual files, which appear
  the following year; the labels can be rebuilt from those then.
- **Timing:** records every 30 min, stamped :10 and :40. NDBC rounds wave
  stamps to the half hour; the `.spec` stamp lies at the record's centre or
  up to 10 min before it, depending on the payload. The stamp is taken as the
  nominal centre (records of 20 min, ±10 min around it).
- **Label (ADR 0010):** the mean of the records centred within 30 min of the
  strip's time stamp. Strips are stamped :10 (sometimes :00), so the window
  normally holds three records (−30, 0, +30 min): 84,823 of 90,434 labelled
  strips; two records 5,172; one 439. The `nearest` and `interpolate` labels
  are stored beside it. The window mean differs from the nearest record by
  at most 0.13 m for 90 % of strips, and gives a different class for 3.4 %
  of the packaged strips. Median spread of the records in the window 0.1 m
  (95th percentile 0.5 m). Context from the nearest record: swell and
  wind-sea heights and periods, mean period and direction, NDBC's steepness
  class.

## Strips

94,187 archived strips at 80 buoys, 2026-09-23 to 2026-11-14 (hourly, day
and night). 90,434 have a label; 3,753 have no record within 30 min, and
45001 (Lake Superior) has no `.spec` records at all.

**Format.** Six views side by side, 2880 × 300 px (6 × 480; 44008:
2304 × 246, 6 × 384), with a burned-in caption (station ID, UTC time,
heading). **The caption is a fixed band of the bottom 30 rows in both
formats**, measured on daylight strips of 44008 and five other stations on
2026-11-14. The code had scaled it to 25 rows for the 246-px format; it no
longer scales. Every packaged strip has the band cut, so a model cannot read
the station or the time. Eleven strips of 3360 or 3840 px (7 or 8 views)
are left out.

**Light.** The cameras raise their exposure in twilight: strips taken with
the sun 3–6° below the horizon are in colour and show the sea, brighter
(median grey 112) than strips with the sun just above the horizon (52),
where the views facing away from the sun come out dark. At night the views
are black (median grey 2, standard deviation 0.1). Morning and evening agree,
and no station's brightness disagrees with the sun's position at its listed
location, so the clocks and positions hold. Hence two rules:

- the sun at most 6° below the horizon at the buoy (`sun_elevation_deg ≥ −6`);
- at least 3 **usable views**: mean grey ≥ 20 and standard deviation ≥ 4
  (0–255, caption excluded). Each view's mean and deviation are in
  `labels.csv`, so a model can mask unusable views.

**Kept: 41,091 strips at 79 buoys** (median 563 per buoy, 35 to 649); 35,843
have all six views usable. 34,460 day, 2,757 low sun, 3,868 civil twilight.
Left out: 45,944 too dark (sun lower), 3,388 with fewer than 3 usable views,
3,753 unlabelled, 11 of another format.

Buoy positions are the median of the saved `buoycams.php` listings; no buoy
moved more than 5 km. 46006 was redeployed between 2026-09-26 and 09-28 (old
camera W67A, new W97A, both listed for 46 h); its wave records resume on
09-28 at 19:50, so none of the doubled hours has a label. Strip IDs carry the
camera code (`46006-W97A-2026-10-01T1810Z`).

## Classes

The WMO bands of [Currituck](currituck_sea_state.md#classes) (≤ 0.5, ≤ 1.25,
≤ 2.5, > 2.5 m, closed on the right):

| Class | Strips | Share |
|---|---|---|
| smooth | 3,405 | 8.3 % |
| slight | 10,713 | 26.1 % |
| moderate | 18,007 | 43.8 % |
| rough or more | 8,966 | 21.8 % |

Open-ocean buoys in autumn: "rough" is 22 % here against 2 % at Currituck,
and "smooth" is the rare class. Hs runs from 0 to 13.1 m (median 1.6 m; 46002
off Oregon in the storm of 2026-11-13). The zeros are real: NDBC reports
below 0.05 m as 0.0, and 46081 (Prince William Sound) shows glassy water on
those days. 7.7 % of the strips lie within 4 % of a class boundary.

## Splits

| Split | Train | Val | Test |
|---|---|---|---|
| **station**: unseen buoys, groups ≥ 300 km apart, 70/15/15 by strips (seed 0) | 28,597 strips, 55 buoys | 6,311, 12 | 6,183, 12 |
| **time**: same buoys, later weeks, 72 h left out after each boundary | 26,265, 79 (09-23 to 10-23) | 4,745, 75 (10-27 to 11-02) | 5,870, 70 (11-06 to 11-14) |

- **Station split, the main test.** Buoys closer than 300 km to each other
  (chained) form one group and go to one partition, so a test buoy never sees
  a training buoy's weather from nearby: 43 groups, the largest the US West
  Coast (21 buoys). Measured: the closest test–train pair is 322 km apart
  (46085–46001), test–val 313 km, val–train 371 km. **Test buoys:** 41002,
  41043, 41047, 41049 (open Atlantic), 42057 (Caribbean), 44065 (off New
  York), 46059, 46066, 46085 (NE Pacific and Gulf of Alaska), 51000, 51002,
  51003 (Hawaii). **Val:** the Gulf of Mexico 42002, 42036, 42039, 42060; the
  New England group 44007, 44008, 44011, 44013, 44020, 44027; 46035, 46073
  (Bering Sea).
- **The test buoys are mostly open ocean**, so "smooth" has only 94 test
  strips (train 2,410, val 901). Its row of the confusion matrix will be
  noisy; counts are given beside every rate. Cross-validation over the
  groups would use every buoy as a test buoy once; not built yet.
- **Time split.** The same 79 buoys, train before 2026-10-24, val, test from
  2026-11-03; the first 72 h after each boundary are left out (4,211 strips).
  The leakage audit finds no train and test strip of the same buoy closer
  than 72 h.
- Intervals come from a bootstrap over whole buoys (station split) or whole
  buoy-days (time split). With 12 test buoys, the station split's intervals
  are wide by design.

## Baselines (no image)

| Split / partition | Baseline | Balanced acc. | Macro-F1 |
|---|---|---|---|
| station / test | majority ("moderate") | 0.25 | 0.17 [0.14, 0.25] |
| time / test | majority | 0.25 | 0.11 [0.10, 0.12] |
| time / test | each buoy's own majority class in its training weeks | 0.39 [0.36, 0.42] | 0.35 [0.31, 0.39] |

**Still to add: ERA5 at each buoy**, the physics baseline (as at Currituck).
It needs the 79 buoys as sites and about 240 monthly requests to the
Copernicus data store. Without it there is no bar a camera must beat beyond
these trivial ones.

## Status

- [x] Pairing and package (2026-11-14, `7b8a93c`): 41,091 strips at 79
      buoys, leakage audits pass, `SHA256SUMS` checked, captions cut.
- [x] Majority baselines scored.
- [ ] ERA5 baseline at each buoy.
- [ ] Training (the Currituck script needs a reader for six-view strips;
      not started, needs Steven's go).
- [ ] Relabel from NDBC's quality-controlled annual files when they appear.
