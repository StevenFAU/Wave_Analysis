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
`wave_analysis.datasets.buoycam`. Built on 2026-11-14 from commit `7b8a93c`, rebuilt with the ERA5 baseline
from `5198fab` (images unchanged; `dirty: false`); package `data/processed/packages/ndbc_buoycam_sea_state_v0`
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
| station / val | **ERA5 at the buoy, corrected** (92 % of strips) | 0.79 [0.73, 0.87] | 0.81 [0.73, 0.88] |
| **station / test** | **ERA5 at the buoy, corrected** (91 %) | **0.74 [0.67, 0.90]** | **0.77 [0.67, 0.91]** |
| time / test | ERA5 at the buoy, corrected (41 %: to 11-09) | 0.78 [0.73, 0.83] | 0.80 [0.74, 0.84] |

**ERA5 at each buoy** (added 2026-11-14, package rebuilt from commit
`5198fab`): one global wave-height request per month serves all buoys
(`scripts/buoycam_era5.py`); each buoy takes the grid point with sea data
nearest it (median 19 km away, at most 44 km: 44065 off New York). Its series
is interpolated to the strip's time and corrected with a line fitted on the
training strips: Hs = −0.05 + 1.09 · Hs_ERA5, nearly the identity, so offshore
ERA5 is close to the buoys before any correction. ERA5 ends 2026-11-09 16:00
(about 5 days behind real time; all of it still the preliminary ERA5T), so it
covers 92 % of the strips and only 41 % of the time split's test weeks;
re-running the download after 2026-11-20 completes them. **On the test buoys
ERA5 scores 0.74 / 0.77: that is the bar.** Its interval is wide because 12
buoys are few.

## Model

`scripts/train_sea_state.py --split station` (or `time`) with the config's
`model` section, added on 2026-11-14 after the package was built (the data
sections are unchanged). `MultiViewNet`
(`wave_analysis.models.sea_state`): one `convnext_tiny` (ImageNet weights)
reads each of the six views at 270 × 480 px (44008's 216 × 384 views scaled
up); the mean feature of the **usable** views (the package's rule: mean grey
≥ 20, deviation ≥ 4) goes to a four-class head, so a dark or blank view
cannot change the answer (test `test_strip_views_and_masked_mean`). The views
share the weights, so their order (compass heading) means nothing to the
model. Training as at Currituck: class-weighted cross-entropy (inverse
frequency over the training strips, mean 1), AdamW with a one-epoch warm-up
and cosine decay, 10 epochs, batches of 32 strips (192 views), fp32; each
view gets its own random crop (scale 0.85–1), tilt (±5°: the horizon tilts
with the buoy) and colour change; no mirroring. The epoch is chosen by
macro-F1 on val; the test buoys are predicted once.

Two runs on the station split: the config as written, and the same with
`--class-weights none` (every strip counts the same), to show what the
weighting does. Then the time split.

## The bar (fixed 2026-11-14, before any training)

1. **Station split, test buoys:** the camera must beat **ERA5 at each buoy**
   on the same strips, in balanced accuracy *and* macro-F1, with the 95 %
   interval of the gain above zero (paired bootstrap over whole test buoys,
   `paired_block_bootstrap_ci`). ERA5 here: the nearest grid point with sea
   data, interpolated to the strip's time, corrected with a line fitted on
   the training strips (`scripts/buoycam_era5.py`, then the package).
2. **Time split, test weeks:** the same against ERA5, and against each buoy's
   own majority class.
3. The unweighted run is reported beside the weighted one; it is not a
   second chance at the bar: the config's run (weighted) is the one judged.
4. ERA5 lags real time by about 5 days, so the last test days are covered
   only once the data store has them; the comparison uses the strips ERA5
   covers, and its coverage is reported.

## First result: station split (2026-11-17)

Run `buoy-station-20261117t2310`: the config as written (weighted, seed 0,
10 epochs, epoch 9 chosen on validation macro-F1), one shared H200, 55 min,
commit `5145b11` (`deploy/dgx/run.sh`). Scored with `scripts/buoycam_bar.py`
on the 12 test buoys, on the 5,642 strips (91 %) that ERA5 covers:

| | balanced accuracy | macro-F1 |
|---|---|---|
| camera | 0.701 | 0.648 |
| ERA5 at each buoy | 0.735 | 0.771 |
| gain (95 % interval over buoys) | −0.034 [−0.154, +0.013] | −0.123 [−0.385, −0.026] |

**The bar is not met.** ERA5 is better on macro-F1 (interval below zero);
balanced accuracy is level within the interval. The camera mostly calls the
sea one class too calm: 29 % of moderate strips are called slight and 23 % of
rough ones moderate (ERA5: 92–93 % recall on both). It finds smooth seas more
often than ERA5 (recall 0.56 against 0.26, 85 strips), at the price of calling
107 slight strips smooth. The weakest test buoy is 41043 (49 % correct).

### Repeat, unweighted run and time split (2026-11-18)

Three runs side by side on three shared H200s, commit `1be691a`, 40–55 min
each. Test strips ERA5 covers; gains with 95 % intervals from the paired
bootstrap of the config (whole buoys, or buoy-days for the time split):

| run | balanced accuracy | macro-F1 | against | bar |
|---|---|---|---|---|
| station, weighted, seed 0 (the judged run) | 0.701 | 0.648 | ERA5 0.735 / 0.771 | not met |
| station, weighted, seed 1 | 0.674 | 0.534 | ERA5 0.735 / 0.771 | — |
| station, unweighted, seed 0 | 0.694 | 0.667 | ERA5 0.735 / 0.771 | — |
| time split, weighted (41 % of test strips have ERA5 so far) | 0.691 | 0.696 | ERA5 0.784 / 0.798: −0.09 [−0.15, −0.04] / −0.10 [−0.15, −0.04] | not met |
| time split, weighted, all test strips | 0.714 | 0.715 | own-buoy majority 0.390 / 0.354: +0.32 [+0.29, +0.36] / +0.36 [+0.32, +0.40] | met |

- **Seed matters more than anything tried.** The same configuration with
  seed 1 scores macro-F1 0.116 lower on the test buoys ([−0.142, −0.079]),
  though it scored higher on the validation buoys (0.753 against 0.743). It
  leans much further toward calm seas: 41 % of test strips called too calm
  (seed 0: 23 %), 413 strips called smooth of which 80 are. How far the model
  drifts on unseen buoys is not settled by training; and 12 validation buoys
  do not pick the epoch that transfers best.
- **The class weights are not the cause of the lean.** Unweighted minus
  weighted (seed 0): balanced accuracy −0.003 [−0.022, +0.017], macro-F1
  +0.022 [−0.009, +0.050]; too-calm share 19 % against 23 %.
- **Time split:** at buoys it has seen, in later weeks, the camera is far
  better than each buoy's usual sea state, but below ERA5 on the 41 % of test
  strips ERA5 covers so far. Item 2 of the bar needs both, so it is not met
  either way; the full ERA5 comparison follows the download after
  2026-11-20 (rescore `predictions.csv`, no retraining).

**In short:** from these six views alone, the camera tells sea states apart
well above chance at buoys it has never seen (balanced accuracy about 0.70
against 0.25), but less well than ERA5, and its calibration across buoys
varies from run to run. At Currituck the camera added skill *to* a hindcast
rather than replacing it; the same test here (camera with ERA5 against ERA5
alone) would need its own bar, fixed before it is run.

## Why it calls the sea too calm: it sees the wind sea, not the swell (2026-11-18)

Exploratory check on the saved predictions (`scripts/buoycam_swell_check.py`,
research question Q-M8; questions fixed before looking). NDBC splits each
record's spectrum into swell (SwH) and wind-wave (WWH) height; *swell share* is
SwH² / (SwH² + WWH²). Judged run, 6,156 test strips at 12 buoys; intervals by
bootstrap over buoys:

1. **More swell, more "too calm".** Within each true class, the camera calls
   the sea at least one class too calm on 12 % of the third of strips with
   least swell, 24 % of the middle third and 34 % of the third with most:
   +0.22 [+0.16, +0.27]. **Within each buoy and class**, so that place, light
   and camera are fixed: +0.20 [+0.15, +0.25]. ERA5 on the same strips: flat
   (2 %, 2 %, 4 %; +0.01 [−0.02, +0.04]).
2. **Its calls follow the wind waves.** Standardised slopes of the camera's
   expected class on log wind-wave and log swell height: 0.68 and 0.37
   (difference +0.16 to +0.46). The true class depends on both about equally
   (0.58 and 0.49).
3. **Per buoy.** Over the 24 buoys held out of training (validation and test),
   the more swell a buoy carries, the further the camera under-calls there
   (Spearman −0.89). The test buoys carry more swell (mean share 0.67) than
   the training buoys (0.62), and the validation buoys far less (0.40): the
   epoch chosen on validation is chosen where swell matters least, which fits
   the seed-to-seed spread on the test buoys.

The same holds in all four runs (too-calm gain +0.22 to +0.32; within buoys
+0.15 to +0.28), including the time split at buoys seen in training, where the
model has learned each buoy's usual swell and its calls follow swell and wind
waves about equally (0.54 and 0.58).

**Reading.** Six small snapshots show whitecaps and short, steep wind waves;
long swell is low and smooth and nearly invisible in them. The camera measures
the wind sea, and guesses the swell from what it learned is typical. Wave
models do the opposite well: they carry swell from far away. This points the
next experiments at a division of labour (camera for the wind sea, the wave
model for the swell) rather than at more image-only models of total height.

## v1: the camera for the wind sea (bars fixed 2026-11-18, before any v1 training)

The v0 runs show the camera sees the wind sea and guesses the swell. v1 asks it
for what it sees, and then asks whether that adds to a wave model, which
carries the swell. Decided with Steven on 2026-11-18.

**Package `ndbc_buoycam_windsea_v1`** (`configs/experiments/ndbc_buoycam_windsea_v1.yaml`):

- **Same strips, period, light rules and partitions as v0.** Every buoy keeps its
  v0 partition (`split.station.partitions_from`): without a wind-wave record, two
  training buoys (44014, 46061) drop out, and a fresh draw would have moved the
  others. Every v1 strip has the same partition, path and image bytes as in v0.
- **Label:** NDBC's wind-wave height (WWH, its split of each record's spectrum),
  as the mean of the records centred within 30 min (ADR 0010, as Hs), in the WMO
  bounds: 37,652 strips at 77 buoys, smooth 35 %, slight 35 %, moderate 23 %,
  rough or more 7 %. Each strip also carries the total height, the swell height
  and the buoy's own wind (8-min averages within 30 min; 99.9 % of strips).
- **Baselines (no image):** ERA5's own wind-wave height (`shww`) at the buoy,
  corrected with a line fitted on the training strips; and the **buoy-wind
  control**, the buoy's measured wind speed through a power law fitted on the
  training strips (test buoys: balanced accuracy 0.49, macro-F1 0.47).
- **Model:** as v0, without the random rotation (it hid the buoy's tilt). Three
  seeds (0, 1, 2), **judged by their averaged class probabilities**
  (`scripts/ensemble_runs.py`), because one seed moved v0's test macro-F1 by 0.12.

**Test 1: is the camera a better wind-sea sensor than ERA5?** On the 12 test
buoys, on the strips ERA5 covers, the averaged camera must beat ERA5's wind-wave
height in balanced accuracy **and** macro-F1, each gain's 95 % interval above
zero (paired bootstrap over whole buoys; `scripts/buoycam_bar.py`). Reported
beside it, not part of the bar: the buoy-wind control. If the camera does not
beat the control, its value is where no anemometer exists (ships, coasts), not
at buoys.

**Test 2: does the camera add to ERA5 for the total sea state?**
(`scripts/buoycam_fusion.py`, choices in its header.) A multinomial logistic
combiner of the camera's wind-wave class log-probabilities and the logs of
ERA5's total, wind-sea and swell heights, fitted on the val buoys, scored once
on the test buoys, against the same combiner on the three ERA5 heights alone
fitted on the same strips. The target is the v0 total sea-state class. The
camera + ERA5 combiner must gain in balanced accuracy **and** macro-F1, each
interval above zero (bootstrap over buoys). Reported beside it: v0's ERA5
baseline. Caveat known in advance: the val buoys carry less swell (mean share
0.40) than the test buoys (0.67).

**Time split:** the same rules (bar: ERA5's wind-wave height and each buoy's
majority class), run after the station split.

### v1 results (2026-11-18)

Three seeds on three shared H200s, commit `47af9eb`, about 55 min each (epochs
7, 10 and 9 chosen; validation macro-F1 0.81 each); the seeds agree with their
average on 95 % of strips. ERA5 wind-sea and swell heights through 2026-11-13
(98 % of strips).

**Test 1: met.** On the 12 test buoys (6,024 strips ERA5 covers):

| | balanced accuracy | macro-F1 |
|---|---|---|
| camera, three seeds averaged | 0.795 | 0.807 |
| ERA5 wind-wave height (line on training strips) | 0.694 | 0.709 |
| gain (95 % interval over buoys) | **+0.101 [+0.053, +0.159]** | **+0.098 [+0.060, +0.151]** |
| buoy-wind control (all 6,157 strips) | 0.486 | 0.468 |

Each seed alone also meets the bar (gains +0.08 to +0.12). The camera beats the
buoy's own anemometer by +0.31 balanced accuracy [+0.27, +0.34] and +0.34
macro-F1. **At buoys it has never seen, the camera reads the wind-sea state
better than ERA5 does at the same place and hour, and far better than the wind
measured on the same hull.** Its calls follow the wind-wave height (standardised
slope 0.94) and not the swell (0.02), as the true class does (0.93, 0.01). A
smaller lean to calm with more swell remains (+0.10 within classes), which ERA5's
wind-sea height shows as much (+0.14), so it is more likely in NDBC's split of
the spectrum than in the camera.

**Test 2: not met.** Camera + ERA5 against ERA5 alone, total sea state, both
combiners fitted on the val buoys, scored on 6,019 test strips:

| | balanced accuracy | macro-F1 |
|---|---|---|
| camera + ERA5 | 0.854 | 0.796 |
| ERA5 alone, refitted on val | 0.798 | 0.808 |
| gain (95 % interval) | +0.056 [+0.034, +0.095] | −0.012 [−0.025, +0.045] |

Adding the camera raises the F1 of slight (0.80 → 0.82), moderate (0.88 → 0.90)
and rough (0.89 → 0.91) seas and the recall of smooth ones (0.64 → 0.82), but
calls 101 slight strips smooth, so smooth-sea F1 falls (0.67 → 0.55); with 88
smooth test strips weighing a quarter of macro-F1, the two cancel. Balanced
accuracy, which counts recall only, rises clearly. This is the pattern of the
Currituck fusion (balanced accuracy up, macro-F1 level). Whether a combiner
without class weights would keep the gains without the smooth false alarms is
a new question for a new pre-set test, not a rescue of this one.

**Test 2b (fixed 2026-11-18 after Test 2 was scored, before 2b was run).**
Everything as Test 2 (same camera average, features, val strips, test strips
and bar), except that **neither combiner uses class weights**
(`scripts/buoycam_fusion.py --class-weights none`, output in
`fusion_unweighted/`). The camera + ERA5 combiner must gain over ERA5 alone in
balanced accuracy **and** macro-F1, each 95 % interval above zero (bootstrap over
whole test buoys). Test 2's result is what suggested 2b, and it is scored on the
same test buoys, so a pass here is a follow-up finding, not a confirmation: that
needs strips the choice has not seen (later months, or the time split).

**Test 2b result: not met.** 6,019 test strips, 12 buoys:

| | balanced accuracy | macro-F1 |
|---|---|---|
| camera + ERA5, unweighted | 0.783 | 0.813 |
| ERA5 alone, unweighted | 0.730 | 0.769 |
| gain (95 % interval) | +0.053 [−0.007, +0.082] | +0.044 [−0.024, +0.060] |

Without weights the smooth false alarms go (smooth precision 0.42 → 0.81) and
both gains are positive, but neither interval clears zero. The gain is now almost
all in calm seas: ERA5 alone finds a quarter of the smooth strips (recall 0.26),
camera + ERA5 half (0.50; smooth F1 0.41 → 0.62), while slight, moderate and rough
stay level (within 0.01). All smooth test strips are at two buoys (44065: 69,
42057: 25), so a gain that lives in that class moves with which buoys a bootstrap
draws.

**Reading of Tests 2 and 2b together.** Adding the camera to ERA5 raises the
total-sea-state call by about +0.05 either way (balanced accuracy in Test 2, both
metrics in 2b), and the gain sits in calm seas, which ERA5 misses. On these 12
test buoys, two of which carry every calm strip, that is not yet clear of chance.
More calm-sea buoys or months, not another combiner, would settle it; no further
variant is tried on this test set.

## v2: heights in metres, against published results (fixed 2026-11-18, before v2 training)

Steven's goal: the best model for this application, measured against the
published work he is compared with. One model now gives heights in metres, so
every paper's measure can be computed from the same predictions.

**Package `ndbc_buoycam_heights_v2`:** v0's 41,091 strips, partitions, image
bytes and total-height labels (checked), plus the window-mean wind-wave and
swell heights (92 % of strips), the buoy's wind, and each view's de-mounted
horizon roll and pitch (`views.csv`; each camera's median over its own strips
removed, no labels used).

**Model** (`wave_analysis.models.wave_heights.MultiViewRegressor`,
`scripts/train_wave_heights.py`): one ConvNeXt-Tiny per view, a one-layer
transformer over the usable views, the masked mean, three outputs (log total,
wind-wave and swell height; Huber loss on standardised logs); no random crop or
rotation, so the horizon keeps the buoy's real height and tilt; the epoch with
the lowest validation RMSE of the total height is kept. **Arm A:** images only.
**Arm B:** each view's horizon roll and pitch join its image feature. Three
seeds per arm, each arm judged by the mean of its seeds' log heights
(`scripts/compare_papers.py`).

**Published reference points** (other data and protocols: set side by side with
our 95 % intervals over test buoys, not tested against):

| Paper | Their task and data | Their number | Ours, from the same predictions |
|---|---|---|---|
| Kamagata et al. 2026 | H_s in metres from coastal video, 5 held-out videos, label source not stated | RMSE 0.586 m, R² 0.107, Pearson 0.436 | RMSE, R², Pearson of H_s at the 12 unseen buoys |
| Choi et al. 2020 | NDBC buoy cameras (one view), calm (< 1.52 m) vs rough (> 3.05 m), same buoys in train and test | 84.05 % accuracy | the same two classes at the unseen buoys (predicted H_s split at 2.285 m) |
| Kim et al. 2023 | one fixed camera, 5 classes of 0.5 m of the *average* wave height (0.75–3.25 m), 30 one-hour sessions, random frame split | 78.3 % snapshot, 93.2 % 180-s video | the same classes on 0.631 × our H_s (Kim's Table 1 ratio; definition verified 2026-11-18, before v2 was scored) |
| Yang et al. 2026 | ship camera + attitude, **simulation** | RMSE 0.032 m (simulated) | no number to set beside it; Bar V2-1 is its question on real data |

**Bars** (paired bootstrap over the 12 test buoys, 1,000 draws):

- **V2-1, does the buoy's motion help (Yang's question, real data)?** Arm B must
  lower the RMSE of the total height **and** of the wind-wave height against
  arm A, each interval of the difference below zero.
- **V2-2, camera against ERA5 in metres.** For each arm: the wind-wave RMSE must
  be below that of ERA5's wind-wave height (line fitted on the training strips),
  and, reported the same way, the total-height RMSE against ERA5's total
  height. Expected from v0 and v1: the camera wins the first and not the
  second.

### Test 3: the best estimate from everything at the buoy (fixed 2026-11-18, before v2 was scored)

`scripts/buoycam_combined.py`: a ridge regression (standardised features, alpha 1.0)
of log H_s and of log wind-wave height on the camera's three v2 heights (the arm with
the lower **validation** RMSE of H_s), ERA5's total, wind-sea and swell heights at the
buoy, and the buoy's own wind speed and gust; fitted on the val buoys, scored once on
the test buoys. **Bar:** camera + ERA5 + wind must beat **ERA5 + wind** (the same ridge
without the camera) in RMSE of H_s **and** of the wind-wave height, each interval of
the difference below zero (paired bootstrap over test buoys). It asks whether the
camera adds anything that an anemometer and a wave model at the same buoy do not.
Reported beside it: ERA5 alone, camera + wind, the camera alone.

### Bar B1: a better backbone (fixed 2026-11-18, before any B1 run)

The backbone review (`reports/Vision backbones for wave cameras.md`, notes in
`research_notes/`) found that among pretrained networks of similar size, published
wave studies moved by 0.1-2 points when the backbone changed, and that self-supervised
DINOv3 features keep fine detail and survive image corruption far better than
ImageNet-supervised ones (frozen ConvNeXt-Tiny: depth error 0.448 against 0.666; mean
corruption error 19.6 for DINOv3 ViTs against 24.1 for DINOv2). The cleanest test is
the like-for-like swap: **the same ConvNeXt-Tiny, DINOv3 weights**
(`timm:convnext_tiny.dinov3_lvd1689m`, distilled from DINOv3, 2025; Meta's DINOv3
licence: research use with acknowledgement) in place of the ImageNet weights, with v2
arm A's recipe unchanged (images only, full fine-tuning, 10 epochs, the epoch chosen on
validation RMSE of H_s), three seeds averaged.

**Bar B1:** the DINOv3 average must lower the test RMSE of H_s **and** of the wind-wave
height against v2 arm A's average, each interval of the difference below zero (paired
bootstrap over the 12 test buoys). Every published measure is reported for both
(`scripts/compare_papers.py`). Frozen probes of larger self-supervised ViTs (DINOv2
ViT-B/14, DINOv3 ViT-B/16) and LP-FT are follow-ups, judged on the validation buoys
first.

## The camera as a motion sensor: the horizon follows the buoy's slope (2026-11-18)

Exploratory, nothing trained (`scripts/buoycam_horizon.py`, questions fixed in its
header before looking; research question Q-M5). In every usable view of the v0
package a straight horizon is fitted (`wave_analysis.processing.horizon`: the
steepest bright-to-dark step in each column, RANSAC line); each camera's mounting is
removed by subtracting its median angle and height over its strips; a strip's motion
is the root mean square of what remains over at least three clear views (median roll
2.3°). The buoy's slope is its own: the mean-square slope of its measured spectrum
(sum of S(f) k², deep water), averaged over the records within 30 min.

- **Coverage:** 209,615 of 236,973 usable views (88 %) give a clear horizon;
  38,860 strips have three or more.
- **A. Motion follows slope.** Within each buoy, strips whose horizons roll more come
  from steeper seas: Spearman 0.56 [0.54, 0.58] over 76 buoys, positive at 75; for
  the horizon's height (pitch), 0.56 [0.54, 0.58], positive at all 76. One hourly
  photograph strip, by geometry alone, measures what the buoy's motion sensor
  measures.
- **B. It too follows the wind sea.** The motion correlates with the wind-wave height
  (roll 0.56, pitch 0.56) and hardly with the swell height (0.05 each). Wave slope
  grows with the square of the wavenumber, so long swell tilts the hull little; the
  tilt does not fill the camera's swell gap.
- **C. Six moments, not one.** The de-mounted roll of adjacent views hardly agrees
  (correlation 0.03 to 0.04; opposite views −0.06 to −0.08), as it would if a rigid
  ring of cameras fired at one instant. The views are most likely taken one after
  another, so a strip holds up to six separate samples of the motion.

This answers the platform-motion question for the buoy cameras: the camera can serve
as its own attitude sensor, which matters on platforms with a camera and no motion
sensor. It also means v0's ±5° rotation augmentation erased a real signal (v1 trains
without it).

## Status

- [x] Pairing and package (2026-11-14, `7b8a93c`): 41,091 strips at 79
      buoys, leakage audits pass, `SHA256SUMS` checked, captions cut.
- [x] Majority baselines scored.
- [x] ERA5 baseline at each buoy (2026-11-14): 0.74 / 0.77 on the test
      buoys. To complete the last test days: download again after 2026-11-20.
- [x] Training code for six-view strips, a CPU test of the whole path
      (2026-11-14).
- [x] Training, station split (2026-11-17, on FAU's H200 server over SSH,
      [cluster guide](../operations/cluster.md#gpu-server-over-ssh-faus-h200-server-the-labs-way)):
      bar not met (above). The desktop GPU stays unused until its cooling is
      fixed.
- [x] The unweighted station run, the time split and the station run with
      seed 1 (2026-11-18, three GPUs at once): above.
- [ ] Time split against ERA5 on all test days: download ERA5 again after
      2026-11-20 and rescore the saved predictions.
- [x] v1, the camera for the wind sea (2026-11-18): Test 1 met (the camera
      beats ERA5's wind-wave height at unseen buoys), Test 2 not met (camera +
      ERA5: balanced accuracy up, macro-F1 level), Test 2b not met (unweighted:
      both about +0.05, intervals reach below zero; the gain is in calm seas,
      found at two test buoys).
- [ ] v1 time split (three seeds), after the ERA5 download of 2026-11-20.
- [ ] Relabel from NDBC's quality-controlled annual files when they appear.
