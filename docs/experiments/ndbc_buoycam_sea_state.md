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
  groups uses every buoy as a test buoy once: built 2026-11-18 ("Grouped
  cross-validation", below).
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

### v2 results (2026-11-18)

Three seeds per arm (seeds 0 and 1 at `acbc5bf`, seed 2 at `fc05193`, the training
code unchanged between them), each arm the mean of its seeds' log heights; scored once
with `scripts/compare_papers.py` at the 12 unseen test buoys (6,183 strips; intervals
from a bootstrap over whole test buoys).

| Measure (H_s unless named) | Arm A, images | Arm B, + horizon | Published |
|---|---|---|---|
| RMSE | **0.450 m** [0.390, 0.509] | 0.488 m [0.415, 0.555] | Kamagata 0.586 m |
| R² | **0.81** [0.73, 0.86] | 0.78 [0.67, 0.84] | Kamagata 0.107 |
| Pearson | **0.915** [0.890, 0.933] | 0.910 [0.883, 0.930] | Kamagata 0.436 |
| Bias | −0.16 m | −0.23 m | |
| Calm vs rough (3,190 strips) | **99.1 %** [97.9, 99.7] | 98.7 % [96.9, 99.6] | Choi 84.05 % (same buoys and days) |
| Kim's five classes (4,791 strips) | 69.4 % [61.8, 77.9] | 68.0 % [59.4, 77.3] | Kim 78.3 % snapshot, 93.2 % video (random frames) |
| Wind-wave height RMSE | 0.219 m (R² 0.93) | 0.223 m | |
| Swell height RMSE | 0.597 m (R² 0.54) | 0.650 m | |

- **V2-1 not met.** Arm B minus arm A: H_s +0.038 m [+0.018, +0.057] (the horizon
  makes the total height *worse*), wind-wave +0.005 m [−0.001, +0.010]. Arm B was the
  better arm on the validation buoys (0.449 against 0.470 m) and the worse one on the
  test buoys, the same val/test gap in swell seen since v0. On real buoys the measured
  motion adds nothing the images do not already carry; Yang 2026's gain from attitude
  (simulation) does not carry over.
- **V2-2: the camera beats ERA5 on the wind sea, not on the total.** Wind-wave height
  0.215 against ERA5's 0.341 m (−0.126 [−0.187, −0.065]; arm B −0.122). Total height
  0.434 against 0.267 m (+0.167 [+0.091, +0.242]) and swell 0.582 against 0.361 m:
  ERA5 wins both, as expected from Q-M8.
- **Against the papers.** At buoys it never saw, the camera alone is ahead of
  Kamagata 2026 on every measure they report (RMSE, R², Pearson; their data are shore
  video at five held-out clips, ours open-ocean buoys, so this is a side-by-side, not a
  test) and ahead of Choi 2020's calm/rough accuracy on a harder split. It is **below
  Kim 2023's snapshot accuracy** (69 % against 78 %; the upper end of our interval,
  77.9 %, falls just short). Kim's test frames come from the same hours as training
  frames and each hour has one label, which our split forbids; whether a random split of
  our strips would pass 78 % is a separate, labelled-as-such check. The camera calls the
  total height too low (bias −0.16 m), the swell it cannot see.

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

**Result (2026-11-18): not met.** The rule picked arm B (validation RMSE 0.449 against
0.470 m). Ridge fitted on 6,172 val strips, scored on 6,045 test strips at 12 buoys:

| Target | camera + ERA5 + wind | ERA5 + wind (the bar) | ERA5 alone | camera alone | difference to the bar |
|---|---|---|---|---|---|
| H_s | 0.258 m | 0.259 m | 0.259 m | 0.473 m | −0.001 [−0.052, +0.049] |
| wind-wave height | **0.199 m** | 0.286 m | 0.306 m | 0.220 m | **−0.087 [−0.114, −0.060]** |

For the total height the camera adds nothing once ERA5 is there; for the wind sea it
cuts the error of ERA5 + wind by 30 %, an interval well below zero. The bar asked for
both, so it is not met. The division of labour is clear: ERA5 carries the swell and so
the total height; the camera carries the wind sea, and the buoy's anemometer adds
little to either (camera + wind 0.205 m against camera alone 0.220 m).

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

**Result (2026-11-18): not met; DINOv3 is worse with this recipe.** Three seeds
(`530b279`; two of them shared one GPU, which changes only the time), averaged and
scored with `compare_papers.py` against arm A:

| | ImageNet (arm A) | DINOv3 | DINOv3 minus ImageNet |
|---|---|---|---|
| H_s RMSE | 0.450 m | 0.489 m | +0.039 [+0.008, +0.073] |
| Wind-wave RMSE | 0.219 m | 0.244 m | +0.025 [+0.011, +0.039] |
| H_s R², Pearson | 0.81, 0.915 | 0.78, 0.904 | |

The DINOv3 seeds were also worse on the validation buoys (best epochs 0.48-0.52 m), so
this is not a val/test swap. The review's caveat applies: the recipe (every weight
learning at 3e-4 from the first step) was tuned for ImageNet weights and is the kind
that can wash out pretrained features; "DINOv3 does not help" is not yet shown, only
"not with this recipe". Bar B2 tests the gentle recipe.

### Bar B2: a recipe for pretrained weights (fixed 2026-11-18, after B1, before any B2 run)

**Gentle recipe:** the backbone frozen for the first 2 epochs (LP-FT, `--lp-epochs 2`),
then learning at a tenth of the head's rate (`--backbone-lr-scale 0.1`: 3e-5 against
3e-4); all else as arm A (images only, 10 epochs, the epoch chosen on validation RMSE of
H_s). Two arms of three seeds: **ImageNet-gentle** and **DINOv3-gentle**.

**Bar B2:** DINOv3-gentle must lower the test RMSE of H_s **and** of the wind-wave
height against arm A (the best model so far), each interval below zero (paired
bootstrap over the 12 test buoys). Reported beside it: ImageNet-gentle minus arm A (the
recipe alone) and DINOv3-gentle minus ImageNet-gentle (the weights at one recipe).

**Result (2026-11-18): not met, by a hair; DINOv3-gentle is the best model so far.**
Three seeds per arm (`0c09717`), averaged, 12 unseen test buoys:

| | Arm A (ImageNet, full fine-tuning) | ImageNet-gentle | DINOv3-gentle |
|---|---|---|---|
| H_s RMSE | 0.450 m | 0.524 m | **0.440 m** [0.381, 0.501] |
| H_s R², Pearson | 0.81, 0.915 | 0.75, 0.900 | **0.82, 0.921** |
| Wind-wave RMSE | 0.219 m | 0.225 m | **0.214 m** |
| Swell RMSE (R²) | 0.597 m (0.54) | 0.726 m (0.32) | **0.572 m (0.58)** |
| Calm vs rough (Choi) | 99.1 % | 98.2 % | 99.2 % |
| Kim's five classes | 69.4 % | 66.4 % | 70.2 % [62.7, 78.7] |
| WMO classes, balanced accuracy | 0.62 | 0.60 | 0.72 |
| Validation H_s RMSE | 0.470 m | 0.439 m | 0.440 m |

- **Bar B2:** DINOv3-gentle minus arm A, H_s −0.010 m [−0.024, +0.001], wind-wave
  −0.004 m [−0.010, +0.001]. Both intervals reach zero: not met.
- **The recipe alone** (ImageNet-gentle minus arm A): H_s +0.075 m [+0.052, +0.095].
  ImageNet features have to be retrained to see waves.
- **The weights at one recipe** (DINOv3-gentle minus ImageNet-gentle): H_s −0.085 m
  [−0.107, −0.061], wind-wave −0.011 m [−0.016, −0.005].
- **Reading.** Weights and recipe interact. DINOv3's features serve almost as they are
  and full fine-tuning (B1) damages them; ImageNet's need full fine-tuning. Each with
  its own recipe, the two are level within about 0.01 m: on these strips the backbone
  is worth about as much as the review found in published wave studies (0.1-2 points).
  DINOv3-gentle is the best point estimate on every measure, and on Kim's classes the
  upper end of its interval now passes Kim's 78.3 %. The validation buoys do not rank
  the arms as the test buoys do (ImageNet-gentle is best on validation and worst on
  test), the low-swell validation set seen since v0; choices made on them carry that
  risk.

### Bar B3: the backbone the review recommends (fixed 2026-11-18, before B2 was scored)

B1 and B2 swap the weights of one small CNN. The review's first choice for the strips
is a self-supervised ViT: **DINOv3 ViT-B/16** (`timm:vit_base_patch16_dinov3.lvd1689m`,
86 M parameters, web images; DINOv3 licence), each view at its native 270 x 480 (17 x 30
patches, padded to 272), the patch tokens averaged, the same view transformer and
heads. Two arms of three seeds: **frozen** (`--lp-epochs 10`: the backbone never trains,
a probe of its features) and **gentle** (B2's recipe). The arm with the lower
validation RMSE of H_s (its seeds' average) is B3's candidate.

**Bar B3:** the candidate must lower the test RMSE of H_s **and** of the wind-wave
height against arm A, each interval below zero (paired bootstrap over the 12 test
buoys). The other arm and every paper's measure are reported beside it.

**Result (2026-11-18): not met.** Runs at `bb1a568`. Validation picked the gentle arm (H_s 0.475 m against
the frozen arm's 0.550 m). On the test buoys, three seeds averaged:

| | Arm A | ViT-B/16 gentle (candidate) | ViT-B/16 frozen |
|---|---|---|---|
| H_s RMSE | 0.450 m | 0.455 m (R² 0.81, Pearson 0.910) | 0.560 m (R² 0.71) |
| Wind-wave RMSE | 0.219 m | 0.212 m | 0.257 m |
| Kim's five classes | 69.4 % | 69.6 % | 64.4 % |
| Minus arm A, H_s | | +0.005 [−0.009, +0.020] | +0.111 [+0.086, +0.139] |
| Minus arm A, wind-wave | | −0.007 [−0.016, +0.001] | +0.039 [+0.023, +0.055] |

Frozen features, even DINOv3's, are well short: the camera has to be trained to see
waves. Fine-tuned gently, the 86 M-parameter ViT lands level with the 28 M ConvNeXt.

**What B1-B3 show together.** Once each backbone gets a recipe that suits it, the
choice of backbone moves H_s by about 0.01 m (0.440-0.455 m), inside the intervals; the
recipe moves it by up to 0.11 m. The best model so far is DINOv3 ConvNeXt-Tiny with the
gentle recipe (0.440 m). Larger gains have to come from elsewhere: the swell the camera
cannot see (ERA5 in Test 3), more buoys and calm seas, or motion (the video test).

### Protocol check R: the published split on our data (fixed 2026-11-18, before any run on it)

Not a bar and not our result. Choi 2020 split examples at random with the same buoys
and days on both sides; Kim 2023 split frames at random inside 30 one-hour sessions.
Our numbers hold out whole buoys 300 km away. To show how much the split itself is
worth, v2 arm A's recipe (images only, ImageNet ConvNeXt-Tiny, three seeds) is trained
once more on a **random split of the same 41,091 strips** (80/10/10, seed 0; package
`ndbc_buoycam_heights_v2r` = v2 plus a `split_random` column, built by
`scripts/buoycam_random_split.py`; config `ndbc_buoycam_heights_v2r.yaml`), so
neighbouring hours of one buoy and day sit in training and test, as in those papers.
Every paper's measure is reported for it beside the unseen-buoy result. The headline
stays the unseen-buoy number; the difference is what the random protocol adds.

**Result (2026-11-18).** Three seeds (runs `buoy-random-20261118t1704`/`t1715`, commit `1163db2`),
4,109 random test strips at all 79 buoys:

| | Unseen buoys (arm A) | Random split |
|---|---|---|
| H_s RMSE | 0.450 m | 0.433 m [0.378, 0.497] |
| H_s R², Pearson | 0.81, 0.915 | 0.87, 0.933 |
| Bias | −0.16 m | −0.06 m |
| Calm vs rough (Choi) | 99.1 % | 99.2 % |
| Kim's five classes | 69.4 % | 68.8 % [65.7, 71.8] |
| Swell R² | 0.54 | 0.71 |

The random split adds little: the error falls by 0.02 m and the bias shrinks (the
training set now holds every buoy's swell climate), but Kim's five-class accuracy does
not move. **The gap to Kim's 78 % is not the split.** It is most likely the task: one
camera at one site, 30 one-hour sessions and their mean height, against 79 buoys
across oceans. (The R² is higher partly because the random test set spans all buoys,
so it holds more variance.)

## Grouped cross-validation: every buoy a test buoy once (fixed 2026-11-18, before any run on the folds)

Rank 1 of the [architecture review](https://github.com/StevenFAU/Wave_Analysis/blob/main/reports/Wave%20camera%20architectures%20beyond%20backbones.md).
Every bar so far was decided by 12 test buoys, and the 12 validation buoys that chose
each epoch carry far less swell (share 0.40) than the test buoys (0.67). B1-B3 ended
with differences of about 0.01 m whose intervals straddle zero. Here each buoy is
predicted once by a model that saw neither it nor any buoy within 300 km, so each
comparison rests on all 79 buoys. **The sealed 12-buoy numbers stay the headline;
the folds are the protocol for choosing and comparing.**

**Package `ndbc_buoycam_heights_v2cv`** (`scripts/buoycam_cv_folds.py`, config
`ndbc_buoycam_heights_v2cv.yaml`): v2's strips, labels and image bytes (hard links), plus
`cv_fold`, `split_fold0` .. `split_fold4` in `labels.csv` and `folds.csv`. The station
split's 43 groups (buoys under 300 km apart, chained) go to five folds balanced by
strips: largest group first, each to the fold with the fewest strips. The US West Coast
group (18 buoys with kept strips, 9,440 strips) is fold 0 on its own. Each fold picks its
epoch on validation groups drawn from its own four training folds: random draws of
groups up to 15 % of all strips (at least 8 buoys), and of 5,000 draws the one whose
swell share is closest to that of its training strips (training labels only).

| Fold | Test buoys (groups) | Test strips | Test swell share | Val buoys (strips) | Val swell share | Train strips (swell share) |
|---|---|---|---|---|---|---|
| 0 | 18 (1: US West Coast) | 9,440 | 0.79 | 14 (6,567) | 0.57 | 25,084 (0.57) |
| 1 | 17 (9) | 7,823 | 0.60 | 12 (6,431) | 0.63 | 26,837 (0.63) |
| 2 | 15 (10) | 8,039 | 0.50 | 13 (6,346) | 0.65 | 26,706 (0.65) |
| 3 | 15 (12) | 8,063 | 0.53 | 11 (6,199) | 0.65 | 26,829 (0.65) |
| 4 | 14 (11) | 7,726 | 0.62 | 13 (6,547) | 0.63 | 26,818 (0.63) |

Fold 0 is the hardest: its model never sees the swell-dominated West Coast (share 0.79)
and trains at 0.57.

**Model:** v2's `MultiViewRegressor` and recipe, plus a **distribution head** beside the
point head: a mean and a variance of each log height, trained with **beta-NLL, beta = 0.5**
([Seitzer et al., ICLR 2022](https://arxiv.org/abs/2203.09168)): the Gaussian likelihood
of each strip weighted by its predicted variance to the power beta, so hard strips are
not neglected. It is added to the point head's Huber loss with equal weight; the epoch
is still chosen on the point head's validation RMSE of H_s. No evidential regression.
Three seeds (0, 1, 2) per fold form the ensemble: the point is the mean of their log
heights (as v2); the distribution is one log-normal with the seeds' first two moments
(mean of their means; mean of their variances plus the variance of their means).

**Scores, out of fold** (`scripts/buoycam_oof.py`; 95 % intervals from a bootstrap over
whole buoys):

- RMSE, bias, R², Pearson and MAE of each height, and every paper's measure
  (`compare_papers.py`'s rules) on the out-of-fold heights.
- **CRPS** of the ensemble's distribution: the average error of the whole predicted
  distribution, in metres. For a single-number forecast it equals the absolute error,
  so it sits directly beside the MAE; lower is better.
- **Seed spread:** each seed's own out-of-fold RMSE (its five fold models).
- **90 % intervals, buoy-level split conformal:** the score of a strip is
  |log y - mu| / sigma; fold k's threshold is calibrated on the out-of-fold scores of the
  other four folds' buoys (held out of the models that scored them), every buoy weighted
  equally ([Lee, Barber & Willett 2023](https://arxiv.org/abs/2306.06342)). That gives
  90 % on average over new buoys if buoys are exchangeable, and no guarantee at any one
  buoy. **Coverage** (the share of true values inside the interval; 0.90 is right) is
  reported over all strips, as a mean over buoys, per buoy and per WMO sea-state class
  of the true height, with the interval's mean width. Beside it: the head's own
  mu ± 1.645 sigma (uncalibrated), the same rule calibrated on each fold's validation
  buoys only, and an unnormalised score (|log y - log point|), which shows what the
  predicted spread adds.
- The camera against ERA5 out of fold (ERA5 corrected by a line on each fold's
  training strips).

**Bar B2-CV.** Two arms, five folds x three seeds each, both with the distribution
head: **arm A** (ImageNet ConvNeXt-Tiny, full fine-tuning: the config as written) and
**DINOv3-gentle** (`--backbone timm:convnext_tiny.dinov3_lvd1689m --lp-epochs 2
--backbone-lr-scale 0.1`). DINOv3-gentle must lower the out-of-fold RMSE of H_s **and**
of the wind-wave height against arm A, each 95 % interval of the difference below zero
(paired bootstrap over all 79 buoys, 1,000 draws). Reported beside it, not part of the
bar: the same bootstrap over the 43 station groups (buoys of one group share weather,
so that interval is the more cautious one), the CRPS difference, the seed spread, and
the out-of-fold RMSE at the 12 sealed test buoys.

**Decided in advance.** If the two arms differ by less than the seed spread (the range of
either arm's three single-seed RMSEs), backbones are settled for the strips and the
work moves to fusion (stage 2) and new outputs (stage 3). The arm with the lower
out-of-fold H_s RMSE is the camera model for stages 2 and 3.

**Result (2026-11-19): not met; DINOv3-gentle is the camera model for stages 2 and 3.**
Five folds x three seeds per arm (30 runs at `f00423f`, `runs/buoycam-cv-stage1-20261119`).
Every buoy is predicted once, by models that saw neither it nor any buoy within 300 km.
RMSE is the typical error in metres; CRPS is the average error of the whole predicted
distribution, in metres (lower is better).

| Out of fold: 79 buoys, 41,091 strips | Arm A (ImageNet) | DINOv3-gentle |
|---|---|---|
| H_s RMSE | 0.550 m [0.504, 0.597] | **0.532 m** [0.488, 0.575] |
| H_s R², Pearson | 0.78, 0.884 | **0.79, 0.892** |
| H_s CRPS | 0.310 m | **0.279 m** |
| H_s RMSE of single seeds | 0.559-0.575 m | 0.539-0.555 m |
| H_s RMSE by fold 0 / 1 / 2 / 3 / 4 | 0.73 / 0.57 / 0.46 / 0.44 / 0.47 | 0.70 / 0.53 / 0.46 / 0.42 / 0.46 |
| H_s RMSE at the 12 sealed buoys | 0.442 m | **0.432 m** |
| Wind-wave RMSE | 0.262 m | **0.260 m** |
| Swell RMSE (R²) | 0.741 m (0.50) | **0.717 m (0.53)** |
| Calm vs rough (Choi) | 98.3 % | 98.4 % |
| Kim's five classes | 65.3 % | 66.7 % [63.9, 69.6] |
| WMO classes, balanced accuracy | 0.64 | 0.67 |

- **Bar B2-CV:** DINOv3-gentle minus arm A, H_s −0.019 m [−0.028, −0.011] (over the 43
  groups [−0.027, −0.005]); wind-wave −0.002 m [−0.008, +0.004]. H_s is lower at all 79
  buoys; the wind-wave interval reaches zero. The bar asked for both: not met.
- **Seed spread:** each arm's three single seeds span 0.016 m of H_s RMSE; the arms differ
  by 0.019 m, just more. So by the rule fixed in advance the backbone is not quite
  settled for H_s; for the wind sea it is (0.002 m apart, seeds span 0.007-0.008 m).
  DINOv3-gentle has the lower H_s RMSE and is the camera model for stages 2 and 3.
- **Reading.** With all 79 buoys, B2's H_s gain on 12 buoys (−0.010 m [−0.024, +0.001])
  is twice as large and clearly below zero. It comes from the swell (−0.024 m
  [−0.034, −0.014]); the wind sea, which the camera reads best, is level. CRPS gains
  more than RMSE (−0.031 m [−0.040, −0.023]): DINOv3's predicted spread fits better,
  not only its centre. The sealed-test headline stays 0.440 m (B2); the out-of-fold
  0.432 m at those 12 buoys agrees with it.
- **Against ERA5** (corrected by a line on each fold's training strips): wind-wave
  camera 0.260 m, ERA5 0.343 m (the camera is better by 0.087 m [0.067, 0.110]); H_s
  camera 0.514 m, ERA5 0.292 m on the 40,240 strips with ERA5 (worse by 0.22 m
  [0.18, 0.26]). v2's finding holds at all buoys: the camera is a wind-sea instrument.
- **Fold 0** (the US West Coast, never seen in training) is the hardest, H_s 0.70 m:
  the camera under-reads a swell it has not learned.

**Intervals** (DINOv3-gentle; coverage is the share of true values inside a 90 %
interval, 0.90 is right):

- The head alone (mu ± 1.645 sigma, uncalibrated) is overconfident at new buoys: it
  covers 0.75 of H_s (arm A 0.59) and 0.88 of wind-wave heights.
- **Buoy-level conformal:** H_s coverage 0.904 over all strips, 0.907 averaged over
  buoys, mean width 1.83 m (arm A needs 2.34 m for 0.894); wind-wave 0.916, width 0.72 m.
- By sea state of the true H_s: smooth 0.80, slight 0.90, moderate 0.92, rough or more
  0.92.
- **Per buoy:** 11 of 79 below 0.80 for H_s (3 for wind-wave), none below 0.50. Six are
  West Coast buoys whose H_s the camera under-reads by 0.6-0.7 m on average (46042,
  46047, 46013, 46069, 46022, 46014); the other five it over-reads by 0.4 m (46081 at
  0.55 the worst, 44013, 42002, 44008, 45002). Conformal promises 90 % on average over
  new buoys, not at each one; a buoy whose level the camera misjudges falls outside.
- Calibrated on each fold's own validation buoys instead, the intervals over-cover
  (0.944) and are wider (2.04 m). Without the predicted spread (an unnormalised score)
  0.908 needs 2.38 m: the spread narrows H_s intervals by about a quarter.

## Stage 2: partition-aware fusion (fixed 2026-11-18, before any fusion was fitted or scored)

Rank 3 of the architecture review. The camera reads the wind sea better than ERA5
(Test 1, V2-2), but adding it to ERA5 + wind left the total height level (Test 3),
because swell carries two thirds of the energy at the test buoys. A combiner should
therefore keep the wave model's swell and replace only its wind sea.

**First, do the partitions add as energies?** (`scripts/buoycam_partition_check.py`.)
NDBC computes WVHT = 4 sqrt(m0) on board and splits the same spectrum at a separation
frequency into swell (SwH) and wind waves (WWH), so WVHT² = SwH² + WWH² should hold up to
the 0.1 m rounding of the three reported values. **It does, on all 295,547 records of the
79 camera buoys** (2026-08-12 to 11-15): every record lies inside the rounding bound
(|WVHT² - SwH² - WWH²| <= 0.1 (WVHT + SwH + WWH) + 0.0075 m²), and for seas of 2 m and
more, sqrt(SwH² + WWH²) is within 4 % of WVHT for 99 % of records (largest gap 5.7 %), as
rounding alone would allow. ERA5's own three heights add the same way to within 1 % at
91 % of buoy-hours (median gap 0.04 %). The rule H_s² = swell² + wind-wave² is therefore
used as a physical relation, not a fitted one.

**Fusion** (`scripts/buoycam_partition_fusion.py`, choices in its header). Inputs per
strip: the stage-1 camera's out-of-fold heights and the spread of its wind-wave height;
ERA5's total, wind-sea and swell heights at the buoy; the buoy's wind speed and gust;
and five gate features that say how well the camera can see (its spread, the sun's
elevation, the number of usable views, their mean grey and mean contrast). Every
combiner that predicts fold k is fitted on the other four folds' strips, whose camera
heights are themselves out of fold.

- **Comparator (the bar):** Test 3's ERA5 + wind ridge, refitted under the folds. Its
  outputs are a total T and a wind sea W.
- **Main candidate, the gated partition swap:** H_s² = T² + k g (b C² - W²) and wind
  sea² = g b C² + (1 - g) W², where C is the camera's wind-wave height, g (0 to 1) is a
  learned gate on the five features that falls back to the comparator where the camera
  cannot see, b rescales the camera, and k (0 to 2) is how much of the wind-sea
  correction reaches the total (k = 1 is the full swap (T² - W²) + C²). With g = 0 the
  candidate is the comparator, so the bar asks only whether the camera's wind sea adds
  information.
- **Beside it, not part of the bar:** the partition rule on raw ERA5 with nothing fitted
  (H_s² = swh² - shww² + C²); the full swap (k = 1), gated and with the gate open; the
  same gated fusion on raw ERA5 with fitted scales; Test 3's full ridge with the camera;
  gradient boosting on every input; ERA5 with a line; the camera alone. TabPFN is not
  run, because its weights need a non-commercial licence to be accepted.

**Bar S2.** The gated partition swap must lower the out-of-fold RMSE of H_s **and** of
the wind-wave height against ERA5 + wind, each 95 % interval of the difference below
zero (paired bootstrap over whole buoys; the camera arm is stage 1's winner on
out-of-fold H_s RMSE). Reported beside it: the bootstrap over station groups, every
other candidate, and where the gate closes (by light and by camera spread).

**Confirmation on new weeks (fixed now, run when the data exist).** The 12 test buoys
have served Tests 2, 2b and 3, so a pass here is confirmed on strips archived after the
package (from 2026-11-14 16:00 UTC), with the same light rules and labels. Each new strip
is predicted by the three seed models of its own buoy's fold, so its buoy stays unseen,
and fused by the combiners fitted for that fold (no refitting). The bar is the same two
conditions. The set needs ERA5, which runs about 5 days behind real time, and a `.spec`
snapshot saved after the weeks it labels. It is large enough to score when ERA5 covers
at least 7 days after the package (about 6,000 strips at 70 or more buoys), which is
expected around 2026-11-26.

**How the confirmation runs (fixed 2026-11-19, before any new strip was scored).**
`scripts/buoycam_fusion_confirm.py` (`build`, `predict`, `score`; commands in its header).
Nothing in it is fitted.

- *Strips and labels:* the pairing of the package (`scripts/buoycam_pairs.py --start
  2026-11-14T16:00Z`), with the realtime snapshot taken on the day of the run; the
  package's filters; each buoy's strips end 1 h before its latest snapshot, so every
  30-min label window is complete. The caption is cut as in the package (the same bytes
  on package strips).
- *Camera:* each strip by its buoy's fold's three seed models, on the desktop's CPU in
  float32 (stage 1 ran in bfloat16 on GPUs). On 200 package strips (40 per fold) the
  merged heights differ from the out-of-fold table by 0.4 % on average for H_s (largest
  3.5 %) and 0.3 % for the wind-wave height, with no bias; their RMSE on those strips
  changes by 0.001 m. About 0.45 s per strip and model. Each model's predictions are
  kept per strip (keyed by the image's checksum), so strips archived before the run
  can be predicted ahead of it; reused and fresh predictions agree to 3 × 10⁻⁷.
- *ERA5 and combiners:* each fold's ERA5 line and fitted combiners are reused, and
  reproduce the out-of-fold tables on the package's strips to the rounding of the files
  (largest difference 0.00005 m). Fed the package's own out-of-fold strips as if they
  were new, `score` returns the out-of-fold result above exactly (the same strips, every
  RMSE within 10⁻⁷ m, the same intervals).
- *When:* the rule is ERA5's coverage (to 2026-11-21 16:00 UTC); all new strips with
  every input up to ERA5's end on the day of the run are scored. The size estimate is
  reported, not used as a gate: since the package, cameras have archived strips at 70
  of the 79 buoys (none at 41002, 44007, 45002, 46006, 46053, 46061, 46071, 46082 and
  46084; 46085 sent no wave record from 11-01 to 11-18), so about 70 buoys are
  expected. A dry run checks and counts but scores nothing. ERA5 runs about 5 days
  behind, so it should reach 11-21 16:00 UTC by 2026-11-27.

**Result (2026-11-19): met out of fold; confirmation on new weeks pending.** Camera:
stage 1's DINOv3-gentle, out of fold (`runs/buoycam-cv-fusion-20261119`). 39,767 strips
at 77 buoys have every input (ERA5 and buoy wind); 36,791 have a wind-wave label.

| Out of fold, 77 buoys | H_s RMSE | minus ERA5 + wind | Wind-wave RMSE | minus ERA5 + wind |
|---|---|---|---|---|
| ERA5 + wind (the bar) | 0.288 m | | 0.293 m | |
| **Gated partition swap** | **0.267 m** | **−0.021 [−0.031, −0.012]** | 0.240 m | **−0.053 [−0.067, −0.039]** |
| Full swap (k = 1), gated | 0.268 m | −0.020 [−0.031, −0.010] | 0.240 m | −0.053 [−0.067, −0.040] |
| Full swap, gate open | 0.278 m | −0.011 [−0.025, +0.002] | 0.253 m | −0.040 [−0.056, −0.023] |
| Test 3's ridge with the camera | 0.274 m | −0.015 [−0.030, −0.000] | 0.229 m | −0.064 [−0.078, −0.050] |
| Gradient boosting, all inputs | 0.297 m | +0.008 [−0.017, +0.034] | **0.226 m** | −0.067 [−0.081, −0.052] |
| Partition rule on raw ERA5, nothing fitted | 0.323 m | +0.035 [+0.010, +0.062] | 0.252 m | −0.041 [−0.058, −0.025] |
| ERA5 alone (line) | 0.292 m | +0.004 [−0.006, +0.013] | 0.337 m | +0.044 [+0.030, +0.060] |
| Camera alone | 0.514 m | +0.226 [+0.185, +0.265] | 0.252 m | −0.041 [−0.058, −0.025] |

- **Bar S2: met.** Both intervals lie below zero, and stay there over the 43 station
  groups (H_s [−0.034, −0.009], wind-wave [−0.070, −0.035]).
- **What it means.** Swapping ERA5 + wind's wind sea for the camera's lowers the
  total-height error by 7 % (0.288 to 0.267 m) at buoys none of the models saw. On the
  12 test buoys Test 3's ridge left the total level; under the folds it gains
  −0.015 m, with its interval just reaching zero. Adding the camera as wind-sea energy
  gains more, and clearly.
- **Fitted values.** k (the share of the wind-sea correction that reaches the total) is
  0.78-0.88 across folds, and b (the camera's scale) 0.94-0.99. The gate acts mostly as
  a fixed blend of wind-sea energy, about four parts camera to one part ERA5 + wind (mean
  0.79; below 0.5 on 0.1 % of strips). It leans towards ERA5 in low sun (0.77) and on
  the fifth of strips where the camera is least sure (0.74). Forcing the camera in fully
  (gate open) halves the H_s gain, and its interval reaches zero.
- **For the wind sea alone the swap is not the best combiner.** Test 3's ridge (0.229 m)
  and gradient boosting (0.226 m) beat its 0.240 m, because its wind-sea output is a
  blend chosen for the total. Gradient boosting is the worst combiner for the total
  (0.297 m). Raw ERA5 with nothing fitted is worse than ERA5 + wind (+0.035 m): ERA5's
  heights need the buoy-wind correction first.
- **Confirmation** on strips archived after the package is still to come (protocol
  above, about 2026-11-26). Until then this counts as found, not confirmed.

## Stage 3: periods, wind-sea spectrum and direction (fixed 2026-11-18, before any stage-3 run)

Rank 5 of the architecture review: the model outputs heights only, but wave period and
direction are project targets too.

**What the caption carries.** Direction needs each view's compass heading, which the
review expected in the burned-in caption. It is not there. The caption holds one
number, "90°", under the first view, then the station and time. In 640 raw strips (8
at random from each of 80 stations, September to November), that field is
pixel-identical at 79 stations and reads "290°" at all strips of 44008 (the 2304-px
format): it is a fixed label, not a heading. **No strip says which way its views
face.** The one compass the images do contain is the sun, whose azimuth is known from
the time and the buoy's position. So direction is predicted relative to the sun:
cosine and sine of (wave direction minus sun azimuth), with the sun's azimuth added
back afterwards. Only the direction head knows the views' order (a learned embedding
per view position); the heights still pool the views as a set.

**Labels** (`scripts/buoycam_spectral_labels.py`, package `ndbc_buoycam_spectral_v3cv` =
v2cv plus these columns). Every value comes from the strip's own buoy: its `.spec`
summaries and its realtime spectral records (density with NDBC's separation frequency,
and the directional alpha1 and r1). The spectral files stamp each record 10 min after
its `.spec` stamp, so each joins the nearest `.spec` record (at most 15 min apart).
Then the heights' window applies: the mean over the records within 30 min of the strip.
The spectra reproduce NDBC's WVHT to within 2-2.5 % at every buoy (4 sqrt(m0) / WVHT
0.976-0.988), which confirms the join.

| Output | Definition | Strips | ERA5 counterpart |
|---|---|---|---|
| T_p | 1 / frequency of the highest density | 99.9 % | `pp1d` |
| Tm-1,0 | m-1 / m0, the energy period | 99.9 % | `mwp` (same definition) |
| APD | NDBC's average period, from m0 and m2 | 100 % | `mp2` |
| WWP, SwP | NDBC's wind-wave and swell peak periods | 92 %, 91 % | `mpww`, `mpts` |
| wind-sea Tm-1,0 | m-1 / m0 at or above NDBC's separation frequency | 94 % | `mpww` (same definition) |
| mean direction | energy-weighted over the spectrum (Kuik et al. 1988) | 97 % | `mwd` (same definition) |
| wind-sea shape | each of the 46 bins' share of the wind-sea energy | 94 % | none (no-camera guess: the mean training shape) |

ERA5's periods and directions were downloaded for September to November and read at
the grid point the heights use (`scripts/buoycam_era5.py`; through 2026-11-13). Periods
are corrected by a line fitted on each fold's training strips, as the heights;
directions are used as they are.

**Model** (`wave_analysis.models.wave_spectral.MultiViewSpectral`,
`scripts/train_wave_spectral.py`, config `ndbc_buoycam_spectral_v3cv.yaml`): stage 1's
model, with the periods as more log-scale targets of its point and distribution heads,
a shape head (softmax over the 46 bins, cross-entropy), and the sun-referenced direction
head (loss 1 - cos of the angle error). It uses stage 1's winning arm (backbone and
recipe) on the same folds, with three seeds per fold. The epoch is chosen on validation
H_s, as in stage 1. Scored by `scripts/buoycam_spectral_oof.py`.

**Bar S3**, one line per output, each against ERA5 at the same unseen buoys out of fold
(paired bootstrap over whole buoys; it passes when the 95 % interval of camera minus
ERA5 is below zero):

- **P1** peak period against `pp1d`, RMSE in seconds;
- **P2** energy period Tm-1,0 against `mwp`, RMSE in seconds;
- **P3** mean direction against `mwd`, mean absolute angle in degrees.

**H_s is checked for harm:** the stage-3 model's out-of-fold H_s RMSE minus stage 1's
winning arm's, on the same strips. Harm means that interval lies above zero.

Reported beside the bar: APD against `mp2`; the wind-wave and swell periods; the wind
sea's Tm-1,0 against `mpww`, both from its own output and from the predicted shape; the
direction with the sun above and below 10°; the shape's divergence from the buoy's
against the mean training shape; and the heights.

**Expected in advance:** ERA5 should win P1-P3, because swell sets the total period and
direction at these buoys and the camera has no compass but the sun. The camera's best
chance is the wind sea: its own period and the shape, which are reported beside the bar.

**Result (2026-11-19): P1-P3 not met, as expected; the camera beats ERA5 on the wind
sea's period, and the joint model harms H_s.** DINOv3-gentle, five folds x three seeds
(15 runs at `f00423f`, `runs/buoycam-cv-spectral-20261119`). Periods are scored by RMSE
in seconds; directions by the mean absolute angle between predicted and true direction,
in degrees (a random guess gives 90°).

| Out of fold, 79 buoys | Camera | ERA5 | Camera minus ERA5 |
|---|---|---|---|
| **P1** peak period T_p (`pp1d`) | 3.44 s | 1.70 s | +1.75 [+1.41, +2.07] |
| **P2** energy period Tm-1,0 (`mwp`) | 2.40 s | 0.67 s | +1.73 [+1.49, +1.97] |
| **P3** mean direction (`mwd`) | 75.8° | 13.8° | +62.0 [+58.5, +65.6] |
| APD (`mp2`) | 1.50 s | 0.58 s | +0.93 [+0.79, +1.06] |
| Swell peak period SwP (`mpts`, 77 buoys) | 2.92 s | 1.88 s | +1.04 [+0.85, +1.26] |
| **Wind-wave peak period WWP** (`mpww`, 77 buoys) | **0.69 s** | 0.93 s | **−0.24 [−0.27, −0.21]** |
| **Wind-sea Tm-1,0** (`mpww`, 77 buoys) | **0.55 s** | 0.75 s | **−0.21 [−0.24, −0.17]** |
| Wind-sea Tm-1,0 from the predicted shape | 0.54 s | 0.75 s | −0.21 [−0.24, −0.17] |

- **Bar S3: not met** on all three lines, as expected in advance. Swell sets the total
  period and direction at these buoys, and the camera reads swell poorly.
- **The wind sea's period: the camera beats ERA5** by about a quarter (WWP −0.24 s,
  wind-sea Tm-1,0 −0.21 s, both intervals below zero), the same pattern as the heights.
  It was reported beside the bar, not tested by it, so it counts as found, not
  confirmed.
- **Shape:** the predicted wind-sea spectrum is far closer to the buoy's than the mean
  training shape is. The Hellinger distance (0 for identical spectra, 1 for no overlap)
  is 0.30 against 0.48; the KL divergence is 0.33 against 0.78. The period computed
  from the shape matches the direct output (0.54 s).
- **Direction:** 76° from the true direction. A random guess gives 90°, and one fixed
  direction from the training strips gives 104-111° (compass or sun-relative; not part
  of the bar). So the sun gives the camera a little, nothing usable. It is worse with the
  sun at most 10° high (83°) than above (74°). ERA5 is within 14°.
- **H_s is harmed:** the joint model's H_s RMSE is 0.618 m, against stage 1's 0.532 m on
  the same strips (+0.086 m [+0.068, +0.110]); wind-wave +0.020 m [+0.013, +0.026].
  Nine scalar targets, the shape and the direction share one network with the heights.
  Heights stay with stage 1's model; periods would need their own head on it, or a model
  of their own.

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
- [x] v2 heights (2026-11-18, three seeds per arm): camera alone H_s RMSE 0.450 m,
      R² 0.81 at unseen buoys; V2-1 not met (the horizon does not help); V2-2 met
      for the wind sea (0.215 against ERA5's 0.341 m), not for the total.
- [x] Test 3 (2026-11-18): not met; the camera cuts ERA5 + wind's wind-wave error
      by 30 % and adds nothing to the total height.
- [x] Bar B1 (2026-11-18): not met, DINOv3 ConvNeXt-Tiny worse than ImageNet
      with arm A's recipe (H_s +0.039 m, wind-wave +0.025 m).
- [x] Bar B2 (2026-11-18): not met (DINOv3-gentle −0.010 m [−0.024, +0.001] against
      arm A); DINOv3-gentle is the best model so far (H_s 0.440 m, R² 0.82).
- [x] Bar B3 (2026-11-18): not met; gentle ViT-B/16 level with arm A (+0.005 m), frozen
      far worse (+0.111 m). Backbones differ by about 0.01 m once each has its recipe.
- [x] Protocol check R (2026-11-18): the random split lowers H_s error by 0.02 m and
      leaves Kim's five-class accuracy at 69 %: the gap to Kim is not the split.
- [x] Grouped cross-validation (2026-11-19, five folds x three seeds per arm): B2-CV not
      met. DINOv3-gentle lowers H_s RMSE at all 79 buoys (−0.019 m [−0.028, −0.011]) but
      not the wind sea (−0.002 m). Out of fold H_s RMSE 0.532 m, R² 0.79; buoy-level
      conformal 90 % intervals cover 0.90, but 11 buoys fall below 0.80.
- [x] Stage 2, partition-aware fusion (2026-11-19): bar S2 met out of fold. The gated
      swap lowers ERA5 + wind's H_s RMSE from 0.288 to 0.267 m (−0.021 m [−0.031, −0.012])
      and its wind-wave RMSE from 0.293 to 0.240 m.
- [ ] Stage 2 confirmation on strips archived after the package, from about 2026-11-26.
- [x] Stage 3, periods, spectrum and direction (2026-11-19): bar S3 not met (ERA5 wins
      peak period, energy period and direction). The camera beats ERA5 on the wind sea's
      period (0.55 against 0.75 s) and harms H_s when trained jointly (+0.086 m). The
      caption carries no heading (a fixed "90°").
- [ ] Relabel from NDBC's quality-controlled annual files when they appear.
