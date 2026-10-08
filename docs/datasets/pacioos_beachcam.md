# PacIOOS beach cameras (Oahu, 2009–2013)

Registry id `pacioos_beachcam` · module `wave_analysis.sources.pacioos` ·
**v0.2.0** · verified 2026-09-26.

## What it is

Four retired beach cameras operated for the Pacific Islands Ocean Observing
System (PacIOOS) by the University of Hawaii (PI: M. A. Merrifield), published
on the PacIOOS ERDDAP server. Each camera is a tabledap dataset with one row per
image; the images are served from ERDDAP's `files/` tree.

| Dataset | View | Images | Size | Period (UTC) |
|---|---|---:|---:|---|
| `beachcam_003` | Waimea Bay (beach) | 8,935 | 1.85 GB | 2009-02-06 – 2013-10-15 |
| `beachcam_004` | Waimea Bay (offshore) | 8,919 | 1.42 GB | 2009-02-06 – 2013-10-15 |
| `beachcam_001` | Waikiki (oblique) | 15,564 | — | 2009–2013 |
| `beachcam_002` | Waikiki (nadir) | 15,590 | — | 2009–2013 |

Waimea Bay counts and sizes are from the full indexes of 2026-09-26; Waikiki
counts from a query of the same date. Images are hourly JPEGs from about 06:00
to 17:00 local time (e.g. 1624 × 1254 px).

## Why Waimea Bay

The Waimea pair is the project's first **historical** camera + buoy benchmark:
it needs no prospective collection. CDIP 106 (a directional Waverider, also
NDBC 51201) is 6.3 km offshore and was recording through the whole camera
period ([CDIP](cdip.md)). North Shore winter swell spans small to very large
surf, so the range of wave heights is wide. 96.4 % of the image hours of both
cameras have a CDIP 106 sample starting within [t − 30 min, t + 1 h).

The buoy is in 200 m of water and the cameras watch the bay and surf zone.
Waves change as they shoal and break, so buoy H_s describes the incident
offshore sea state, not the wave heights visible in the image. Any result on
this benchmark must state that, and models should be compared with
non-vision baselines that use the same buoy (landscape review §7).

## Facts from the dataset metadata

From `erddap/info/beachcam_003/index.csv`, and checked on the full indexes:

- **Time zones.** `time` is UTC. File names and the `files/` directories use
  Hawaii Standard Time (UTC−10): `beachcam_003.20120105T170000.jpg` is
  2012-01-06 03:00 UTC. All 17,854 Waimea file names are exactly `time` − 10 h.
- **The index time is not the capture time.** "Minutes and seconds always
  rounded to 00:00 in filename. Actual time of photo may occur anytime within
  the specified hour. See timestamp on photo image for more precise time." An
  image labelled 17:00 was taken between 17:00 and 18:00. Pairing must use the
  whole hour, or read the timestamp burned into the image (done: see *Audit and
  capture times*, which also found a period when neither holds). The burned-in
  text must also be masked before training.
- **Sizes.** Each index row has the file size in bytes; the archiver checks
  every download against it.
- **Licence.** "The data may be used and redistributed for free but is not
  intended for legal use, since it may contain inaccuracies." Acknowledge
  PacIOOS. Citation form: "PacIOOS. [year-of-data-download], [Title], [Data
  access URL], accessed [date-of-access]."

## Archive

```bash
uv run wave-analysis pacioos beachcam --index-only     # fetch and check the indexes
uv run wave-analysis pacioos beachcam                  # Waimea pair (default); resumable
uv run wave-analysis pacioos beachcam beachcam_001 --start 2012-01-01 --end 2012-02-01
```

Layout under `data/raw/pacioos/beachcam/`:

| Path | Content |
|---|---|
| `<dataset>/<YYYY>/<MM>/<DD>/<file>.jpg` | Images, mirroring ERDDAP's `files/` tree (HST dates) |
| `_index/<dataset>_<stamp>.csv` | The index as served at each run |
| `_manifests/<dataset>.csv` | Per-request ledger: URL, SHA-256, size, status; a size that differs from the index is noted |

The ledger is kept with the archive, like the buoy-camera ledger (ADR 0008),
and is copied offsite with it. A run skips images already archived with the
ledger's size and URLs recorded as 404, so an interrupted run resumes where it
stopped. Requests are sequential at most one per second; the Waimea pair takes
about 11 hours.

The first full download was started on 2026-09-26 as a one-off user service:

```bash
journalctl --user -u wave-analysis-pacioos-waimea -f       # progress every 500 images
systemd-run --user --unit=wave-analysis-pacioos-waimea \
  --working-directory="$PWD" .venv/bin/wave-analysis pacioos beachcam   # resume after a reboot
```

## Audit and capture times (2026-09-27)

`scripts/waimea_extract.py` then `scripts/waimea_timing_audit.py` (outputs in
`data/interim/waimea_audit/`; `image_times.parquet` has one row per image with
its trusted UTC capture time or none).

**Archive.** 17,689 of the 17,854 indexed images are archived. Each matches its
ledger SHA-256 and the index size, ends with a JPEG EOI marker, and decodes at
1624 × 1254 RGB; no two are identical. The other 165 return HTTP 404 from
PacIOOS: `beachcam_003` 2010-01-12 to 01-22 (112) and `beachcam_004`
2009-12-29 to 2010-01-03 (53).

**Coverage.** The period 2009-02 to 2013-10 has long gaps in both cameras:
2009-04-14 to 09-23 (162 days), 2010-08-12 to 11-03 (83), 2011-02-18 to 10-25
(249), 2013-02-09 to 05-23 (103), and several of 1-6 weeks. The views are
stable: monthly median views move by at most about 7 px (of 1624) with no
re-aiming.

**Overlays.** Pixel variance over 250 daytime images per camera, sampled from
all years:
- Both cameras have the caption band at rows 1232-1253.
- `beachcam_004` also has a fixed black polygon over the shore: rows 985-1231,
  full width below row 1110. It has the same shape in every quarter from 2009
  to 2013, so it carries no date cue.
- `beachcam_003` has no mask. Its lower third shows the highway with passing
  vehicles and a fixture in the foreground; the water is in the upper part.

**Caption.** Every image carries `YYYY.MM.DD HH:MM:SS   F: YYYYMMDDTHHMMSS`
(bottom-left, 10 px monospaced text, followed by "C.Kontoes, T.Hilmer and
D.Young, UH"). All 17,689 captions were read by template matching; the two
copies of the time agree within 1 s on every image. There is no EXIF.

**Which time to use.** Normally the caption falls inside the hour named in the
file name: photos at hh:55 (60 %) or, in 2010-2011, hh:23-24. So the file name
is usually 25-55 min early and the caption is the capture time. Not trusted
(638 images, `time_source = uncertain`):

- 23 May - 27 July 2013 (636 images): captions often 1-10 h after the named
  hour, and some daylight images are stamped 19:33-23:37 HST, so the camera
  clock was wrong. The file name is plausible but not verified; excluded.
- 27 Feb 2010, 08:00 files (2 images): stamped about 18:00 (the day of the
  Chile tsunami warning).

**Checks of the caption time** (17,051 trusted images):

- Time zone: read as HST, no image is taken with the sun more than 6 deg below
  the horizon; read as UTC, 13,752 would be. The captions are HST.
- Clock offset: images without any bright region occur only with the sun below
  -3 deg (5 of 1,147 twilight images). Clock shifts from -26 to +4 min explain
  the dark and bright twilight images equally well; larger shifts do not. This
  bound is coarse: auto-exposure keeps images bright to about -3 deg, so few
  images are dark.
- Date: the white-water fraction in the water blocks (chosen from the view)
  correlates with CDIP 106 Hs (Spearman 0.56 for `beachcam_003`, 0.19 for
  `beachcam_004`). Over all years the peak is at +6 h (003) and 0 h (004), and
  the correlation falls within a day; year by year it peaks within half a day
  of zero in 2009, 2010, 2012 and 2013. 2011 (800 images) is inconclusive with
  this crude index (peaks of 0.18 and 0.08). The index resolves dates, not
  hours.
- Hour scale: hour-to-hour changes of the index do not correlate with buoy
  changes (rho about 0.01), so it cannot test timing within hours.
- Visual: the images at the highest buoy Hs (5-6 m) show the bay full of
  breaking waves; those at the lowest (0.7-0.8 m) a glassy bay; also in 2011.

**The two cameras are not simultaneous.** In 1,367 of the 8,257 hours with a
trusted image from both, the captions differ by more than 2 min (up to 22 min;
e.g. `beachcam_004` at hh:17 and `beachcam_003` at hh:23-25). The pattern
changes from hour to hour within a day, which clock drift would not do, so the
captions record real capture times. Each image is paired on its own caption.

## Pairing with CDIP 106 (2026-09-27)

`scripts/waimea_pairs.py` (after the two audit scripts) writes
`data/processed/pairs/waimea_cdip106/`; its summary and provenance are also in
`data/manifests/processed/`. **No label rule is applied.** For each image with a
trusted time, every CDIP 106 record centred within ±3 h is kept
(`candidates.parquet`: offset from the record centre, whether the capture falls
inside the 1600-s sample, rank by distance in time, the record's values, flags
and deployment). `samples.parquet` has one row per indexed image, with the
reason it cannot be paired, sun position, the offset to the other camera's
image of the same hour, and label diagnostics.

| Images | Count |
|---|---:|
| Indexed | 17,854 |
| Paired: a CDIP record centred within 30 min | 16,404 (003: 8,174; 004: 8,230) on 800 days |
| `no_buoy_within_30min` | 647: the buoy changeovers 2010-01-30 to 02-23 and 2013-01-29 to 02-10, and single-day gaps |
| `time_uncertain` | 638 |
| `image_missing` (HTTP 404 at PacIOOS) | 165 |

**CDIP 106 in 2009-2013.**
- 84,893 records: 1600-s samples every 30 min, no duplicates, 68 gaps over 1 h.
- CDIP's Hs matches the Hm0 recomputed from the spectrum within 1.2 cm on every
  paired record. One unpaired record (2009-05-12) differs by 5 %.
- 683 records carry secondary flag 10, `hf_transmission_errors_fixed`, with a
  good primary flag; 55 of them are paired.
- Deployments d09-d12 report the same position, and d13 moved 280 m. The
  distance to the cameras is 6.31 km throughout.

**Timing.**
- The nearest record centre is a median 10.6 min and at most 29 min from the
  capture. 73 % of captures fall inside that record's sample.
- Travel time from the buoy to the bay: at the peak-period deep-water group
  velocity it is a median 12 min (90 %: 19 min), and shoaling makes it longer.
- The travel time is of the same order as the clock bound (-26 to +4 min) and
  the 27-min sample, so no lag correction is applied yet.

**Label noise.**
- Consecutive CDIP records, 30 min apart, differ by a median 4 % of Hs (90 %:
  10 %). That puts the sampling noise of one record at σ ≤ 4 %.
- Around each image, the records centred within ±1 h spread by a median 0.13 m
  (9 %; 90 %: 0.33 m, 16 %). That is what this noise alone predicts for about
  four records, so the spread is mostly sampling noise, not timing.
- Nearest record or interpolation in time: the label changes by a median 1 cm
  (90 %: 6 cm; 99 %: 19 cm).
- Example: the largest-wave image (2009-03-14 02:28 UTC) has records of 5.46,
  6.44, 6.05 and 5.41 m within the hour. The nearest record gives 6.44 m and
  the interpolation 6.18 m. Taking one record inflates extremes, so an averaging
  rule is preferable.

**Label range.**
- Hs runs from 0.73 to 4.5 m (1st to 99th percentile), with a maximum of 6.44 m.
- 1,065 images are at 3 m or more.
- 299 images are at 4 m or more, on only 27 days. Results at high sea states
  rest on few independent events, so splits and confidence intervals must be by
  day or event.
- Illumination: 15,540 day, 653 low sun (0-6°), 211 civil twilight; 17 are dark.

## Label rule (decided 2026-10-01, ADR 0010)

The options, for an image taken at 10:55 with records starting at 10:23 and
10:53 (centres 10:36 and 11:06):

| Rule | Uses | For | Against |
|---|---|---|---|
| `nearest` | the 10:53 record | simple; one real measurement | about 4 % noise; inflates extremes (6.44 m vs about 6.0 m at the largest swell) |
| `interpolate` | 10:36 and 11:06, weighted by time | follows trends | same noise; differs from nearest by a median 1 cm |
| **`window_mean`** | every record centred within ±30 min (normally both) | **halves the noise variance; covers the clock bound (−26 to +4 min)** | smooths a swell arriving within the hour |
| lagged | any of the above after moving the image ~12 min earlier | physically right direction | the lag is smaller than the clock bound and the record, so no gain yet |
| `in_record` | the record containing the capture (73 % of images) | cleanest subset | fewer samples |

The primary label is the **window mean**, with the spread of its records as
the label's uncertainty. `nearest` and `interpolate` labels are kept beside it,
and every result is reported under both window mean and nearest. All rules
are implemented in `processing.synchronize.label_from_candidates` and apply
directly to `candidates.parquet`.

Peak period and direction are the exception: they come from the **nearest**
record. A record's peak is the band with the most energy, and it jumps when
two swells carry similar energy, so a mean of two peaks may match neither. On
the 16,404 paired images the two window records' T_p differ by 3 s or more for
808 (4.9 %) and their D_p by 60° or more for 401 (2.4 %). The 929 images with
either (5.7 %) carry `peak_records_disagree`. CDIP T_p is the centre of a
frequency band (37 distinct values in 2009–2013, adjacent ones 0.6–1.3 s apart
between 9 and 17 s), so 3 s is at least two bands there.

Result on the real archive (2026-10-01): 16,370 labels use two records, 31 use
one and 3 use three. The window mean differs from the nearest record by a
median 3 cm (90 %: 10 cm; 99 %: 25 cm), and the two records spread by a median
3.9 % of H_s (90 %: 10 %), matching the record noise above. Every count outside
the summary's `label` block is unchanged from 2026-09-27.

## Benchmark v0: splits and baselines (2026-10-08)

`scripts/waimea_benchmark.py` with `configs/experiments/waimea_hs_v0.yaml`.
Outputs in `data/processed/benchmarks/waimea_cdip106_v0/`; the per-day split
table is `data/manifests/ml/waimea_cdip106_v0.days.csv`, and the results,
strata, comparisons, summary and provenance are copied to
`data/manifests/processed/waimea_cdip106_v0.*`.

**Population.** The 16,404 paired images (8,174 + 8,230) on 800 HST days. The
other 1,450 indexed images are in the exclusion ledger.

**Splits.** Each one keeps both cameras of a day together, because they share a label.

| Protocol | Unit | Train / val / test images | Test days |
|---|---|---|---|
| `day` | HST day, 70/15/15 by images, seed 0 | 11,469 / 2,468 / 2,467 | 120 |
| `week` | ISO week (HST) | 11,423 / 2,445 / 2,536 | 121 |
| `year` | train 2009-2011, val 2012, test 2013; 72 h embargo (20 images) | 9,286 / 5,168 / 1,930 | 91 |

**Leakage audit.** All three pass: no day (or week) is in two partitions, and
no train and test images are closer than 6 h across both cameras. The
smallest train-test gap is 12.9 h, the night between two days. Two things
the audit cannot settle:
- **Neighbouring days.** Daily mean H_s has a lag-1-day correlation of 0.67,
  because swells last several days. In the `day` split, 114 of 120 test days
  sit next to a training day; in the `week` split, 22 of 121. The image
  baseline scores the same under both (R² 0.66 and 0.69), so day adjacency
  does not visibly inflate it.
- **Image hashes are not a leakage test here.** Two images of one camera on
  different days are a median 16 bits apart (64-bit dHash), and 0.2-0.4 % of
  such pairs are within 4 bits. With about 5,700 training images per camera,
  almost every test image has such a match by chance (1,643 of 2,467). For a
  fixed camera, a near-identical hash means a similar calm sea, not the same
  event.

**Baselines.** Each is fitted on train + val (val chooses the ridge penalty,
one model per camera) and scored on test, with 95 % intervals from a
bootstrap over HST days. `persistence_kh` is the last CDIP 106 record centred
at least k h before the image: what a buoy that stopped k hours ago would
give. ERA5 is the nearest sea point, 40 km offshore. `ridge_blocks` uses the
16 × 16 white-fraction and mean-luma blocks of `waimea_extract.py` (caption
excluded). `ridge_whitewater` uses only the white fraction of the water
blocks.

Test RMSE (m) and R², window-mean label (ADR 0010):

| Baseline | `day` RMSE [95 % CI] | `day` R² | `week` RMSE | `week` R² | `year` RMSE | `year` R² |
|---|---|---|---|---|---|---|
| Training mean | 0.69 [0.54, 0.86] | 0.00 | 0.78 | −0.01 | 0.76 | −0.46 |
| Climatology by month | 0.58 [0.45, 0.74] | 0.28 | 0.68 | 0.23 | 0.39 | 0.61 |
| Persistence, 24 h | 0.58 [0.46, 0.70] | 0.30 | 0.63 | 0.33 | 0.39 | 0.61 |
| Persistence, 3 h | 0.18 [0.15, 0.20] | 0.94 | 0.20 | 0.93 | 0.13 | 0.96 |
| Persistence, 1 h | 0.13 [0.11, 0.14] | 0.97 | 0.14 | 0.97 | 0.09 | 0.98 |
| ERA5, raw | 0.55 [0.50, 0.60] | 0.36 | 0.54 | 0.52 | 0.46 | 0.47 |
| ERA5, linear correction | 0.30 [0.25, 0.35] | 0.81 | 0.33 | 0.82 | 0.23 | 0.86 |
| Ridge, water-block white fraction | 0.67 [0.55, 0.82] | 0.05 | 0.76 | 0.04 | 0.72 | −0.33 |
| **Ridge, image blocks** | **0.40 [0.35, 0.46]** | **0.66** | 0.43 | 0.69 | 0.36 | 0.67 |
| ERA5 corrected + image ridge on its residual | 0.26 [0.23, 0.30] | 0.86 | 0.30 | 0.85 | 0.25 | 0.85 |
| *Nearest label vs window-mean label* | *0.06* | *0.99* | *0.07* | *0.99* | *0.05* | *0.99* |

Paired differences in RMSE (m, candidate minus reference, bootstrap over the
same days; negative means the candidate is better):

| Candidate vs reference | `day` | `week` | `year` |
|---|---|---|---|
| ERA5 + image vs ERA5 | **−0.041 [−0.064, −0.017]** | −0.028 [−0.045, −0.011] | +0.012 [−0.004, +0.027] |
| Image blocks vs ERA5 | +0.099 [+0.042, +0.165] | +0.099 [+0.041, +0.152] | +0.127 [+0.086, +0.166] |
| Image blocks vs climatology | −0.182 [−0.285, −0.086] | −0.248 [−0.340, −0.148] | −0.029 [−0.076, +0.015] |

What this says:
- **The corrected ERA5 nowcast is the baseline to beat.** Raw ERA5
  over-predicts by 0.40-0.46 m (bias). The correction fitted on training data
  (about H_s = 1.03 × ERA5 − 0.52 m) brings RMSE to 0.23-0.33 m.
- **Image statistics alone do not beat it.** The block ridge is 0.10-0.13 m
  worse in every split. It beats climatology on held-out days and weeks, but
  not on a held-out year.
- **Within the same years, the image adds to the nowcast.** Modelling ERA5's
  residual from the image lowers RMSE by 0.03-0.04 m on held-out days and
  weeks. This is a first, partial answer to Q-M6.
- **On 2013, the image adds nothing.** The ERA5 + image model is no better
  than ERA5 alone, and the image-only model no better than climatology.
  Image statistics learned in 2009-2011 do not carry over to 2013. Not yet
  diagnosed; candidates are scene changes (sand, vegetation, exposure) and
  2013's partial seasonal coverage. Any learned model must be reported on
  this split as well as on held-out days.
- **The label rule barely matters at this accuracy (Q-M7).** Under the
  nearest-record label, RMSE changes by less than 0.01 m for every baseline
  except persistence (+0.015 to +0.020 m). The two labels differ by 0.05-0.07
  m RMSE, far below the errors of any model that does not use the buoy.
- **Errors grow with wave height** (`*.strata.csv`). The day-split test has
  only 2 days (34 images) at 4 m or more, so results above 4 m rest on very
  few events. On those 34 images, ERA5 corrected under-predicts by 0.85 m and
  the image ridge by 1.15 m.
- **Persistence is not a camera competitor.** It shows how predictable the
  label is when the buoy itself is available: a buoy 1 h out of date gives
  0.09-0.14 m.

## Next steps

- First learned image model on these splits, scored against the table above
  on both `day` and `year`. Mask the caption band (rows 1232-1253) before
  training.
- Diagnose the 2013 drift: per-year image statistics, and retrain with 2012
  in training (`year` uses it for validation only).
- Day splits are fine for the baselines, but report `week` beside `day` for
  learned models, which can exploit neighbouring days more than a ridge can.
- A cross-camera test (train on one view, test on the other) must also hold
  out the days.
- Image-quality features for strata or exclusion (fog, rain drops on the lens,
  glare), recorded in the manifest builder's ledger.
