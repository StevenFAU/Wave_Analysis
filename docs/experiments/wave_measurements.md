# Wave measurements beyond height: slope, breakers and the rest

**Question.** The buoy-camera study found the camera to be a wind-sea instrument: it
reads the wind-wave height better than ERA5 and adds little to the total
([sea-state study](ndbc_buoycam_sea_state.md), stage 2 and stage 3). Surface slope is a
wind-sea quantity, so the next measurement is the slope the buoy itself records. Codex
prepared this and other measurement experiments on 2026-11-19 (branch
`codex/wave-measurement-experiments`, code `0674b9c`); this doc fixes their bars and
records the review of that code (branch `claude/wave-measurements-review`).

Launch and rebuild guide: [measurement_checkin.md](measurement_checkin.md). Rationale and
published comparisons: [Wave measurements to learn next](../../reports/Wave%20measurements%20to%20learn%20next.md).

## Bar M-S1: surface slope from the buoy cameras (fixed 2026-11-19, before any run)

**Target: the buoy's RMS surface slope, dimensionless.** For the strip's own buoy,

    s = sqrt( sum_i E(f_i) df_i k_i^2 ),   k_i = (2 pi f_i)^2 / g   (deep water)

over NDBC's 46 realtime spectral bands from 0.030 to 0.495 Hz (centres 0.0325 to
0.485 Hz; nominal WPM widths 0.005, 0.01 and 0.02 Hz). E is the measured spectral
density (m²/Hz) of the records within 30 min of the strip, averaged as the heights are
(ADR 0010); k is in rad/m, so k² E df is a slope variance and s is a slope (rise over
run): s = 0.056 is a typical slope of about 3.2°. s² is the mean-square slope of the
waves the buoy resolves ([Longuet-Higgins 1957](https://doi.org/10.1098/rsta.1957.0002),
the second moment of the wavenumber spectrum; band table: [NDBC](https://www.ndbc.noaa.gov/faq/wavespectra.shtml),
[Earle 2003](https://www.ndbc.noaa.gov/wavemeas.pdf)). The upper edge, 0.495 Hz, is a
wavelength of about 6.4 m, so this is **not** the optical slope of sun-glitter studies
([Cox & Munk 1954](https://doi.org/10.1364/JOSA.44.000838)), which is dominated by
shorter waves and ripples. Built by `scripts/measurement_labels.py` (column `slope_rms`)
for all 41,091 strips: mean 0.056, standard deviation 0.021. With the nominal widths,
4 sqrt(m0) reproduces NDBC's WVHT to 0.976-0.988 at all 79 buoys (the
stage-3 check, now with the right widths). Beside it, not scored by the bar: the same
slope with finite-depth dispersion (`slope_finite_depth`; the 79 depths from NDBC's
station pages; 0.05 % from deep water on average, up to 8 % at the shallowest buoys)
and without the bands above 0.35 Hz (`slope_below_035`), for the noisy high-frequency
tail. **50 strips have a spectrum with no energy at all** (all at buoy 46081, WVHT 0.0):
dropouts, not calm seas; they get no slope and are left out of its fit and score.

**Folds and checkpoints.** The grouped cross-validation of bar B2-CV: five folds over
the 43 station groups (`split_fold0` .. `split_fold4`; the measurement package's
columns are identical to `ndbc_buoycam_heights_v2cv`'s on all 41,091 strips, checked
2026-11-19). Fold k is scored only on its test buoys with the seed-0 DINOv3-gentle model
of fold k, which trained on that fold's training buoys and chose its epoch on its
validation buoys. The probe checks each checkpoint's fold and that its own test strips
are exactly the fold's test strips, and records its SHA-256:

| Fold | Checkpoint run | SHA-256 (first 12) |
|---|---|---|
| 0 | `buoy-fold0-20261118t2234-seed0-cv-dinov3-gentle` | `ea0a3ff2ed74` |
| 1 | `buoy-fold1-20261118t2342-seed0-cv-dinov3-gentle` | `a7edaec7bbe8` |
| 2 | `buoy-fold2-20261118t2346-seed0-cv-dinov3-gentle` | `a272a4e2ea52` |
| 3 | `buoy-fold3-20261119t0038-seed0-cv-dinov3-gentle` | `41d2e02fa7dc` |
| 4 | `buoy-fold4-20261119t0130-seed0-cv-dinov3-gentle` | `d4a593b9d7c8` |

**Probe** (`scripts/measurement_buoy_probe.py`, one run per fold; nothing trained on the
GPU). The camera's features are the frozen model's pooled view-mixer output (256
numbers per strip, the masked mean over usable views). Every arm is a ridge regression
(penalty 10 on standardised inputs); the standardisation and the fit use the fold's
**training strips only**, and the test buoys are only predicted.

- **Comparator, wind + geometry:** the buoy's wind speed and gust, their squares and
  product; and the horizon over at least three clear views: the strip's raw roll RMS and
  the spread of its horizon row, plus the roll and pitch RMS after removing each camera's
  mounting (its median angle and row over its own strips, from images only, as the
  [motion study](ndbc_buoycam_sea_state.md#the-camera-as-a-motion-sensor-the-horizon-follows-the-buoys-slope-2026-11-18)).
- **Candidate, all inputs:** the same plus the 256 camera features.

Both arms score the same strips: the test strips with a slope, wind and at least three
clear horizons (the "eligible" strips; their count and buoys are reported).

**Bar M-S1.** All inputs must lower the out-of-fold RMSE of the slope against wind +
geometry, with the 95 % interval of the difference (all inputs minus wind + geometry)
below zero: paired bootstrap over the 43 station groups (2,000 draws, seed 0), over the
eligible strips of all five folds pooled. Groups, not buoys, are the unit, because
buoys of one group share weather; this is the more cautious of the two intervals B2-CV
reported.

Reported beside the bar, not part of it: the difference with every group weighted
equally; coverage; the difference per fold; the camera alone, wind alone and geometry
alone; the same arms for the wind-wave height (a check that the camera still sees the
wind sea); the other probe targets; and the **validation-fit sensitivity**: the same two
arms fitted on the fold's validation strips instead of its training strips.

**Why the sensitivity.** On the training strips the camera's features are in-sample:
the encoder was fitted to their heights, so they read those strips better than they will
read a new buoy, and a ridge fitted there leans on them too much. That can only make the
camera look worse at the test buoys. The validation strips were seen only to choose the
epoch, so their features behave like a new buoy's; their cost is fewer buoys (11-14 per
fold) to fit on.

**What justifies the slope fine-tune** (`configs/experiments/buoy_slope_v1.yaml`: ten
epochs, slope and wind-wave heads, started from the same fold's encoder and mixer): bar
M-S1 met, **or** the validation-fit sensitivity meeting the same condition (its interval
below zero), because the fine-tune trains end to end and does not share the in-sample
problem. If neither interval lies below zero, the frozen camera adds nothing to wind and
geometry at unseen buoys and the fine-tune is not run. If it runs, its own bar is fixed
here before it does.

**Expected in advance.** Wind alone is a strong comparator: across all strips the slope's
rank correlation is 0.87 with the wind speed and 0.90 with the wind-wave height, and
Codex's CPU pilot put a wind-only ridge at about 0.0116 against the slope's spread of
0.021. The camera reads the wind sea better than ERA5, so a small gain over wind is
plausible; a large one is not.

**Not part of this bar, run if time allows:** the same probe with only the water below
each horizon (`--water-only`), the control that the camera does not read the slope from
the horizon line alone; and the confirmation of a pass on strips archived after the
package (from 2026-11-14 16:00 UTC), as stage 2's.

## Review of the prepared experiments (2026-11-19)

Codex's diff against `8f225d5c8` (26 files, about 3,400 lines) was read before any run.
Fixed means fixed on this branch before the run it affects; open means not fixed and
not needed for the runs here; rejected means not run.

**Leakage in the slope probe.**

- Folds: the measurement package's `cv_fold`, `split_fold0` .. `split_fold4`, station
  and group columns equal `ndbc_buoycam_heights_v2cv`'s on all 41,091 strips; every
  group sits in one partition per fold. Checked.
- Checkpoints: the five seed-0 DINOv3-gentle runs are full images-only v2cv runs of
  their own fold (lp 2, backbone scale 0.1), and each one's own test strips are exactly
  its fold's test strips; their SHA-256 is the same on the desktop and the server.
  The probe now enforces the test-strip match. Checked.
- Fitting: standardisation and every ridge come from the fold's training strips only.
  **Fixed:** the fit was written as "not test"; once the validation strips are
  extracted that would have folded them in, so the training strips are now named
  explicitly (also in the band dictionary and the ERA5 correction). A unit test with
  test labels 250 times the training relation shows the test labels pull nothing.
- The de-mounted geometry uses each camera's median over its own strips, the test
  buoy's included: a calibration from images, with no label. Stated in the bar.
- The camera's features are in-sample on the training strips, which can only hurt the
  camera: the validation-fit sensitivity is added for that.

**Labels.** The slope is computed correctly: deep-water k = (2 pi f)² / g, NDBC's
nominal band widths, s = sqrt(sum k² E df), dimensionless; definition, units and
citations are in the bar. **Fixed:** 50 strips with an all-zero spectrum (buoy 46081,
WVHT 0.0) get no slope. The finite-depth variant uses all 79 depths from NDBC's
station pages.

**Comparator.** Wind speed and gust with their squares and product, fitted inside the
same folds, is fair. **Fixed:** the geometry was the raw horizon angle, which carries
each camera's mounting tilt; the de-mounted roll and pitch RMS of the motion study are
added, which can only strengthen the comparator.

**Code quality.** The tests pass: 342 (Codex reported 329) and 344 with the two new
probe tests. `uv run pytest` in a fresh worktree fails at collection, because the
default environment lacks the `imaging` and `ml` extras; use the main checkout's
environment or `uv sync --extra imaging --extra ml --extra netcdf`.

- **Fixed:** the probe decoded every image in the main process (no loader workers).
- **Fixed:** the aggregator reported the slope delta only; it now reports the bar,
  coverage, per-fold deltas, the sensitivity and the wind-wave check.
- **Fixed:** the probe could not be launched through `deploy/dgx/run.sh`, whose
  container entry point adds `--config` and `--split`; `deploy/dgx/measure.sh` and
  `deploy/container/measure.sh` launch it the same way (committed code, package tar,
  labelled container).
- Open, for the fine-tune only: `metrics.json` does not record `--init-heights-run`
  (only the provenance inputs do), and the zero spectra would enter it at the 0.05 %
  floor of the log target. Both are to be fixed before a fine-tune runs.
- **Rejected, breakers (`train_measurement_breakers.py`), not run.** (1) Both encoders
  are frozen, yet every clip is pulled from the 5 GB zip and decoded by ffmpeg again for
  every item of every epoch: about 700,000 decodes for the twelve runs, where 9,000
  would do. (2) Its 16 frames are spread over the whole 30-s clip, one every 1.9 s, so
  the R3D-18 "video" arm cannot follow a breaking event that lasts a second or two; the
  comparison with frames + GRU does not test motion, and the two arms also differ by
  backbone. (3) There is no bar, and the published comparator (Yin et al.'s own
  leave-one-site-out results) is not in hand: the paper page refuses automated reading.
  These are design changes, not fixes; they are left to a session of their own.
- Open, minor: `measurement_period.py` maps `jennette_south` to the Jennette site but
  not `jennette_north`, which would become a site of its own in a site holdout (no such
  clip in the current eight); its error says "dates" for every split.
  `measurement_heave.py` uses the deprecated `np.fromstring`.

**Shared code and merge conflicts** (`scripts/train_wave_heights.py`,
`src/wave_analysis/models/wave_heights.py`; not resolved here):

- `claude/strips-grouped-cv-085fd2`: none in these files. Its commits after `8f225d5c8`
  touch neither. `CHANGELOG.md` will conflict textually (both add entries at the top).
  This bar lives in its own doc so as not to touch `ndbc_buoycam_sea_state.md`, which
  that branch edits.
- `claude/dinov3-continued-pretraining-fc2b1a` (branched at `1dfcc5f29`, so it must
  first take the strips branch's changes to these files): three hunks in
  `train_wave_heights.py`. (a) Both add an option right after `--backbone-lr-scale`
  (`--init-heights-run`, `--backbone-weights`): keep both. (b) The model's
  `pretrained=...` line and the block after it: keep both blocks and refuse the two
  options together (each replaces the encoder's weights). (c) The provenance `inputs`
  expression (that branch edits its closing line, Codex appends after it): keep both;
  the same for `run_info` once `--init-heights-run` is recorded there.
  `wave_heights.py` (Codex's `units`) does not conflict with that branch.

**Hostnames.** **Fixed:** the handoff's `GPU_HOST=<lab-host>` is now `<ssh-alias>`. The whole
diff has no other lab hostname, bastion name or IP address.

**Data licences** (read 2026-11-19):

- Yin et al., VWBT_9000 (figshare, [doi:10.6084/m9.figshare.28814993.v5](https://doi.org/10.6084/m9.figshare.28814993.v5)):
  **CC BY 4.0** in the record's metadata; use with attribution (the dataset and the
  [data paper](https://doi.org/10.1038/s41597-025-06005-5)). Its clip names are WebCOOS
  cameras (`jennette_north`, `oakisland_east`, ...); WebCOOS's usage guidelines
  ([webcoos.md](../datasets/webcoos.md#usage-guidelines)) apply to that footage too.
- Carini, Duck 2016 thermal IR and lidar ([hdl:1773/46586](http://hdl.handle.net/1773/46586)):
  **CC BY** (`dc.rights` of the UW repository record); use with attribution.

Neither licence stops training; the breakers are not run for the reasons above.

**Not runnable yet** (readiness only):

- Crest timing: 8 clocked clips at three sites on five local dates, reader sheets A and
  B ready; needs two independent readers and a tolerance fixed from their agreement.
- Foam: 25 buoy views ready for masks (codes 0, 1, 2, 255); needs a reader.
- ERA5 band correction: one station-day request (41004, 2026-10-01) and a tested
  decoder; needs the CDS retrieval and a coverage check before any fit.
- Stereo: all 53 usable bursts peak at the crop's longest wavelength, so no wavelength
  claim; elevation RMS and period only.
- Shore video: owned by the video session (`claude/video-whole-clip-491c90`).
