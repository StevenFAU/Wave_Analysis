# Next tests: the programme after the first results (fixed 2026-11-21, before any run)

**Recommendation: confirm the findings on untouched strips, then spend effort on time
context, on what the camera actually sees of swell, and on more seasons. Do not spend
it on bigger networks.** The deciding principle comes from the results so far: error is
set by what the data cover (one autumn, few swell regions) and by what one snapshot can
show. Network size changed error by about 1 cm.

This folder holds one file per test. Each file is the test's **bar**: the question, the
rows, the arms, the comparator, the pass mark and the controls, written before any run
of that test. A session that runs a test commits its bar file unchanged first, then
appends results to the same file.

The programme has three sources: the eight ranked tests and the nine new directions of
the internal report (sections 12 and 13), and five additions made here (B4's
replication contrast, B10, S5, and the data actions). Nothing in it has been run.

## The tests

Wave 0 is already fixed elsewhere and only needs running on its date. Wave 1 can start
now and runs in parallel. Wave 2 starts as wave 1 frees people and GPUs. Wave 3 waits
for a date or for data.

| ID | Test | What it settles | It must beat | Where, cost | Wave |
|---|---|---|---|---|---|
| A1 | Fresh-strip confirmation (claims C1 to C5) | Whether every headline claim holds on strips no decision has seen | Each claim's own comparator | Desktop CPU, about 7 h. Bar: `fresh_strip_confirmation.md` on branch `claude/fresh-strip-confirmation` | 0 · about 2026-11-27 |
| A2 | 14-day replication of the blends | Whether a state-dependent camera weight beats a fixed blend on later dates | Fixed and variance-weighted blends | Desktop CPU. Bar: [wind_wave_reliability.md](../wind_wave_reliability.md#independent-two-week-replication) | 0 · about 2026-12-17 |
| A3 | Storm-set bars T2-A, T2-B, T3-K | The one open comparison with a target paper (Kim) | Kim's networks on the same clips | Running in the storm session's worktree. Bar: `sensing_per_wave_bars.md` on branch `claude/sensing-shore-storm` | 0 · in progress |
| [B1](b1_hours_before.md) | Hours before | Whether a buoy's recent image history recovers swell and total height | The same head on one strip | Cached features, CPU | 1 |
| [B2](b2_swell_cue.md) | Find the swell cue | Which part of the view carries the camera's swell skill, and how much is a light-and-season proxy | Wind alone; wind + sun + brightness | 3 to 4 GPU-hours | 1 |
| [B4](b4_camera_as_observation.md) | The camera as an observation | Whether a state-dependent, variance-based update of the forecast beats a fixed blend | The fixed 50/50 blend | CPU, saved predictions. **Freeze by 2026-11-27** | 1 |
| [B7](b7_per_buoy_intervals.md) | Per-buoy prediction intervals | Whether ten labelled hours make the camera's intervals honest at every buoy | The global intervals | CPU, saved predictions | 1 |
| [B8](b8_general_model_baseline.md) | A general vision-language model | What training on instrument truth adds over an off-the-shelf model | Reported beside ours and ERA5 | 2 to 6 GPU-hours | 1 |
| [S1](s1_shore_stills.md) | One model for all shore stills | Whether fine-tuning helps on shore, whether a model transfers to an unseen beach, and whether buoy pretraining helps | Frozen probe; ERA5 at the beach | 10 to 20 GPU-hours | 1 |
| [B5](b5_image_to_spectrum.md) | Image to spectrum | Whether the camera adds spectral information beyond forecast and wind, bin by bin | Forecast features to the same bins | CPU first, then 12 to 14 GPU-hours | 2 |
| [B6](b6_neighbour_buoys.md) | Neighbour buoys | Whether a camera improves the forecast at a buoy 50 to 300 km away | The forecast alone | CPU, saved predictions. After B4 | 2 |
| [S2](s2_video_end_to_end.md) | Video trained end to end | Whether "motion adds nothing" is about waves or about frozen encoders | The same network on shuffled frames | 12 GPU-hours pilot, 150 cap | 2 |
| [S3](s3_label_free_period.md) | Label-free wave period | Whether a learned periodic signal beats the brightness spectrum | The brightness spectrum (2.59 s) | CPU and under 10 GPU-hours | 2 |
| [S4](s4_truth_in_view.md) | Truth in the camera's view | Whether shore errors come from the method or from labels 6 to 29 km away | Site climatology; a 2.0 s ceiling | Small open downloads, CPU | 2 |
| [X1](x1_camera_plus_imu.md) | Camera plus a real IMU | Whether real motion data adds to imagery, on a ship dataset already on disk | Imagery alone | CPU. A pilot: about 53 bursts | 2 |
| [X2](x2_render_swell.md) | Render the swell limit | Whether swell is visible in one snapshot from buoy height, in principle | Reported either way | CPU first, then 10 to 30 GPU-hours | 2 |
| [B3](b3_second_season.md) | A second season | Whether the results hold into winter and whether more data lowers error | The autumn models on the same strips | 20 to 25 GPU-hours | 3 · about 2027-01-07 |
| [B10](b10_third_wave_model.md) | A third wave model in the corrector | How much of the camera's gain survives a stronger forecast | The current corrector | CPU | 3 · needs an account |
| [S5](s5_nearshore_forecast.md) | Shore camera against a nearshore forecast | Whether a shore camera adds to the strongest shore comparator | NWPS at the buoy | CPU | 3 · needs 30 dates |

Report idea "one wave encoder for buoys and beaches" is arm C of S1. Report idea
"per-site calibrated uncertainty" is B7.

**Order within wave 1.** B4 first, because it has a deadline. Then B1, because it is
nearly free, attacks the camera's largest weakness and uses the one thing no other
group has: an hourly image record at many buoys. B2 next, because its answer decides
what to build for swell.

**Stopped on purpose.** Bigger backbones, more self-supervised pretraining at this data
size, horizon tilt as an input, ten-minute summary images, label-free site adjustment
and seismic height each have a controlled null result. Do not reopen them without a new
reason written into a bar.

## Rules shared by every bar

1. **Units held out.** Buoy strips: the five outer folds over the 43 station groups
   (`cv_fold` in package `ndbc_buoycam_heights_v2cvn`: 41,091 strips, 79 buoys,
   2026-09-23 to 2026-11-14). A buoy is predicted only by networks and heads that never
   saw it or any buoy within 300 km. Shore: whole local dates for video, whole years
   and whole sites for stills.
2. **The camera reference.** DINOv3 ConvNeXt-Tiny with the gentle recipe, three seeds
   per fold: `runs/buoycam-cv-stage1-20261119/oof_dinov3-gentle.csv`. Its cached
   features are in `runs/buoy-probe-*-t5-s{0,1,2}/fold{0..4}/features.npy`: 41,091 rows
   of 256 numbers per fold and seed, row order and role (train, val, test) in
   `feature_rows.csv`. For fold k use the fold-k files only.
3. **The no-camera comparator.** The corrected forecast (ERA5 + GFS-Wave + location and
   depth + the buoy's measured wind): `runs/forecast-baseline-oof-v1.parquet`, inner
   version `runs/forecast-baseline-inner-v1.parquet`. The nested fusion rows are
   `runs/forecast-fusion-nested-v1/rows.parquet` (34,723 strips, 71 buoys, 41 groups).
   Every baseline is fitted in the same folds on the same rows as the arm it is
   compared with.
4. **Statistics.** The primary score is pooled RMSE in metres unless the bar says
   otherwise. A contrast is the paired difference on identical rows. Its 95% interval
   comes from 2,000 resamples of whole station groups (buoys) or whole local dates
   (shore), seed 0. A bar is **met** when the interval's upper end is below zero and
   the gain reaches the size the bar names. Also report equal-group scores, per-fold
   scores and the counts of strips, buoys and groups.
5. **Seeds.** A trained network gets three seeds. A head on cached features gets one
   head per extractor seed, and the three predictions are averaged in log space.
6. **Found, then confirmed.** A pass on the package is a finding on strips that earlier
   decisions have seen. It becomes confirmed only on window **W3**: strips from
   2026-12-12 00:00 UTC to 2026-12-31 24:00 UTC at the same 79 buoys, each predicted by
   the fold that held it out, with nothing refitted. Before any W3 strip is scored,
   commit the SHA-256 of every fitted file. W3 is scored once, when ERA5 reaches
   2027-01-01 00:00 UTC (about 2027-01-07). The week from 2026-11-21 16:00 UTC to
   2026-11-27 24:00 UTC is spare: use it for plumbing checks, never for a reported
   number. The windows of A1 (2026-11-14 16:00 to 2026-11-21 16:00 UTC) and A2
   (2026-11-28 to 2026-12-11) belong to those tests.
7. **The sealed 12 buoys.** Nothing is tuned on them. They may be reported.
8. **Amendments.** A bar may be made more exact before its first score, in a commit
   whose message says so. After a score exists it does not change. A result outside a
   bar is reported as exploratory.
9. **Reporting.** One table, one sentence saying what is compared and what the number
   means, and the misses as plainly as the passes.

## Runs tied to a date (started by hand; nothing is scheduled)

| When | What | How |
|---|---|---|
| By 2026-11-27 | Commit B4's frozen parameter files and their checksums | B4's bar |
| About 2026-11-27, once ERA5 reaches 2026-11-21 16:00 UTC | A1 | The commands in `fresh_strip_confirmation.md`, in that branch's worktree |
| 2026-12-04 | Fallback date for A1's claims C2, C3, C4b and C5 | Same |
| About 2026-12-17, once ERA5 reaches 2026-12-12 00:00 UTC | A2, and B4's contrast on the same strips | `scripts/wind_wave_replication.py`; B4's bar |
| Before about 2026-12-29 | NDBC snapshot covering A1's window (the realtime files keep 45 days) | `systemctl --user start wave-analysis-ndbc-realtime.service` |
| About 2027-01-07, once ERA5 reaches 2027-01-01 00:00 UTC | W3 confirmations of every test that passed; then B3 | Each bar; B3's bar |
| When 30 local dates exist at two cameras | S5 | S5's bar |

## Data actions (no bar)

- **GFS-Wave.** The local archive ends 2026-11-14 and no collector runs. NOAA's public
  bucket keeps the bulletins, so nothing is lost. Backfill with
  `scripts/gfs_wave_archive.py <first> <last> --fetch` before A1, A2, W3 and B3.
- **NWPS.** The nearshore forecast is collected daily since 2026-11-15 by a transient
  timer that stops at a reboot. NOMADS keeps about five days. Check the timer after
  every reboot.
- **Shore stills.** The desktop collector needs the data drive. It looks back three
  days, so a gap longer than that is lost on the desktop.
- **Shore video.** More rough-weather dates at the same four cameras are covered by the
  existing approval. S2, S3 and S5 all gain from them.
