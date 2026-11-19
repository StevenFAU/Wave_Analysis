# Wave measurement experiments · November 2026

Start with the buoy slope comparison. It has a physical target for all 41,091 strips and can reuse the five geographic folds from the height study. The first question is whether the camera reduces error beyond measured wind, gust and buoy tilt. At the shore, the immediate opportunity is to measure local crest timing directly while the video collection builds independent dates. The supporting evidence and published comparisons are in the [research review](../../reports/Wave%20measurements%20to%20learn%20next.md).

## First check after the current runs finish

From the measurement worktree, run the following read-only commands. The second command prints the five seed-0 launch commands for a selected arm. Use `a` for the ImageNet ConvNeXt and `dinov3` for DINOv3 ConvNeXt. An absent or ambiguous fold prints a comment. Set `MEASURE_RUNS` to a different downloaded run root if needed.

```bash
cd ~/Projects/Wave_Analysis-measurements
scripts/measurements status
scripts/measurements plan --arm a --seed 0 --stage probe
```

1. Check the upstream grouped-CV summary, the two model families × five folds × three seeds, partition-fusion summary, spectral summary, and the video and continued-pretraining logs in their original jobs. The status command inventories **local files**. A missing local artifact calls for checking or pulling the producing job. The grouped-CV rule selects the lower out-of-fold significant-height RMSE for transfer and judges evidence of backbone superiority separately on height and wind-wave RMSE. Record the two deltas, intervals, seed spread, and selected arm before training slope. [Registered selection rule](ndbc_buoycam_sea_state.md)
2. Review the paired S3 peak definitions if its saved predictions are available. Check row and group counts and the subset with a peak range above three seconds. Review circular direction against the solar-position, location and UTC-time control. The direction control is fitted inside each training fold. Seed agreement measures prediction consistency.
3. For shore, count independent dates in the **intersection** of clips, whole-clip index, pixel spectra, and matched features used by each experiment. The 536-clip preparation snapshot covers 12 local dates across three sites. The existing registered video comparison requires 15 dates. The eight clips chosen for reader training and protocol development cover three sites and five local dates. [Registered video comparisons](webcoos_video.md)
4. Pick the first set and confirm its output directory at the check-in. Full training uses fresh output names. Files under `runs/prep-v1/smoke` are engineering checks. The data preparation and smoke results establish coverage and operability.

## Buoy launch sheet

```bash
P=runs/prep-v1
B=$P/buoys-depth
# Set S3_OOF to the downloaded stage-3 oof_spectral.csv.
scripts/measurements peaks --package "$B" --oof "$S3_OOF"
scripts/measurements direction --package "$B" --oof "$S3_OOF"
# Set ARM to the upstream selection at the check-in.
scripts/measurements plan --arm "$ARM" --stage probe --seed 0
scripts/measurements plan --arm "$ARM" --stage finetune --seed 0
```

Run the printed probe commands first. Each loads only the matching fold’s encoder and view mixer. One run produces held-out predictions for wind, horizon geometry, their combination, the frozen camera, and all inputs. Repeat the same commands with `--water-only` and distinct output directories for the below-horizon control. Inspect exclusion counts for missing horizons or unusable views. Evaluate a common set of sample IDs when comparing crops. Use the fine-tune commands for the two-output slope and wind-wave head once the frozen comparison warrants the cost. The fine-tune output encodes slope in percent. Divide by 100 before comparing with the dimensionless probe output.

```bash
scripts/measurements probes runs/measurement-probes/"$ARM"-s0-fold{0,1,2,3,4}
```

| Endpoint | Reference and comparison |
|---|---|
| Band-limited RMS slope | Primary is dimensionless deep-water RMS slope over 0.030–0.495 Hz. Compare all inputs against wind + gust + horizon geometry using paired out-of-fold RMSE, a 95% interval over station groups, and equal-group error. Inspect finite-depth slope using the 79 station water depths and the truncated high-frequency tail. Record errors per buoy, wind/swell regime, illumination and future collection week before claiming transfer. |
| Peak period | The depth package retains the original mean of peaks, nearest spectrum peak, peak of mean density, timestamp offsets and the peak range. 41,004 strips have a strict nearest peak. The probe can fit the nearest-record target. The S3 sensitivity table reuses identical predictions on common records. |
| Band energy and direction | Four absolute energy bands share edges at 0.030, 0.095, 0.155, 0.255 and 0.495 Hz. Score equivalent band height in metres and normalized shape separately. Compare the direct heads with a four-shape nonnegative dictionary learned on training buoys. The directional target is the energy-weighted first-moment magnitude over wind sea and its implied angular width. 37,841 strips meet the directional coverage rule. Score width MAE in degrees. S3’s 46-bin normalized output covers wind sea alone. Restrict any comparison against it to that same partition. |
| Visible foam | The 25-view annotation packet is in `$P/foam`. Use the filenames in `views.csv` and single-channel masks of the original dimensions. Code 0 for other, 1 for clear water, 2 for visible foam, 255 for unknown. Then run `scripts/measurements annotation fractions --packet "$P/foam" --out "$P/foam-fractions.csv"`. After masks cover multiple station groups, run `scripts/measurements annotation foam-probe --packet "$P/foam" --labels "$B/labels.csv" --out runs/foam-probe.csv` to compare brightness and RGB/texture regression on held-out groups. The denominator is reader-delineated visible water including foam. Use this sample to settle reader agreement in glare, twilight and residual foam before expanding station coverage. |

For the spectral prior, the prepared `era5-pilot-request.json` requests 24 directions × 30 frequencies at hourly cadence on October 1 for the grid around station 41004. Submit to CDS dataset `reanalysis-era5-complete` after the check-in if spectral fusion is selected. With a retrieved NetCDF, run the following. Inspect the grid, hours and decoded band energy against the local buoy bands before extending acquisition. A single station-day validates the decoder. Generalization requires matched physical spectra covering the training and held-out folds. Pass the same joined prior CSV with `--prior-bands` to all five probe commands. The corrected-camera comparison uses the same wind, geometry and model inputs on both sides. Only the upper three complete bands qualify. [ECMWF wave-spectrum encoding](https://confluence.ecmwf.int/pages/viewpage.action?pageId=185085702)

```bash
scripts/measurements era5_spectrum --station 41004 --netcdf "$ERA5_NETCDF" \
  --labels "$B/labels.csv" --out "$P/prior-41004.csv"
```

## Shore and breaker launch sheet

**Local crest timing.** Give one copy of the eight clocked videos in `$P/crest-readers` and one reader CSV to each independent reader. Mark each recognizable crest as it crosses the yellow midpoint. Record increasing video seconds, space-separated, within the indicated interval. Use `NONE` for an observed interval with zero arrivals and leave a cell blank for an undecided interval. Agree on a matching tolerance in seconds using reader disagreement on development clips, then freeze it. Use additional held-out clips for reporting. Keep both Oak Island views in one site. Report precision as total hits/calls, recall as hits/observed, and matched inter-arrival MAE as summed absolute error/matched intervals. Report the number of comparable intervals.

```bash
scripts/measurements period crest --stacks "$P/crest-stacks" \
  --annotations "$P/crest-readers/reader-A.csv" --reader A \
  --split site --tolerance "$TOLERANCE_SECONDS" --out runs/crest-site.csv
```

**Offshore period and height.** Set `VIDEO_PKG` to the prepared WebCOOS package and `VIDEO_FEATURES` to the registered midpoint probe’s `features/` directory containing `clips.csv`, `video.npy`, `still.npy` and `shuffled.npy`. The entry point matches clip IDs and reports predictions on the common subset for the brightness spectrum, its fold-trained ridge, three frozen feature arms and the fold mean. The full-clip comparison from the video worktree remains the registered motion baseline. Match a site/time hindcast separately if claiming added value over offshore models. Read date, week and site splits individually.

```bash
scripts/measurements period tm02 --package "$VIDEO_PKG" --features "$VIDEO_FEATURES" \
  --split date --out runs/shore-tm02-date.csv
scripts/measurements period tm02 --package "$VIDEO_PKG" --features "$VIDEO_FEATURES" \
  --displacement "$P/cdip-xyz-536/displacement.csv" --duration 1600 \
  --target mean_height_m --allow-unevaluated --split week --out runs/shore-mean-height-qc2.csv
```

The CDIP inventory matched 511 of the 536 clips to archived 1,600-second reference windows. Every matched record includes provider quality flag 2 (unevaluated). The strict flag-1 analysis therefore has zero eligible records for this snapshot. The explicit opt-in above runs a sensitivity study. Alternative fixed-duration endpoints are `max_height_m` and `tz_s`. Several clips can share one offshore reference. Keep shared events together and report unique reference windows. [CDIP displacement processing](https://cdip.ucsd.edu/themes/media/docs/documents/html_pages/dw_timeseries.html)

**Local breakers.** Duck supplies 1,782 matched thermal wave masks and individual lidar height and period targets across 53 runs and two days. Compare its compact mask CNN with mask fraction and the training mean on both day holdouts. Yin Set A supplies 9,000 verified videos from six sites; its breaker types now run through `deploy/dgx/breakers.sh` with bars and results in [breaker_types.md](breaker_types.md) (the R3D-18 / GRU commands below are superseded). Report Duck height/period MAE. The 536-clip match flagged 32 Yin clips overlapping two WebCOOS clips by camera and UTC interval. Keep shared footage together in any cross-dataset experiment. [Yin data paper](https://doi.org/10.1038/s41597-025-06005-5) · [Duck lidar study](https://hdl.handle.net/1773/46586)

```bash
for day in 2016-11-07 2016-11-08; do
  scripts/measurements train-breakers duck --index "$P/duck" --day "$day" \
    --device cuda --out "runs/duck-$day.csv"
done
for site in Site{A,B,C,D,E,F}; do
  for arm in video gru; do
    scripts/measurements train-breakers yin --source "$P/yin" --index "$P/yin-index-536" \
      --site "$site" --arm "$arm" --device cuda --out "runs/yin-$site-$arm.csv"
  done
done
```

**Ship stereo.** The archive index groups 612 minute files into 55 bursts on ten days and identifies duplicate minutes across two archives. The prepared table uses one minute per burst and a common rectified 24 m square with at least 95% joint radiance/elevation coverage. Fifty-three bursts qualify. All 53 elevation spectra peak at the longest admitted wavelength (12 m), so this crop supplies **zero interior wavelength peaks**. Defer peak-wavelength and peak-speed comparisons until a wider patch or another instrument resolves the peak. The code flags boundary peaks and prepares temporal surface-elevation RMS and interior period for a held-day radiance ridge and fold-mean comparison. The direct radiance spectral estimates accompany each row. Report site, bandwidth, eligible bursts and reconstruction coverage alongside errors. [Stereo source and calibration](https://hdl.handle.net/1773/38314)

```bash
scripts/measurements stereo compare --features "$P/stereo-screened.csv" --out runs/stereo-held-days.csv
```

## Rebuild, transfer and provenance

The prepared artifacts live under this worktree’s ignored `runs/prep-v1/`. Git contains the builders, recipes, tests and this guide. The portable buoy folder contains verified image bytes, the per-view horizon table, labels, station checks and a SHA-256 manifest. Copy `portable-buoys/` to the training host as the package and transfer the selected matching checkpoint directories. On that host verify with `sha256sum --quiet -c SHA256SUMS` from the package. For generated commands, set `MEASURE_PREP` to the host prep root and install the portable directory there as `buoys-depth`. Set `MEASURE_RUNS` to its upstream runs root. The other experiment inputs can be transferred independently. Preserve the source archive’s license when handling the Yin data.

The launcher locates the main worktree’s data and Python environments automatically. On a fresh host install the project’s ML, imaging and NetCDF requirements and supply `ffmpeg`. Set `WAVE_ANALYSIS_DATA`, `MEASURE_PYTHON`, and `MEASURE_ML_PYTHON` to explicit paths as needed. The two torchvision breaker checkpoints are cached locally under `runs/measurement-weights/hub/checkpoints/`. Copy that cache to an offline host or provide `TORCH_HOME`. The project’s existing video probe provides the video feature cache after its extraction stage.

To reproduce preparation from the original observations, use a fresh destination. The examples below assume the same input data snapshot. Keep the depth table, per-station bandwidth checks, reference quality fields, clocked reader packets, split IDs and provider manifests with the experiment.

```bash
D=runs/prep-rebuild
mkdir -p "$D"
BASE="${WAVE_ANALYSIS_DATA:-../Wave_Analysis/data}/processed/packages/ndbc_buoycam_spectral_v3cv"
scripts/measurements depths --labels "$BASE/labels.csv" --out "$D/station-depths.csv"
scripts/measurements labels --depths "$D/station-depths.csv" --out "$D/buoys-depth"
scripts/measurements export --package "$D/buoys-depth" --out "$D/portable-buoys"
scripts/measurements shore stacks --transects configs/experiments/measurement_transects_v1.csv \
  --seconds 64 --out "$D/crest-stacks"
scripts/measurements annotation crest --stacks "$D/crest-stacks" --out "$D/crest-readers"
scripts/measurements annotation foam --per-light 8 --out "$D/foam"
scripts/measurements breakers duck --out "$D/duck"
scripts/measurements yin --clips --out "$D/yin"
scripts/measurements breakers yin --source "$D/yin" --out "$D/yin-index"
scripts/measurements references cdip-windows --out "$D/cdip-windows.csv"
scripts/measurements heave --index "$D/cdip-windows.csv" --out "$D/cdip-xyz"
scripts/measurements references stereo --out "$D/stereo.csv"
scripts/measurements references stereo-sample --index "$D/stereo.csv" --out "$D/stereo-sample.npz"
scripts/measurements stereo features --index "$D/stereo.csv" --out "$D/stereo-screened.csv"
scripts/measurements era5_spectrum --station 41004 --date 2026-10-01 --out "$D/era5-request.json"
```
