# Claude handoff for two buoy-camera experiments

## Fixed historical comparisons

The storm-video workflow has its own queue. These are separate experiments on the September–November historical buoy archive. Both use existing geographic folds. The November confirmation and the subsequent 14-day wind-wave replication have their own frozen models and scoring windows.

### A. Train before October 29 and score later at unseen buoys

Use the original 79-buoy height package and five outer geographic folds. For each fold use only its assigned training and validation strips dated before October 29, 2026 at 00:00 UTC. This cutoff applies to training labels, target normalization, checkpoint selection and forecast-corrector selection. Test on its held-out buoys from October 31 at 00:00 UTC. Use the exact 7,855 matched strips (67 buoys, 39 groups) of the historical [date-cutoff comparison](wind_wave_reliability.md#earlier-calibration-later-scoring).

Train one seed-0 camera per fold from public DINOv3 ConvNeXt-Tiny weights using the established gentle height recipe and total-height validation criterion. Fit the original ERA5 + GFS-Wave + measured-wind corrector on the same calendar and geographic roles with its original eligibility checks. Report camera alone, corrected forecast alone and the arithmetic mean of their wind-wave height predictions in metres. The sole primary contrast is the 50/50 blend minus forecast in pooled RMSE. Require its paired 95% upper endpoint below zero over 2,000 whole geographic-group resamples with seed 0. Report equal-group and per-fold results. Use all five folds at seed 0 regardless of interim outcomes. This is a retrospective test with ERA5 and an already inspected historical scoring window.

### B. Estimate swell from the part of the image below the horizon

Use the same five original seed-0 geographic height checkpoints for whole views and water crops. Use the existing below-horizon transform and its minimum of three usable water views. Extract the water embeddings and reuse the matching cached whole-view embeddings. Predict the lowest fixed band (0.030–0.095 Hz) as equivalent height in metres, four times the square root of its integrated variance. A zero-total spectrum is missing. A zero swell band in an otherwise valid spectrum is allowed.

Use the identical eligible training and test rows for each arm within each fold. Fit the established training-standardized ridge (penalty 10) on training buoys. Evaluate whole features alone, water features alone, measured wind alone, and each image input with wind. Wind features are speed, gust, both squares and their product. Clip heights at zero. The primary contrast is water + wind minus wind in pooled RMSE, with the same paired geographic bootstrap as A. Compare water + wind against whole + wind to estimate the cost of the crop. Report pooled R², equal-group contrasts, per-fold outcomes and coverage before and after filtering. Resizing and discarding part of the scene can also affect skill. This control tests information retained under that specific crop.

## Where Claude should start

Use the main desktop checkout that holds `data/` and `runs/`. The prepared launcher checkout is `runs/hpc-buoy-launch`, pinned to `68b227ee059b698cb18c38f90de074e2b7ed67eb`. That code is already on the H200 server. Its locked environment is `1d319000489b` with torch 2.14.0+cu130. The public DINOv3 backbone loaded there with networking disabled. The server has both verified packages, and its five frozen checkpoint hashes match the five desktop whole-view feature caches. Neither new GPU experiment was launched during preparation. The final whole-clip and storm-video containers were both running at the last check.

The CPU forecast fit for A is complete in `runs/buoy-cutoff-forecast-v1/`. All five correctors produced the expected 7,855 scored strips. The prepared `forecast.parquet` has SHA-256 `07bb221e14e9d0388e43dfbb483d93c4dd6cde41356eeff5d636cd0a277891f5`. The sibling `summary.json` identifies the source snapshots. Early-strip ERA5 interpolation uses only records before the cutoff. `coverage.csv` records geographic roles and dates before forecast input filtering. A fold-0 CPU smoke run exercised the actual pretrained camera trainer. A separate CPU smoke run extracted and saved below-horizon features with the original fold-0 checkpoint. Smoke outputs are for plumbing checks only.

### Launch on the shared server

Work from the main repository root in Bash. Set `GPU_HOST` to the lab server's configured SSH alias. The launchers reuse the desktop's authenticated SSH connection. Use the existing project launch lock when starting each job. Recheck free memory and project occupancy inside the lock, keep the project within its seven-container limit, and use a GPU with at least 40 GiB free. Give the active storm work priority. Allow one new follow-up container at a time. The five-fold camera baseline previously used 3.94 H200-hours in aggregate. The previous five-fold water extraction used about 30 minutes.

```bash
: "${GPU_HOST:?Set GPU_HOST to the configured lab SSH alias}"
export GPU_HOST
PIN=68b227ee059b698cb18c38f90de074e2b7ed67eb
LAUNCH="$PWD/runs/hpc-buoy-launch"
if [[ ! -d "$LAUNCH" ]]; then
  git worktree add --detach "$LAUNCH" "$PIN"
fi
test "$(git -C "$LAUNCH" rev-parse HEAD)" = "$PIN"
RUN="$LAUNCH/deploy/dgx/run.sh"
MEASURE="$LAUNCH/deploy/dgx/measure.sh"
LOCK="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/wave-analysis-gpu-launch.lock"
"$RUN" status

launch_one() (
  set -euo pipefail
  flock -x 9
  snapshot=$("$RUN" check)
  echo "$snapshot" >&2
  count=$(awk '/^our containers running:/ {print $NF}' <<< "$snapshot")
  [[ "$count" =~ ^[0-9]+$ ]] && (( count < 7 )) || { echo "Defer: project slots occupied" >&2; exit 1; }
  devices=$(ssh -T -o BatchMode=yes "$GPU_HOST" 'set -o pipefail
    docker ps -q --filter label=app=wave-analysis |
    xargs -r docker inspect -f "{{range .HostConfig.DeviceRequests}}{{range .DeviceIDs}}{{println .}}{{end}}{{end}}"')
  gpu=$(awk -v taken="$devices" '
    BEGIN {n=split(taken, used, /[[:space:]]+/); for (i=1;i<=n;i++) busy[used[i]]=1}
    NF==3 && $1~/^[0-7]$/ && $3>=40 && !($1 in busy) {print $2, $1}
  ' <<< "$snapshot" | sort -nk1,1 -nk2,2 | awk 'NR==1 {print $2}')
  [[ -n "$gpu" ]] || { echo "Defer: no suitable GPU" >&2; exit 1; }
  GPU="$gpu" "$@"
) 9>"$LOCK"
```

For A, run the block for one fold at a time. Start with `k=0`, fetch the completed run and check its exit status, then repeat with 1, 2, 3 and 4. Before a retry, check the running containers and already fetched run IDs. A closed SSH session can leave a healthy detached training job on the server.

```bash
k=0
launch_one env PKG=ndbc_buoycam_heights_v2cv \
    CONFIG=configs/experiments/ndbc_buoycam_heights_v2cv.yaml \
    TRAIN_SCRIPT=scripts/train_wave_heights.py TAG=cutoff-oct29 \
    "$RUN" train "fold$k" --seed 0 --backbone timm:convnext_tiny.dinov3_lvd1689m \
    --lp-epochs 2 --backbone-lr-scale 0.1 --date-split 2026-10-29T00:00Z 2026-10-31T00:00Z
```

For B, use a single container. The launcher processes all five frozen checkpoints sequentially. Record the run name printed at launch.

```bash
checkpoints=(
  0:buoy-fold0-20261118t2234-seed0-cv-dinov3-gentle
  1:buoy-fold1-20261118t2342-seed0-cv-dinov3-gentle
  2:buoy-fold2-20261118t2346-seed0-cv-dinov3-gentle
  3:buoy-fold3-20261119t0038-seed0-cv-dinov3-gentle
  4:buoy-fold4-20261119t0130-seed0-cv-dinov3-gentle
)
launch_one env PKG=ndbc_buoycam_slope_v1 EXTRA="--water-only --save-features" \
    "$MEASURE" probe swell-water-s0 "${checkpoints[@]}"
```

### Fetch, score and publish

For each run, set `RUN_ID` to the exact ID printed by the launcher. Follow its log until the final exit code. Fetch using the **main checkout's** launcher so outputs land beside the prepared desktop inputs.

```bash
"$RUN" logs "$RUN_ID" -f
deploy/dgx/run.sh fetch "$RUN_ID"
tail -1 "runs/$RUN_ID/train.log"
```

For interrupted monitoring, `"$RUN" status` and `"$RUN" logs "$RUN_ID"` recover the state. The terminal log ends with `finished with exit code 0` and a timestamp on success. The follow command uses remote `tail -f`, so detach the log reader when that line appears. Each completed camera run should have `metrics.json` and `predictions.csv`. Each of the five water fold directories should have `run.json`, `feature_rows.csv` and `features.npy`.

Run the following scores from the main desktop checkout. Set `WATER_RUN_ID` to the fetched B run. The default whole-view input is the cached seed-0 set at `runs/claims-inputs/t5-s0`. The A scorer expects five run directories and checks the fold settings and exact validation and test strip IDs. If there are retries, pass the five successful directories explicitly.

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
uv run --locked --isolated --group forecast python scripts/buoy_hpc_followup.py heights \
  --runs runs/buoy-fold[0-4]-*-seed0-cutoff-oct29
uv run --locked --isolated --group forecast python scripts/buoy_hpc_followup.py swell \
  --water "runs/$WATER_RUN_ID"
```

The default output directories are `runs/buoy-heights-followup-v1` and `runs/buoy-swell-followup-v1`. Use `--out` with a new path for a reproduction. For A, the forecast inputs can be rebuilt with `uv run --locked --isolated --group forecast python scripts/buoy_hpc_followup.py forecast --out runs/buoy-cutoff-forecast-replay`, then selected via `heights --forecast runs/buoy-cutoff-forecast-replay`.

Report the primary decision from each `summary.json`, paired RMSE differences and intervals in metres, equal-group sensitivity from `contrasts.csv`, fold scores from `rmse.csv` and B's crop coverage. For B, include the pooled R² of both image arms and wind. Record the actual run IDs, elapsed GPU time and exclusions alongside the results. Commit and push the small report and score tables. Keep weights, embeddings and row-level outputs under the existing ignored `runs/` tree.

## Results

### A. Train before October 29 and score later at unseen buoys

**The 50/50 blend met the primary criterion.** All five cameras and forecast correctors saw only strips dated before October 29. Averaging the camera with the corrected forecast reduced pooled wind-wave RMSE by **3.07 cm** relative to the forecast alone. The paired 95% interval was −5.03 to −1.80 cm. Its upper endpoint is below zero, as required.

Scoring covers 7,855 strips from October 31 through November 13 at 67 held-out buoys in 39 geographic groups. Measured wind-wave height averaged 1.10 m with a standard deviation of 0.89 m.

| Prediction (fitted on early strips only) | Pooled RMSE (cm) | Pooled R² | Groups where better than forecast |
|---|---:|---:|---:|
| Corrected forecast (ERA5 + GFS-Wave + measured wind) | 26.0 | 0.915 | — |
| Camera | 29.3 | 0.892 | 14/39 |
| 50/50 blend | 23.0 | 0.934 | 32/39 |

The gain held under each sensitivity check. Weighting each geographic group equally gave −4.49 cm (−6.88 to −2.29 cm). Excluding group g41040 gave −3.12 cm (−5.07 to −1.83 cm). The blend beat the forecast in every fold:

| Fold | Scored strips (buoys, groups) | Forecast (cm) | Camera (cm) | Blend (cm) | Blend − forecast (cm) |
|---:|---:|---:|---:|---:|---:|
| 0 | 1,925 (16, 1) | 26.0 | 30.2 | 23.5 | −2.5 |
| 1 | 1,481 (14, 8) | 32.9 | 37.5 | 30.3 | −2.7 |
| 2 | 1,457 (12, 8) | 26.6 | 23.7 | 20.7 | −5.8 |
| 3 | 1,430 (12, 11) | 20.6 | 27.4 | 19.4 | −1.2 |
| 4 | 1,562 (13, 11) | 22.4 | 25.3 | 18.8 | −3.5 |

Compare the [earlier date-cutoff comparison](wind_wave_reliability.md#earlier-calibration-later-scoring) on the same 7,855 strips. There the base models had been fitted geographically on all dates. Restricting their training to early dates raised forecast RMSE from 24.3 to 26.0 cm and camera RMSE from 27.9 to 29.3 cm. The unfitted 50/50 blend still beat the early-only forecast by 3.1 cm. Earlier, the blend with a fitted constant weight beat its forecast by 2.5 cm.

**Coverage.** The camera runs held out 9,397 later test strips at 74 buoys. Of these, 9,087 had a wind-wave label, and 7,855 (86% of labelled strips) are in the fixed scoring population. Of the 1,232 labelled strips left out, 828 fall after the historical eligibility archive ends (November 13, 01:00 UTC). The other 404 lacked a forecast input or measured wind. Camera checkpoints were selected on 23,675 validation strips. Validation and training both ended by October 28, 23:20 UTC (`calibration_coverage.csv`).

What this does and does not show: this is a retrospective test. ERA5 is a reanalysis that was not available at the time of each strip. The scoring window had already been inspected in the earlier comparison. It shows that the camera's added value is not explained by the camera or the forecast corrector having trained on dates inside the scoring window. It is not a prospective forecast claim.

**Run record.** The five seed-0 camera runs used commit `68b227ee0` on one shared H200: `buoy-fold0-20261121t0307-seed0-cutoff-oct29`, `buoy-fold1-20261121t0345-…`, `buoy-fold2-20261121t0424-…`, `buoy-fold3-20261121t0504-…` and `buoy-fold4-20261121t0543-…`, each ending `-seed0-cutoff-oct29`. They ran one after another from 03:07 to 06:22 UTC on 2026-11-21. Folds took 36.7–39.0 minutes each, 3.18 GPU-hours in total. Every run exited with code 0; there were no retries and no exclusions. The scorer checked each run's settings and its exact validation and test strip IDs. The forecast input was `runs/buoy-cutoff-forecast-v1/forecast.parquet` (SHA-256 `07bb221e…891f5`). Score tables: `reports/buoy-hpc-followup/heights/` (`summary.json` with prediction hashes, `rmse.csv`, `contrasts.csv`, `scored_coverage.csv`, `calibration_coverage.csv`, `forecast_preparation.json`). To reproduce the coverage table:

```bash
uv run --locked --isolated --group forecast python scripts/buoy_followup_coverage.py heights \
  --runs runs/buoy-fold[0-4]-*-seed0-cutoff-oct29 --out runs/heights-scored-coverage.csv
```

### B. Swell from the part of the image below the horizon

**The water crop failed the primary test.** Adding water-crop features to measured wind changed the pooled swell-height RMSE by **+0.24 cm** (95% paired interval −2.23 to +2.02 cm). The criterion needed the upper endpoint below zero. The whole view carried far more swell information through the same checkpoints. Water + wind was **18.1 cm worse than whole + wind** (+15.4 to +22.1 cm) and lost in 42 of 43 geographic groups.

The scored target is the 0.030–0.095 Hz band's equivalent height. On the scored test strips it averaged 0.85 m with a standard deviation of 0.95 m. Each ridge fit used the same rows for all five arms within a fold.

| Ridge input (fit on training buoys) | Pooled RMSE (cm) | Pooled R² |
|---|---:|---:|
| Measured wind | 88.3 | 0.13 |
| Water crop | 93.8 | 0.02 |
| Water crop + wind | 88.6 | 0.13 |
| Whole view | 70.8 | 0.44 |
| Whole view + wind | 70.4 | 0.45 |

The water crop alone barely improved on a constant prediction. Weighting each geographic group equally gave a small gain for a typical group: water + wind minus wind was −1.76 cm (−3.12 to −0.24 cm), better in 22 of 43 groups. The pooled loss came from fold 0. That fold is a single geographic group of 18 buoys with 23% of the scored strips. Water + wind did worse there (108.8 vs 104.6 cm) and slightly better in the other four folds (by 0.5–1.9 cm). Excluding group g41040 left every contrast essentially unchanged.

| Fold | Scored test strips | Wind (cm) | Water + wind (cm) | Whole + wind (cm) |
|---:|---:|---:|---:|---:|
| 0 | 8,626 | 104.6 | 108.8 | 91.5 |
| 1 | 7,196 | 82.9 | 81.0 | 67.5 |
| 2 | 7,669 | 88.0 | 86.3 | 66.9 |
| 3 | 7,739 | 81.5 | 80.8 | 57.8 |
| 4 | 7,104 | 79.0 | 78.5 | 59.0 |

**Coverage.** Of 41,091 held-out test strips, 38,334 were scored (93.3%; 77 buoys, 43 groups). Of 132,274 training strips, 123,103 were used (93.1%). The crop's minimum of three usable water views was the main filter: it alone removed 2,221 otherwise-eligible test strips (5.4%) and 7,555 training strips (5.7%). Missing wind removed 486 test strips and zero-total spectra removed 50. Every strip had whole-view features. As the protocol allows, 5,238 scored test strips (13.7%) had a zero swell band. Per-fold counts are in `filter_breakdown.csv`.

What this does and does not show: the whole-view encoders were trained on full frames for total height. The crop resizes the water and drops the horizon, sky and hull. Through these encoders, little of the swell signal survives this crop. It does not show that the water pixels themselves lack swell information.

**Run record.** Water features: `buoy-probe-20261121t0229-swell-water-s0`, commit `68b227ee0`, one shared H200, 2026-11-21 02:29–03:06 UTC. That is 0.61 GPU-hours, with the five folds taking 428–440 s each. Whole-view features: cached `runs/claims-inputs/t5-s0`. The scorer confirmed that each fold's whole-view and water checkpoints have the same hash. Score tables: `reports/buoy-hpc-followup/swell/` (`summary.json`, `rmse.csv`, `contrasts.csv`, `coverage.csv`, `filter_breakdown.csv`). Row-level predictions and embeddings stay under `runs/`. To reproduce the coverage breakdown:

```bash
uv run --locked --isolated --group forecast python scripts/buoy_followup_coverage.py swell \
  --water runs/buoy-probe-20261121t0229-swell-water-s0 --out runs/swell-filter-breakdown.csv
```
