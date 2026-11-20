# Claude handoff · Wave camera measurements · November 19, 2026

**Start with the five-fold frozen buoy slope probe using DINOv3-gentle, seed 0.** The question is whether camera pixels improve estimates of buoy-measured surface slope beyond wind speed, gust and horizon geometry at unseen locations. The user asked to review the overnight runs together before launching the next set. The measurement code, data, and checks are ready for that conversation. This handoff records the morning results and cluster check. Production slope runs are pending.

## Where to work

Use the sibling worktree `../Wave_Analysis-measurements` (next to the main checkout), branch `codex/wave-measurement-experiments`. Experiment code is committed at `0674b9ce6`. The prepared data and portable package are in this worktree’s ignored `runs/prep-v1/`. Upstream predictions and checkpoints are in the main checkout's `runs/`. The `scripts/measurements` launcher resolves those paths from this machine. The preparation passed 329 unit tests, a complete portable-package checksum verification, and small CPU runs of both the frozen probe and slope fine-tune. [Launch and rebuild guide](measurement_checkin.md) · [Research rationale](../../reports/Wave%20measurements%20to%20learn%20next.md)

## What finished overnight

| Study | Result | Consequence |
|---|---|---|
| Geographic cross-validation | All 30 height runs finished. DINOv3-gentle had significant-height RMSE of 0.532 m versus 0.550 m for ImageNet ConvNeXt. The paired delta was −0.0188 m. The wind-wave delta was −0.0024 m, with a buoy-bootstrap 95% interval of [−0.0080, +0.0037]. | The registered lower-height-RMSE rule selects DINOv3 for transfer. The separate superiority criterion required both endpoints to improve and was unmet. |
| Partition-aware fusion | On 39,767 matched strips, the gated partition swap reduced total-height RMSE from 0.288 m for ERA5 + wind to 0.267 m. The group-bootstrap interval for the difference was [−0.034, −0.009] m. | Retain the wind and model baselines when measuring the camera’s contribution. |
| Spectral heads | The three-seed report favors ERA5 for total peak period, energy period and mean direction. The camera beat ERA5 on wind-wave period, 0.69 s versus 0.93 s RMSE. | Prioritize slope and wind-sea structure. In a morning sensitivity check across 40,726 common strips, nearest-record peak RMSE was 3.57 s for the camera and 1.95 s for ERA5. Changing the peak definition preserved the ordering. The direction control using solar position, location and time also beat the camera, 61.64° versus 75.83° mean absolute error across 40,038 strips. |

The three upstream reports are under the main checkout. Their denominators and confidence intervals use the respective study protocols. [Height CV](../../../Wave_Analysis/runs/buoycam-cv-stage1-20261119/report.md) · [Fusion](../../../Wave_Analysis/runs/buoycam-cv-fusion-20261119/report.md) · [Spectral heads](../../../Wave_Analysis/runs/buoycam-cv-spectral-20261119/report.md)

## First actions at check-in

Run these from **the measurement worktree**. The plan prints local commands and matched checkpoints. The final two commands reproduce the morning sensitivity checks using saved predictions.

```bash
cd ../Wave_Analysis-measurements   # from the main checkout
scripts/measurements status
GPU_HOST=<ssh-alias> deploy/dgx/run.sh check
scripts/measurements plan --arm dinov3 --seed 0 --stage probe
S3=../Wave_Analysis/runs/buoycam-cv-spectral-20261119/oof_spectral.csv
scripts/measurements peaks --package runs/prep-v1/buoys-depth --oof "$S3"
scripts/measurements direction --package runs/prep-v1/buoys-depth --oof "$S3"
```

**Cluster snapshot at 8:12 a.m. Eastern, November 19.** Three project containers were running the continued-pretraining encoder’s frozen height evaluations. GPU 4 had the most headroom in the ten-second sample, averaging 47% busy with 59 GiB free. All eight H200s showed activity. Scratch had 8.6 TiB free. Recheck utilization and existing jobs at launch time. The estimate for five sequential frozen folds is **3–5 hours on one H200**, extrapolated from CPU preprocessing and earlier GPU evaluations. The optional ten-epoch fine-tune is **3½–6¼ hours** for five folds, based on 15 comparable H200 runs. Time the first full probe fold and update the remaining ETA.

### Running the slope set after the check-in

1. Stage the committed code and `runs/prep-v1/portable-buoys/`. The server’s verified `ndbc_buoycam_heights_v2cv` package has the same 41,091 images. Reuse those bytes and transfer about 59 MiB of measurement metadata and horizons, then verify `SHA256SUMS`. The GPU host uses Docker over SSH and the project scratch root mounts at `/data`.
2. Start fold 0 with its matching seed-0 DINOv3 checkpoint. The printed plan uses **desktop paths**. Inside the existing CUDA environment invoke `scripts/measurement_buoy_probe.py` with the container’s package, checkpoint and output paths and the measurement checkout’s `src` on `PYTHONPATH`. The generic height-training container entry point adds `--config` and `--split`, which the frozen-probe CLI does not accept. Its correct flags are shown by the plan.
3. Check fold 0’s elapsed time, checkpoint identity, eligible strips and `predictions.parquet`. Complete folds 1–4 using their own checkpoints. Fetch the five output directories under `runs/measurement-probes/` with the names from the plan. Aggregate with `scripts/measurements probes runs/measurement-probes/dinov3-s0-fold{0,1,2,3,4}`. Report dimensionless slope RMSE for all inputs versus wind + geometry, their paired station-group interval, equal-group error and coverage. That comparison determines whether to spend the fine-tuning budget.

## Other prepared experiments

- **Shore video.** The morning package lists 1,282 clips on 20 local dates in both the whole-clip index and pixel spectra. Check the intersection with the registered video feature cache before applying the 15-date threshold. The CDIP and Yin overlap files named `-536` describe the earlier 536-clip snapshot. Refresh both against new clip IDs before using the expanded package. All 511 earlier matched CDIP windows included provider flag 2 and support sensitivity analysis only.
- **Crest timing and foam.** Eight clocked clips at three sites await independent crest timestamps. Twenty-five buoy views await visible-water and foam masks. Reader packet locations, mask codes, and the scoring protocol are in the launch guide.
- **Breakers and stereo.** Yin has 9,000 labeled clips across six sites. Duck has 1,782 thermal masks matched to lidar height and period. For stereo, all 53 usable bursts hit the longest admitted wavelength in the prepared crop. Defer wavelength and phase-speed claims until a larger field resolves the peak. Period and elevation RMS remain available.
- **ERA5 band correction.** A station-day request and decoder are prepared. The acquisition and coverage check precede fitting any spectral correction.
