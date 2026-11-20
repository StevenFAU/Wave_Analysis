# Wave-sensing experiments · first check-in, November 19, 2026

Prioritize the camera-plus-forecast comparison. The autumn GFS archive is now local, and it supplies a finite height for 92.7% of the buoy strips. The practical question is how much the camera reduces error beyond the best forecast and wind estimate available on those same strips at unseen locations.

## Source checks

| Source | What was measured | Consequence |
|---|---|---|
| NOAA Nearshore Wave Prediction System | Collecting station spectra, wave-system tracks and CG1 grids from MHX, AKQ, ILM and HFO. Four completed UTC days, November 15–18, took 900.5, 752.6, 750.8 and 752.2 MB. On the first day, spectra and tracks accounted for 204.4 MB and the 12 grids added 696.1 MB. | Budget roughly 24 GB per 30 days at the four-day mean. The worktree's transient daily timer revisits the five-day provider window at 13:30 UTC. Re-establish the timer after a reboot until the checked-in units are installed. [Collection details](../datasets/nwps.md) |
| Shore camera audio | One archived MP4 each from Currituck Hampton Inn, Jennette South and the Oak Island east and west cameras contained one H.264 video stream and zero audio streams. | Close the sound experiment for the sampled product. The Currituck sample was from September 24, 2026, Jennette from October 9, 2024, and Oak Island from August 14, 2024. |
| EarthScope | Downloaded 24 gap-free hours from IU.KIP on January 15, 2011, and N4.Y60A on January 15, 2023. The median hourly peak in the secondary microseism band was 0.168 and 0.217 Hz, respectively. | The waveform archive is usable. The longer Waimea download is underway. Before comparing amplitudes, remove the epoch-specific instrument response. The predictive test compares seismic features with climatology on a held-out year. [Channels and QC](../datasets/earthscope.md) |
| GFS-Wave | Downloaded and parsed all 212 six-hour issue cycles, September 23–November 14, 2026. The 109,816 short-lead station forecasts cover 74 of 79 buoys. Of 41,091 strips, 38,636 have a bulletin and 38,093 have finite forecast height. | Fit every comparator on the same eligible rows. Finite-height coverage in folds 0–4 is 8,897/9,440, 7,392/7,823, 7,476/8,039, 6,602/8,063 and 7,726/7,726. Five stations lack bulletins entirely (45002, 45004, 45006, 45012, 46081). Matching uses nominal issue time. |

## Library request

The five *Ocean Engineering* PDFs are Y. Liu and Xu et al. on spectral image representations, Ahammed and Dang on polarization-aware vision, Li et al. on multi-domain reconstruction, G. Zhang et al. on Beaufort ordinal learning, and H. Liu, Qin and Ti on video transfer. Compare their targets, splits, geometry and baselines from the full texts before making a novelty claim. [Five DOI links and extraction questions](sensing_reading.md)

## What counts as a result

| Test | Registered comparison and decision | Status |
|---|---|---|
| 1 · Added value of a buoy camera | Compare gated camera fusion against the stronger validation-selected ridge or boosted model using ERA5, GFS and measured wind. On total significant height **and** wind-wave height, the upper 95% bound for the paired RMSE difference must be below zero. Bootstrap entire geographic groups. Withhold the anemometer in a separate arm and permute camera predictions within buoy and forecast-height bin to test image-specific information. | Forecast-to-strip table ready. Corrector fitting and fusion scoring are pending. Full protocol in `codex/sensing-model`, `docs/experiments/sensing_model_bar.md`, commit `f0db1b786`. |
| 2 · Mean individual wave height from shore clips | Classify instrument-measured mean individual height on unseen local dates. The lower 95% bound for whole-clip accuracy gain must exceed zero against both a midpoint still and the training-date majority. Bootstrap whole dates and report the sensitivity to the instrument's unevaluated flag separately. | Two Currituck windows contained 221 and 220 good upcrossing waves, with mean heights 0.520 and 0.502 m. Both fall below the registered classification range. This verifies label construction. The collector is building the 18-date development table; class and date coverage determine whether the primary comparison is estimable. |
| 3 · Reproduce the published model comparisons | On the same held-out examples across three seeds, the project's accuracy gain over Choi and Kim must have a positive lower 95% bound. Its height RMSE difference versus Yang and Kamagata must have a negative upper bound. Decide each comparison separately. Resample whole buoys for the station split and whole dates at shore. | Choi uses the paper's single-image buoy classifier. Its temporal ship regressor requires a different input. The Kim buoy target is a proxy; the shore target is measured. Yang uses relative pitch with a documented projection assumption and includes a no-attitude control. Video waits for eligible shore labels and coordination with the video run. |

The shore protocols are in `codex/sensing-shore` at `docs/experiments/sensing_per_wave_bars.md` (decision rules `0d331c01f`, crossing and quality convention `cbdd6523b`). The strip protocol is in `codex/sensing-strips` at `docs/experiments/sensing_paper_bars.md` (decision rules `6b506bd42`, training recipe `2c0784766`). The previous sealed buoy results are retrospective. The separate November 27 fresh-strip confirmation keeps its original protocol.

## Compute and next check-in

One Choi seed is running on a shared H200, on the sealed station split, with four other project jobs already running at submission. Run `buoy-station-20261119t1805-seed0-paper-choi` uses training commit `0e945272d`; its first epoch completed. Allow roughly 40–55 minutes per strip run using the prior planning estimate. Shared-device load and architecture affect elapsed time. Four strip arms × three seeds gives 8–11 GPU-hours at that rate, before the optional five-fold runs.

The remaining GPU work is ordered for the next check-in. Only Choi seed 0 was submitted. Fetch its predictions and settings, verify completion, then run the other two Choi seeds, Kim snapshot, and Yang with and without attitude. Follow with the frozen frequency-band/wind probes, the time-exposure/brightest-image probes, the shore video comparisons, then zero-shot sea state. Grouped strip folds and probe-qualified fine-tuning follow. The slope and continued-pretraining jobs stay with their existing owners. The seismogram and shore-label collectors can continue independently.

For the next operator, `systemctl --user list-units --all 'wave-sensing-*'` shows collection state. Look for `runs/gfs-wave-strip-match-v1.parquet` on the model worktree and `runs/cdip-development-v1.csv` on the shore worktree. A completed label build must report distinct reference windows, measured-height range and eligible classes by date before choosing a shore training job. Use the strips worktree's `deploy/dgx/run.sh` to check capacity and retrieve run artifacts.

## Decisions for the project owner

Fetch the five PDFs and send the Field Research Facility request drafted at `.claude/drafts/frf_argus_request_email.md` if desired. Keep one GPU allocated to this track until the next budget decision. Merge the local branches when ready. MFWAM requires an optional Copernicus Marine account; the GFS comparison can proceed now. Paid API use, the Yin clip licence and human readers for the crest-timing and crop checks apply only to later work.
