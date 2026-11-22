# Wave_Analysis repository: state of play as of 2026-11-19

Sources are local files only (main checkout at `1dfcc5f29` plus six worktrees). Every bullet ends with a source key; the keys resolve to absolute paths in the link definitions at the end of the file. Where a worktree is ahead of main, the worktree's file is cited and named. Key to the worktrees ([GIT]):

| Key suffix / place | Branch @ commit | Ahead of main | What it holds |
|---|---|---|---|
| main checkout | `main` @ `1dfcc5f29` (origin/main is one commit behind) | — | everything up to bars B1-B3, protocol check R, video design, the first two reports |
| strips-grouped-cv worktree | `claude/strips-grouped-cv-085fd2` @ `80c7e1390` | 6 commits | grouped CV (B2-CV), partition fusion (S2), spectral heads (S3), updated "Where we stand" |
| trusting-lalande worktree | `claude/trusting-lalande-4a4e8b` @ `0a981581b` | 7 (the 6 above + 1) | the S2 new-weeks confirmation tool; its experiment doc is the newest and a superset of strips-grouped-cv's |
| video-whole-clip worktree | `claude/video-whole-clip-491c90` @ `6fd0daef8` | 7 | whole-clip code, the interim scored video run (M1-M5), development dates; uncommitted physics-v2 code |
| dinov3-continued-pretraining worktree | `claude/dinov3-continued-pretraining-fc2b1a` @ `12569d72a` | 10 | DINOv3 ViT-B continued pretraining (FINO recipe), checkpoint chosen, bar P1 unscored |
| `Wave_Analysis-measurements` | `claude/wave-measurements-review` @ `e70873bd3` (one commit on top of `codex/wave-measurement-experiments` @ `efc29768d`) | 6 | measurement experiments prepared (slope, crest timing, foam, breakers, stereo), bar M-S1 fixed, nothing scored |
| `Wave_Analysis-wis` | `wis-hindcast` @ `a1f7cea32` | 0 (already contained in main) | nothing in flight |

## Data held: collections, counts, sites, date ranges, label sources, quirks, usage restrictions, and registry candidates not acquired

### Takeaway
The project's distinctive holding is the NDBC buoy-camera archive (41,091 packaged daylight/twilight six-view strips at 79 buoys, 2026-09-23 to 2026-11-14, labels from the same hull, archive still growing hourly), plus four North Carolina shore-camera views with multi-year stills paired to CDIP Waveriders 9-29 km away, a growing MP4 sample (1,282 clips packaged on 20 local dates on the morning of 2026-11-19), the 2009-2013 Waimea archive, and ERA5/WIS model baselines. Four small external research sets (ship stereo, Duck IR + lidar, Agulhas WASS, Yin breaker clips) are on disk for method checks; about twenty other registry entries (CoastCam, SOCIB, WW3, satellite products, KOLOMVERSE, Guimarães stereo, etc.) are documented but not acquired.

### Cited Findings

**NDBC BuoyCAM strips (same-hull labels)**
- Camera and wave sensor are on the same buoy; the label is `WVHT` from NDBC's realtime `.spec` summary, computed on board from a 20-min motion record; NDBC does not transmit the raw motion series; one record's 90 % interval is about −10 % to +15 % of H_m0 (Earle 2003). — [BUOY]
- 94,187 strips archived at 80 buoys from 2026-09-23 to 2026-11-14 (hourly, day and night); 90,434 have a label; **kept in the package: 41,091 strips at 79 buoys** (median 563 per buoy, range 35-649); 35,843 have all six views usable; 34,460 day, 2,757 low sun, 3,868 civil twilight. Left out: 45,944 too dark, 3,388 with fewer than 3 usable views, 3,753 unlabelled, 11 of another format; buoy 45001 (Lake Superior) has no `.spec` records. — [BUOY]
- Format: six views side by side, 2880 × 300 px (6 × 480); buoy 44008 is 2304 × 246 (6 × 384); a burned-in caption (station ID, UTC time) occupies the bottom 30 rows in both formats and is cut from every packaged strip. — [BUOY]
- Light rules: sun at most 6° below the horizon; at least 3 usable views (mean grey ≥ 20 and standard deviation ≥ 4). — [BUOY]
- Label rule (ADR 0010): mean of the records centred within 30 min of the strip (three records for 84,823 of 90,434 labelled strips); source files are 377 realtime snapshot files with 331,178 WVHT records at 79 stations, saved on the 1st and 15th of each month; these are realtime values, not NDBC's quality-controlled annual files, which appear the following year. — [BUOY]
- Class mix (WMO bands ≤ 0.5, ≤ 1.25, ≤ 2.5, > 2.5 m): smooth 3,405 (8.3 %), slight 10,713 (26.1 %), moderate 18,007 (43.8 %), rough or more 8,966 (21.8 %); H_s runs 0 to 13.1 m, median 1.6 m. — [BUOY]
- Extra labels and inputs added in later packages: window-mean wind-wave and swell heights (92 % of strips; the v1 wind-sea package has 37,652 strips at 77 buoys), the buoy's own wind (99.9 % of strips), each view's de-mounted horizon roll and pitch (`views.csv`); spectral package `ndbc_buoycam_spectral_v3cv` adds T_p (99.9 % of strips), Tm-1,0 (99.9 %), APD (100 %), wind-wave and swell peak periods (92 %, 91 %), wind-sea Tm-1,0 (94 %), mean direction (97 %) and the 46-bin wind-sea shape (94 %). — [BUOY]
- Horizon coverage: 209,615 of 236,973 usable views (88 %) give a clear horizon; 38,860 strips have three or more. — [BUOY]
- Measurement labels (measurements worktree): band-limited RMS slope `slope_rms` for all 41,091 strips except 50 all-zero spectra at buoy 46081 (mean 0.056, standard deviation 0.021), finite-depth slope using 79 station depths, four energy bands with edges 0.030, 0.095, 0.155, 0.255, 0.495 Hz, a directional-width target (37,841 strips meet the coverage rule), and three peak-period definitions (41,004 strips have a strict nearest peak). — [MEAS], [CHECKIN]
- Packages on disk: `ndbc_buoycam_sea_state_v0` (2.9 GB), `ndbc_buoycam_windsea_v1`, `ndbc_buoycam_heights_v2`, `..._v2r` (random split), `..._v2cv` (five folds), `ndbc_buoycam_spectral_v3cv`, `ndbc_buoycam_fresh_v2cv` (new weeks); all reuse v0's image bytes. — [BUOY], [FRESHPKG]
- Since the package: 7,850 strips archived at 70 stations from 2026-11-14 16:00 to 2026-11-19 11:10 UTC, of which 1,506 have a wave record within 30 min so far (labels wait for the next `.spec` snapshot). — [FRESH]
- NDBC inventory: 91 stations list a buoy camera, all 91 with historical directional spectra, 87 with realtime directional spectra; inventory covers 38,908 files and 1,797 stations. — [NDBC], [README]
- Quirks: NDBC deletes superseded images after about 72 h, so the project's archive is the only copy; about 1 % of camera-hours are never published; 638 camera-hours were lost for good on 2026-10-04/05; `.spec` and `.data_spec` stamp the same record 10 min apart. — [NDBC], [COLLECT]
- Quirks found during experiments: the caption carries no compass heading (a fixed "90°", "290°" at 44008), so no strip says which way its views face; the six views are most likely taken one after another (adjacent-view roll correlation 0.03-0.04); 44008's smaller format makes a linear probe extrapolate without bound (15-23 m predicted for 4-5 m seas). — [BUOY], [PRE]
- Licence: U.S. Government public domain. — [REG]

**WebCOOS stills, timex, brightest (shore cameras, North Carolina)**
- Historical transfer finished 2026-11-05: 241,229 stills (108 GB), 108,516 `timex` (54 GB) and 109,053 `brt` (99 GB), one per 30 min in daylight, eight cameras (currituck_hampton_inn, currituck_sailfish, jennette_north, jennette_south, oakisland_east, oakisland_west, masonboro_inlet, cocoabeach); `timex`/`brt` exist from 2024 on. — [WEBCOOS]
- `timex` is the time average and `brt` the per-pixel brightest value of one 10-min video clip (600 s at 20 fps); stills are frames of the same video at 2688 × 1520. No experiment in the repo uses `timex` or `brt` yet; the datasheet calls the comparison with stills "an experiment". — [WEBCOOS]
- Hourly collector since 2026-09-26 for the five Yin et al. (2025) cameras and, since 2026-10-08, eleven beachfront cameras within 25 km of a directional buoy (san_elijo, westport_north/south, masonboro_inlet, currituck_sailfish, crescent_city, oceancity, pgtalabama_south/west, stinson, vabeach_hamptonos). — [WEBCOOS]
- Currituck pairing (CDIP 433 / NDBC 44056, Duck FRF 17 m): 40,313 stills on disk (2022-01-04 to 2026-10-13), 38,361 paired on 1,609 days; H_s 0.33-3.1 m (1st-99th percentile), median 0.81 m, max 4.68 m; 2,465 stills at 2 m or more on 173 days and 414 at 3 m or more on 35 days. The experiment package uses 37,155 daylight/low-sun stills on 1,606 days. — [WEBCOOS], [CURR]
- Camera-to-buoy distance at Currituck: **9.15 km** from the EXIF GPS position (datasheet, 2026-10-13; newer). The older landscape review (2026-09-26) gives 21.6 km and the data-source scan 22.2 km, both from the WebCOOS API position, which the datasheet shows is 13.7 km wrong. — [WEBCOOS]; contradicted by [LAND], [SCAN] (older)
- Oak Island (Kamagata 2026's held-out test views) paired with CDIP 200 on 2026-11-18: oakisland_east 35,586 stills, 29,518 labelled; oakisland_west 37,571 stills, 31,465 labelled; buoy 28.9 km away (the older landscape review table says 24.5 km); CDIP 200's record ends 2026-07-20, so later images have no nearby buoy. — [WEBCOOS]; distance contradicted by [LAND] (older)
- Jennette's Pier south paired with CDIP 243 on 2026-11-18: 48,254 stills (2022-07 to 2026-11, 2592 × 1944), 47,721 labelled; buoy 18.6 km away in 21 m of water. — [WEBCOOS]
- A full-cadence sample of September 2026 (every still at two cameras, every `timex`/`brt` at seven) started 2026-11-10; the month has 3 calm and 4 rough days. — [WEBCOOS]
- Quirks: the Currituck camera has grown hazier every year (share of stills below the blurriest third of 2022-2024: 19 % in 2022, 93 % in 2026); Jennette's pointing changes month to month; Oak Island east was replaced in mid-2024 (2048 × 1536 to 2560 × 1440); 38 files served empty; people are visible on the beach in summer; no calibration is published for any camera; 2026 stills are daylight only. — [CURR], [VIDEO], [WEBCOOS]
- The label is H_s at a buoy 9-29 km away in 13-21 m of water: the incident sea, not the breakers the camera sees. — [VIDEO], [WEBCOOS]
- Usage rules: API token required; "download only a reasonable amount"; contact WebCOOS before large downloads of data older than 90 days (done; they directed bulk requests to their on-premise archive at one request at a time, 1 s wait, 20 MiB/s); acknowledge SECOORA/WebCOOS in publications; redistribution of images is not addressed, so ask before publishing images or releasing them in a dataset. — [WEBCOOS]

**WebCOOS MP4 video sample**
- Clips are public MP4s, about one every 10 min in daylight, back to at least 2024-03; a Currituck clip probed was H.264, 2688 × 1520, 20 fps; Jennette clips are about 290 MB each (2592 × 1944), Oak Island about 92 MB. — [WEBCOOS], [VIDEO]
- Sample design: Currituck, Jennette south, Oak Island east and west; one clip per 30-min CDIP record in daylight; test days first, then six sample weeks 2024-04 to 2026-01; the whole sample is estimated at about 4,800 clips (about 170 GB of cut frames). — [VIDEO]
- Counts over time: 401 clips packed when the whole-clip bars were fixed (2026-11-18); 453 clips on 12 local dates in the measurement report's audit; 536 clips on 12 dates in the measurement prep snapshot; **992 clips on 18 local dates in the interim scored run** (422 Currituck, 234 Jennette, 208 Oak Island west, 128 Oak Island east; 33,115 sixteen-second windows); the local package at 06:28 EST on 2026-11-19 lists **1,282 clips on 20 local dates** (433 Currituck, 388 Jennette, 251 Oak Island east, 210 Oak Island west; 2024-04-08 to 2026-09-24); the pairing summary at 08:04 EST lists 1,553 clips (1,526 usable) on 66 camera-days, H_s 0.39-4.12 m, median 0.87 m. — [VIDEO], [R-MEAS], [CHECKIN], [VIDPKG], [VIDSUM], [HANDOFF]
- Clip length in the 1,282-clip package: median 600 s, shortest 20.65 s, longest 607.8 s (counted from `clips.csv`); 914 of the 992 interim clips have at least 15 sixteen-second windows. The architecture report's "4-minute clips at Currituck" reflects one 260-s clip probed on 2026-11-18; the archive's full clips there are 600 s. — [VIDPKG], [VIDEO], [WEBCOOS]; contradicted by [R-ARCH] (older)
- Labels: H_s is the window mean of CDIP records within 30 min of the clip midpoint; T_p from the nearest record; Tm02 is also in the package. — [VIDEO], [VIDPKG]
- Restriction: the pace agreed with WebCOOS is one request at a time (a lock enforces it); putting raw clips on the GPU server would be a second download of about 0.6 TB, about three times the 200 GB named in the request, and "needs WebCOOS's consent first". — [VIDEO]

**PacIOOS Waimea Bay (2009-2013)**
- Two retired cameras (`beachcam_003` beach, `beachcam_004` offshore): 8,935 + 8,919 hourly daytime JPEGs, 2009-02-06 to 2013-10-15; 17,854 indexed, 16,404 paired with CDIP 106 (6.3 km, 200 m depth) on 800 days; H_s 0.73-4.5 m (1st-99th percentile), max 6.44 m; 299 images at 4 m or more on only 27 days. — [WAIMEA]
- Quirks: index time is not the capture time (burned-in time must be read and masked); 638 images have uncertain time; file names are in HST. Licence: free use and redistribution with acknowledgement of PacIOOS. Two Waikiki cameras (15,564 + 15,590 images) are listed but not paired. — [WAIMEA]

**CDIP Waveriders (shore-camera labels)**
- Stations downloaded and standardized: 106 (Waimea; 84,893 records in 2009-2013), 433 (Currituck; 78,909 records since 2022-01-03; current deployment is a DWR-M4 with 1800-s samples), 200 (Oak Island; 76,963 records, 2013 to 2026-07-20), 243 (Jennette). — [WAIMEA], [WEBCOOS], [CHANGELOG]
- H_s is spectral (4√m0), 1600-s samples every 30 min; flags follow 1 good / 2 not evaluated / 3 questionable / 4 bad; the `historic` file has no displacement, the per-deployment files do (`xyzZDisplacement`, about 1.28 Hz; station 106's 28 deployment files total 18.7 GB). Licence: "may be redistributed and used without restriction", with a per-station acknowledgement. — [CDIP]
- Displacement windows were matched for 511 of the 536 clips of the prep snapshot; every matched record carries provider flag 2 (unevaluated), so a strict flag-1 analysis has zero eligible records. — [CHECKIN]

**NDBC spectra and standard meteorology**
- Seven 45-day realtime files per camera buoy saved twice a month (`.txt`, `.spec`, `.data_spec`, `.swdir`, `.swdir2`, `.swr1`, `.swr2`); spectra use the 46-band WPM layout, 0.0325-0.485 Hz; the partition check used 295,547 records at the 79 camera buoys (2026-08-12 to 11-15). — [COLLECT], [INV], [BUOY]
- NDBC `WVHT` equals 4√m0 of the published spectrum to reporting precision (31,944 records at four stations); historical 2023 files support a buoy-pair study of 211 pairs. — [README]

**ERA5 (model baseline, never a label)**
- Hourly wave parameters on a 0.5° grid; held for the camera sites (Waimea 2009-2013, the NC sites, Currituck 2022-01 to 2026-10) and as global monthly fields for the buoys: total, wind-sea and swell heights plus periods and directions (`pp1d`, `mwp`, `mp2`, `mwd`, `mpww`, `mdww`, `mpts`) through 2026-11-13. — [ERA5], [BUOY], [CHANGELOG]
- Runs about 5 days behind real time; the last ~3 months are preliminary ERA5T; CC BY 4.0. At buoys the nearest sea grid point is a median 19 km away (max 44 km). ERA5 2-D spectra are not held (a one-station-day pilot request is prepared, not submitted). — [ERA5], [BUOY], [CHECKIN]

**WIS hindcast (model baseline)**
- 21 bulk files and 60 monthly files (358 MB) at the reference buoys of the camera sites; ends 2025-12-31, so it covers none of the 2026 collection; station 44056 has no 2022; files are marked "under evaluation"; free for scholarly use with acknowledgement. — [WIS]

**External open datasets (method checks, not training)**
- Schwendeman & Thomson 2017 ship stereo (CC BY 3.0 US, 24.8 GB): 55 five-minute bursts on 10 days (2014-12-28 to 2015-01-11) with elevation and radiance on a 161 × 161 grid at 0.25 m, 5 or 7.5 Hz, the cameras' IMU motion, 263 Waverider records (H_m0 1.05-7.19 m) and 2,522 SWIFT bursts; raw frames only on request. Prep found 612 minute files in 55 bursts; 53 qualify, and all 53 elevation spectra peak at the longest admitted wavelength (12 m) in the prepared 24 m crop, so it supplies zero interior wavelength peaks. — [EXT], [CHECKIN]
- Carini et al. Duck 2016 thermal IR + lidar (CC BY, 6.4 GB): 1,782 tracked waves from 53 runs on 7-8 November 2016, 1,674 with a manual type (1,113 non-breaking, 291 steady, 233 spilling onset, 37 plunging onset); 640 × 480 IR stacks; FRF conditions H_s 0.33-2.52 m. — [EXT]
- Alberello et al. 2017 WASS, S.A. Agulhas II (CC BY 4.0, 1.59 GB): six NetCDF files of stereo elevation on 4 July 2017, rough H_s 4.5-6.5 m; missing values are fill value 0 and `datenum` is wrong; no ship motion or wave instrument in the record. — [EXT], [SCAN]
- Yin et al. 2025 breaker clips: `Set_A.zip` (4.8 GB) with 9,000 verified clips from six sites is in the measurements worktree's prep folder; 32 Yin clips overlap two WebCOOS clips by camera and UTC interval; licence CC BY-NC-ND 4.0 ("derived datasets may not be redistributed"). — [CHECKIN], [HANDOFF], [REG]

**Registry candidates documented but not acquired**
- Verified access but nothing acquired: `usgs_coastcam` (adapter is a placeholder class), `usace_frf` (no Argus imagery on the public server), `noaa_ww3` (not downloaded), `socib_cala_millor` (hourly images since 2011, 1.5 km from an AWAC; bulk images need SOCIB's agreement; request drafted, not sent as of 2026-10-08). — [REG], [COASTCAM-SRC], [SCAN]
- Documented only: `ioos_erddap`, `sentinel1_ocn`, `hls`, `copernicus_marine_waves`, `cmems_insitu_waves`, `noaa_digital_coast`, `wass_stereo`, `guimaraes2020_stereo` (CC BY 4.0), `buscombe2020_owg`, `stringari2021_deepwaves`, `kwon2023_xband`, `cienfuegos2025_soliton`, `salin2023_ladoga`, `kolomverse`, `sofar_spotter_archive`, `esa_cci_sea_state`, `cfosat_swim`, `swot_karin_wave`. — [REG]
- Found in the 2026-10-08 scan, not in the registry and needing a request: Western Australia DoT coast cameras with DoT Waveriders, GLERL Great Lakes webcams, Cam-Era (New Zealand); checked and unavailable: Narrabeen Argus, Zandmotor Argus, FINO1 webcam, KHOA CCTV, Saildrone cameras, SWIFT drifter images, Fusion-Wave (infrared). — [SCAN]

### Inferences
- The registry entry for `yin2025_vwbt` (status "documented") lags the disk: the clips were downloaded for the measurement prep. The registry has no entry for the WebCOOS video sample as a separate product either.
- All instrument-labelled shore data are three North Carolina sites (four views) plus Waimea; all buoy-camera data are seven weeks of one autumn. Geographic diversity comes only from the buoys.
- The `timex`/`brt` archive (about 217,000 files) is the largest held input that no experiment has touched; it is a ready-made "minutes of sea in one image" modality paired to the same labels as the stills.

### Gaps
- Live archive totals (images per collection today) are on the dashboard, not in the docs; I did not recount the raw archive beyond the ledgers and summaries cited.
- I found no file recording that the September 2026 full-cadence sample or the video sample download has finished; the download lock file was last touched 2026-11-19 08:13 EST, which suggests the video sample was still running.
- Whether the SOCIB, DoT, GLERL, Cam-Era or raw-stereo-frame requests were ever sent is not recorded after 2026-10-08.
- WebCOOS's answer on what the acknowledgement should look like, and on whether derived datasets may include frames, is not recorded.

## Methods and models implemented: architectures, backbones, heads, fusion, baselines, splits, uncertainty, evaluation protocol

### Takeaway
Scored models are (a) a six-view ConvNeXt-Tiny strip model (classifier, then a three-height regressor with a view transformer, later with distribution, spectral and direction heads), (b) frozen V-JEPA 2 ViT-L / DINOv3 ViT-L video probes with whole-clip readouts and a label-free pixel spectrum, and (c) tabular fusion of camera outputs with ERA5 and buoy wind. Every comparison is pre-registered as a "bar" committed before the runs, judged by a paired bootstrap over whole buoys or dates, with three-seed averages. Reference re-implementations of all four target papers exist in the code but none has been trained on the project's data.

### Cited Findings

**Still-image models (buoy strips and Currituck)**
- Currituck v0: torchvision `convnext_tiny` (ImageNet), four-class head, class-weighted cross-entropy, AdamW with warm-up and cosine decay, 15 epochs; input is the sea band (rows 27-60 % of the frame, 900 × 168 px); no horizontal flips; one seed. — [CURR]
- Buoy v0/v1 `MultiViewNet`: one shared ConvNeXt-Tiny reads each of the six views at 270 × 480; the masked mean of usable views feeds a four-class head; 10 epochs, batch 32 strips. v0 used ±5° rotation augmentation, which v1 removed because it erased the horizon tilt. — [BUOY]
- Buoy v2 `MultiViewRegressor`: one ConvNeXt-Tiny per view, a one-layer transformer over the usable views, masked mean, three outputs (log total, wind-wave and swell height; Huber loss on standardised logs); no crop or rotation; epoch chosen on validation H_s RMSE; arm A images only, arm B adds each view's horizon roll and pitch; three seeds averaged in log space. — [BUOY]
- Backbones tried in that model: ImageNet ConvNeXt-Tiny (full fine-tune), DINOv3-distilled ConvNeXt-Tiny (`timm:convnext_tiny.dinov3_lvd1689m`), DINOv3 ViT-B/16 (86 M parameters, native 270 × 480, patch tokens averaged), each frozen, fully fine-tuned or with the "gentle" recipe (backbone frozen 2 epochs, then learning at a tenth of the head's rate: 3e-5 against 3e-4). — [BUOY]
- Stage-1 uncertainty: a distribution head (mean and variance of each log height) trained with beta-NLL, beta = 0.5, added to the Huber loss; three seeds merged into one log-normal; CRPS; buoy-level split-conformal 90 % intervals calibrated on the other folds' buoys. Evidential regression was deliberately not used. — [BUOY]
- Stage-3 `MultiViewSpectral`: stage 1's model plus period targets, a 46-bin softmax wind-sea shape head (cross-entropy), and a direction head predicting cos/sin of (wave direction − sun azimuth), with a learned view-position embedding used only by the direction head. — [BUOY]
- Horizon extraction: steepest bright-to-dark step per column, RANSAC line, each camera's median angle and height removed. — [BUOY]

**Video models (shore)**
- Arms on identical pixels: `video` = frozen V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`) on 64 frames at 4 fps (16 s); `still` = the middle frame repeated 64 times; `shuffled` = the 64 frames in a fixed random order; `image` = frozen DINOv3 ViT-L/16 on the middle frame. Frames are 256 px high, fed as two squares (left and right edge) because the cameras do not hold a fixed view, so no surf-zone crop. — [VIDEO]
- Heads: an attentive probe (one self-attention block, learned query, d = 256) on tokens of four layers (50, 67, 83, 100 % depth), plus an MLP on mean tokens; targets log H_s and log T_p; Huber loss; 60 epochs; 5 folds of whole local dates (all cameras of a date together), 5 seeds. — [VIDEO]
- Whole-clip reading: K = floor(duration / 16 s) non-overlapping windows (37 for a 600-s clip); `clip` readout is the energy mean sqrt(mean(H_s²)) for H_s and the geometric mean for T_p; a learned record pooler (`RecordPooler`) and a DINOv3 `clip-image` control are reported beside it. — [VIDEO]
- Pixel spectrum recipe v1 (label-free): 4 × 4-px block brightness series, Welch spectra at 4 Hz with 128-s segments, water blocks chosen as the top 20 % by wave-band share of the brightness-rate spectrum, 10 scalar features (peak period above a power-law background, T_m01, T_m02, five band shares, band power, rate share) into a ridge head. — [VIDEO]

**Fusion schemes**
- Currituck and buoy v1: multinomial logistic combiner of camera class log-probabilities and log hindcast heights, compared against the same combiner on the hindcast alone. — [CURR], [BUOY]
- Test 3: ridge regression (alpha 1.0) on the camera's three heights, ERA5's three heights and buoy wind speed and gust, fitted on validation buoys. — [BUOY]
- Stage 2 gated partition swap: H_s² = T² + k·g·(b·C² − W²), where T and W are the ERA5 + wind ridge's total and wind sea, C the camera's wind-wave height, g a learned gate on five visibility features (camera spread, sun elevation, usable views, mean grey, mean contrast), b a camera scale and k the share reaching the total; fitted fold by fold on out-of-fold camera heights. Beside it: full swap, raw partition rule, Test 3's ridge, gradient boosting. TabPFN was not run because its weights need a non-commercial licence accepted. — [BUOY]
- The energy rule is physical: WVHT² = SwH² + WWH² holds within rounding on all 295,547 records at the camera buoys. — [BUOY]

**Baselines**
- No-image baselines used: majority class, monthly climatology, each buoy's own majority class, persistence (Waimea), ERA5 at the nearest sea point with a line fitted on training data, WIS at the buoy, ERA5 + WIS, the buoy's wind through a power law, ERA5 + wind ridge, a sun/location/time-only direction control, ridge on image block statistics (Waimea). — [WAIMEA], [CURR], [BUOY], [HANDOFF]
- Reference re-implementations of the four papers exist as code: `Choi2020Regressor` (frame differences → VGG-16 → BiConvLSTM), `Kim2023SnapshotCNN` and `Kim2023CNNBiLSTM` (GoogLeNet → BiLSTM 2000), `Yang2026Fusion` (ResNet-18 + attitude MLP + self-attention), `SlowFastWaveHead` (Kamagata), plus `kamagata_physics_loss` and a project `steepness_band_loss`. The methodology page states they are "for controlled replication and ablation on this project's data and splits". A grep of `scripts/`, `configs/` and `deploy/` in main and all six worktrees (run 2026-11-19) finds no training script or config that uses these classes; only the unit tests do. No experiment doc records a scored run of any of them. — [MODELS], [GIT]
- Comparison with the papers is done by computing each paper's measure from the project's own predictions (`scripts/compare_papers.py`): Kamagata's RMSE/R²/Pearson and regression slope; Choi's calm (< 1.52 m) vs rough (> 3.05 m) with predicted H_s split at 2.285 m; Kim's five 0.5-m classes on 0.631 × H_s. — [BUOY]

**Splits**
- Buoy station split: buoys within 300 km of each other (chained) form one of 43 groups; 55 train / 12 val / 12 test buoys (28,597 / 6,311 / 6,183 strips); the closest test-train pair is 322 km apart; a time split with a 72-h embargo also exists. — [BUOY]; the split comment in the v0/v1/v2r configs says 44 groups ([V0CFG]), the experiment doc and the CV fold builder say 43 (the doc is newer)
- Grouped CV: the 43 groups in five folds balanced by strips; the US West Coast group (18 buoys with kept strips, 9,440 strips) is fold 0 alone; each fold's validation groups are drawn from its own training folds to match the training swell share. — [BUOY]
- Validation/test mismatch on the sealed split: validation buoys carry swell share 0.40, test buoys 0.67, training 0.62. — [BUOY]
- Shore: Currituck day split and year split (train 2022-2024, test 2025, later test 2026); Waimea day / week / year; video folds of whole local dates, with 18 dates declared development data and all later dates confirmation data. — [CURR], [WAIMEA], [VIDEO]

**Evaluation protocol ("bars fixed before scoring")**
- Each bar is written in the experiment doc and committed before its runs (commit subjects such as "Fix bar S3 before any run", "Fix bar M-S1 before any run"); a bar typically needs two targets or two metrics to improve with each 95 % interval excluding zero under a paired bootstrap over whole buoys (1,000 draws), station groups, buoy-days or local dates (2,000 resamples). — [BUOY], [VIDEO], [MEAS], [GIT]
- Standing rules: three seeds averaged because one seed moved v0's test macro-F1 by 0.12; the sealed 12-buoy numbers stay the headline and the folds are "the protocol for choosing and comparing"; results reported beside a bar count as "found, not confirmed"; after Tests 2, 2b and 3 used the 12 test buoys, confirmation must come from strips archived after the package; "no further variant is tried on this test set". — [BUOY]
- Required reporting set: named NRMSE normaliser, regression slope, circular direction metrics, ordinal class metrics, block-bootstrap intervals, strata, baselines on the same split. — [EVAL]

### Inferences
- "Improving on the four papers" has so far been tested by matching each paper's task and metric on the project's own model, plus one protocol replication (the random split). The head-to-head step, running Choi's, Kim's, Yang's and Kamagata's own architectures on the project's data and splits, has not been done even though the code is ready.
- Every fusion so far is late and tabular (camera outputs as numbers). No model fuses image features with ERA5, wind or spectra inside the network, and no model takes more than one modality of imagery (still + timex + video) at once.

### Gaps
- Hyperparameters of the stage-3 and slope fine-tune configs were read only at summary level.
- I did not verify by reading `compare_papers.py` in full that Kim's and Choi's rules are implemented exactly as the doc states.

## Results: every scored experiment with numbers, intervals and verdict, and where the project stands against each of the four papers

### Takeaway
At 12 never-seen buoys the best strip model (DINOv3 ConvNeXt-Tiny, gentle recipe) reaches H_s RMSE 0.440 m, R² 0.82, Pearson 0.921: ahead of Kamagata 2026's numbers and of Choi 2020's calm/rough accuracy (99.2 % vs 84.05 %), behind Kim 2023's snapshot accuracy (70.2 % vs 78.3 %), and Yang 2026's attitude gain does not carry over to real buoys. The consistent physical finding is that the camera is a wind-sea instrument: it beats ERA5 on wind-wave height and wind-sea period, loses to ERA5 on total height, peak period and direction, and adds to ERA5 + wind only when fused as wind-sea energy (bar S2, met out of fold, awaiting confirmation). On shore video, motion inside 16 s added nothing for H_s and no learned probe beat climatology for T_p in the interim run.

### Cited Findings

**Table of scored experiments** (RMSE in metres unless stated; intervals are 95 %; "not met" means the pre-set bar failed)

| Experiment (date) | What was compared | Numbers | Verdict | Source |
|---|---|---|---|---|
| Waimea benchmark v0 (2026-10-08) | ridge on image block statistics vs corrected ERA5, day / week / year splits | image ridge 0.40 [0.35, 0.46] / 0.43 / 0.36; ERA5 corrected 0.30 / 0.33 / 0.23; ERA5 + image − ERA5: −0.041 [−0.064, −0.017] (day), −0.028 (week), +0.012 [−0.004, +0.027] (year) | image alone does not beat ERA5; adds on held-out days, not on the held-out year | [WAIMEA] |
| Waimea benchmark v1 (2026-10-08) | adds WIS at the buoy | ERA5 + WIS 0.24 [0.21, 0.28] / 0.27 / 0.23; + image − (ERA5 + WIS): −0.012 [−0.025, −0.000] (day), −0.012 [−0.025, +0.002] (week), +0.007 (year) | image adds almost nothing once WIS is there; no learned image model yet | [WAIMEA] |
| Currituck sea state v0, camera alone (2026-11-14, one seed) | 4 WMO classes, held-out year | 2025: balanced accuracy 0.73 [0.69, 0.77], macro-F1 0.73 [0.68, 0.77] vs WIS 0.77 / 0.80 and ERA5 0.73 / 0.76; 2026: 0.71 / 0.71 vs ERA5 0.75 / 0.77; held-out days 0.78 / 0.78 | bar (beat WIS on both) not met | [CURR] |
| Currituck camera + hindcast (2026-11-14) | logistic combiner vs hindcast alone refitted | macro-F1 gain +0.038 [+0.007, +0.065] (WIS, 2025), +0.057 [+0.022, +0.092] (ERA5, 2025), +0.052 [+0.023, +0.079] (ERA5, 2026); balanced-accuracy gains about +0.02, intervals include zero | camera adds a modest, consistent amount; WIS's bar still not cleared (macro-F1 0.77 vs 0.80) | [CURR] |
| Currituck full-frame ablation (2026-11-14) | whole frame vs sea band | 2025 balanced accuracy −0.068 [−0.100, −0.040]; level on 2026 | sea band stays the input | [CURR] |
| Buoy v0, station split (2026-11-17) | camera vs ERA5 at 12 unseen buoys, 5,642 strips | camera 0.701 / 0.648 vs ERA5 0.735 / 0.771; gain −0.034 [−0.154, +0.013] / −0.123 [−0.385, −0.026]; seed 1: 0.674 / 0.534 | not met | [BUOY] |
| Buoy v0, time split (2026-11-18) | same buoys, later weeks | vs ERA5 on the 41 % of strips covered: 0.691 / 0.696 vs 0.784 / 0.798; vs own-buoy majority: +0.32 [+0.29, +0.36] / +0.36 [+0.32, +0.40] | not met vs ERA5; met vs majority | [BUOY] |
| Swell check Q-M8 (2026-11-18, exploratory) | too-calm share by swell third | within buoy and class +0.20 [+0.15, +0.25] from least to most swell (ERA5 flat); per-buoy Spearman −0.89 over 24 held-out buoys | camera sees the wind sea, not the swell | [BUOY] |
| Horizon motion Q-M5 (2026-11-18, exploratory) | horizon roll/pitch vs the buoy's spectral slope | within-buoy Spearman 0.56 [0.54, 0.58], positive at 75 of 76 buoys (roll) and 76 of 76 (pitch); 0.56 with wind-wave height, 0.05 with swell | the horizon is a motion sensor for the wind sea only | [BUOY] |
| Buoy v1 Test 1 (2026-11-18) | camera wind-sea class vs ERA5 wind-wave height, 6,024 strips | 0.795 / 0.807 vs 0.694 / 0.709; gain +0.101 [+0.053, +0.159] / +0.098 [+0.060, +0.151]; buoy-wind control 0.486 / 0.468 | **met** | [BUOY] |
| Buoy v1 Test 2 (2026-11-18) | camera + ERA5 vs ERA5 for total sea state | +0.056 [+0.034, +0.095] balanced accuracy, −0.012 [−0.025, +0.045] macro-F1 | not met | [BUOY] |
| Buoy v1 Test 2b (2026-11-18) | same, no class weights | +0.053 [−0.007, +0.082] / +0.044 [−0.024, +0.060] | not met; gain sits in calm seas at two test buoys | [BUOY] |
| Buoy v2 arm A (2026-11-18) | heights in metres, 6,183 strips at 12 unseen buoys | H_s 0.450 [0.390, 0.509], R² 0.81 [0.73, 0.86], Pearson 0.915 [0.890, 0.933], bias −0.16 m, regression slope 0.83; wind-wave 0.219 (R² 0.93); swell 0.597 (R² 0.54); Choi task 99.1 % [97.9, 99.7] (3,190 strips); Kim classes 69.4 % [61.8, 77.9] (4,791 strips) | reference model | [BUOY], [RUN-V2] |
| Bar V2-1 (2026-11-18) | arm B (+ horizon attitude) − arm A | H_s +0.038 [+0.018, +0.057]; wind-wave +0.005 [−0.001, +0.010] | not met (attitude makes H_s worse) | [BUOY] |
| Bar V2-2 (2026-11-18) | camera vs ERA5 in metres | wind-wave 0.215 vs 0.341 (−0.126 [−0.187, −0.065]); total 0.434 vs 0.267 (+0.167 [+0.091, +0.242]); swell 0.582 vs 0.361 | camera wins the wind sea, ERA5 the total | [BUOY] |
| Test 3 (2026-11-18) | camera + ERA5 + wind vs ERA5 + wind, 6,045 test strips | H_s 0.258 vs 0.259 (−0.001 [−0.052, +0.049]); wind-wave 0.199 vs 0.286 (−0.087 [−0.114, −0.060]) | not met (needed both) | [BUOY] |
| Bar B1 (2026-11-18) | DINOv3 ConvNeXt-Tiny, full fine-tune − arm A | H_s 0.489 vs 0.450 (+0.039 [+0.008, +0.073]); wind-wave +0.025 [+0.011, +0.039] | not met (worse with this recipe) | [BUOY] |
| Bar B2 (2026-11-18) | DINOv3-gentle − arm A | H_s **0.440** [0.381, 0.501], R² 0.82, Pearson 0.921, slope 0.86: −0.010 [−0.024, +0.001]; wind-wave 0.214: −0.004 [−0.010, +0.001]; ImageNet-gentle 0.524 (+0.075 [+0.052, +0.095]); Kim 70.2 % [62.7, 78.7]; Choi 99.2 % | not met by a hair; best model so far | [BUOY], [RUN-B2] |
| Bar B3 (2026-11-18) | DINOv3 ViT-B/16 − arm A | gentle 0.455 (+0.005 [−0.009, +0.020]); frozen 0.560 (+0.111 [+0.086, +0.139]) | not met | [BUOY] |
| Protocol check R (2026-11-18) | arm A on a random 80/10/10 split (Choi/Kim protocol), 4,109 test strips | H_s 0.433 [0.378, 0.497], R² 0.87, Pearson 0.933; Kim 68.8 % [65.7, 71.8]; Choi 99.2 % | not a bar; "the gap to Kim's 78 % is not the split" | [BUOY] |
| Grouped CV, bar B2-CV (2026-11-19, 30 runs) | out of fold, 79 buoys, 41,091 strips | DINOv3-gentle H_s 0.532 [0.488, 0.575], R² 0.79, Pearson 0.892, CRPS 0.279 vs arm A 0.550 [0.504, 0.597], CRPS 0.310; difference −0.019 [−0.028, −0.011] (groups [−0.027, −0.005]); wind-wave 0.260 vs 0.262: −0.002 [−0.008, +0.004]; by fold 0.70 / 0.53 / 0.46 / 0.42 / 0.46; Kim 66.7 % [63.9, 69.6]; Choi 98.4 % | not met (needed both); DINOv3-gentle chosen for stages 2-3 | [BUOY], [RUN-S1] |
| Conformal intervals (2026-11-19) | 90 % buoy-level split conformal, DINOv3-gentle | H_s coverage 0.904, mean width 1.83 m (head alone 0.75); 11 of 79 buoys below 0.80; wind-wave 0.916, width 0.72 m | calibrated on average, not per buoy | [BUOY], [RUN-S1] |
| Camera vs ERA5 out of fold (2026-11-19) | all buoys | wind-wave 0.260 vs 0.343 (camera better by 0.087 [0.067, 0.110]); H_s 0.514 vs 0.292 on 40,240 strips (worse by 0.22 [0.18, 0.26]) | wind-sea finding holds at all buoys | [BUOY], [RUN-S1] |
| Stage 2, bar S2 (2026-11-19) | gated partition swap vs ERA5 + wind, 39,767 strips at 77 buoys | H_s 0.267 vs 0.288: −0.021 [−0.031, −0.012] (groups [−0.034, −0.009]); wind-wave 0.240 vs 0.293: −0.053 [−0.067, −0.039]; gate mean 0.79; gradient boosting best for wind sea (0.226) and worst for total (0.297) | **met out of fold**; confirmation on new weeks pending | [BUOY], [RUN-S2] |
| Stage 3, bar S3 (2026-11-19, 15 runs) | camera vs ERA5 out of fold | P1 peak period 3.44 vs 1.70 s (+1.75 [+1.41, +2.07]); P2 energy period 2.40 vs 0.67 s (+1.73 [+1.49, +1.97]); P3 mean direction 75.8° vs 13.8° (+62.0 [+58.5, +65.6]); wind-wave peak period 0.69 vs 0.93 s (−0.24 [−0.27, −0.21]); wind-sea Tm-1,0 0.55 vs 0.75 s (−0.21 [−0.24, −0.17]); shape Hellinger 0.30 vs 0.48 for the mean training shape; joint model H_s 0.618 vs stage 1's 0.532 (+0.086 [+0.068, +0.110]) | not met on P1-P3 (expected); wind-sea period win "found, not confirmed"; joint training harms H_s | [BUOY], [RUN-S3] |
| Shore video interim, M1 (2026-11-19) | `video` − `still`, H_s, 992 clips, 18 dates | 0.533 vs 0.535: −0.002 [−0.079, +0.075]; `shuffled` same; DINOv3 image 0.546; climatology 0.740 | not met | [VIDEO] |
| Shore video interim, M2 | `video` − `still`, T_p | 2.86 vs 2.80 s: +0.06 [−0.03, +0.20]; climatology 2.72 s | not met | [VIDEO] |
| Shore video interim, M3 | `clip` − `window`, H_s | 0.513 vs 0.524: −0.011 [−0.018, −0.004] (R² 0.47 to 0.49) | **met** (2 % gain) | [VIDEO] |
| Shore video interim, M4 | `clip` − `window`, T_p | 2.86 vs 2.88 s: −0.022 [−0.036, −0.006] | met "but worth nothing" (no probe beats climatology) | [VIDEO] |
| Shore video interim, M5 | `clip` + pixel spectrum − `clip`, T_p | 2.47 vs 2.86 s: −0.39 [−0.74, +0.02]; on clips with K ≥ 15: −0.45 [−0.81, −0.03]; spectrum alone 2.46 s (R² 0.13, Pearson 0.40) | not met on all clips | [VIDEO] |
| Shore video interim, beside the bars | learned record pooler; Kim's classes; Oak Island | pooler 0.584 (+0.071 [+0.001, +0.131] vs `clip`), dropped; Kim's five classes 58.4 % vs 58.8 % for the majority class; Oak Island (336 clips) H_s 0.661, R² 0.28, Pearson 0.55, T_p 2.79 s | no skill on Kim's task at unseen dates | [VIDEO] |
| Continued DINOv3 pretraining, checkpoint probe (2026-11-19) | ridge probe at 11 validation buoys | step 7,500: H_s 0.582 vs 0.642 off the shelf; wind-wave 0.344 vs 0.371; feature spread 0.150 vs 0.091 | checkpoint chosen; **bar P1 not yet scored** | [PRE] |
| Slope pilot (2026-11-18, exploratory, CPU) | out-of-fold RMSE of buoy RMS slope, 26,328 training strips, 53 buoys, 25 groups | training mean 0.02133; two horizon features 0.01733; linear wind 0.01162; quadratic wind + horizon 0.01131; geometry adds −0.00035 [−0.00050, −0.00018] (about 3.0 %) | wind alone is a strong comparator; texture's contribution unresolved | [R-MEAS] |
| Peak-label sensitivity (2026-11-19 morning check) | S3's saved predictions rescored against the nearest-record peak, 40,726 common strips | camera 3.57 s vs ERA5 1.95 s (registered mean-of-peaks label: 3.44 vs 1.70 s) | ordering unchanged by the label definition | [HANDOFF], [RUN-S3] |
| Direction control (2026-11-19 morning check) | sun position + location + time only vs camera, 40,038 strips | control 61.64° vs camera 75.83° mean absolute error | a no-image control beats the camera's direction head | [HANDOFF] |

**Standing against the four papers** (from the strips-grouped-cv worktree's "Where we stand (2026-11-19)")
- Kamagata et al. 2026 (H_s RMSE 0.586 m, R² 0.107, Pearson 0.436, regression slope 0.12, five held-out videos, labels untraceable): ours 0.450 m [0.39, 0.51], R² 0.81, Pearson 0.915 (arm A), 0.440 m / 0.82 / 0.921 (DINOv3-gentle), at 12 unseen buoys: "ahead on every measure; theirs is shore video, the Oak Island test is the like-for-like one". On the like-for-like Oak Island clips the interim video run gives H_s 0.661 m, R² 0.28, Pearson 0.55 and T_p 2.79 s against Kamagata's 0.586 m / 0.107 / 0.436 and 1.42 s; the doc calls this comparison "loose". — [BT], [VIDEO], [P-KAMA]
- Choi et al. 2020 (84.05 % ± 0.14 calm vs rough, same buoys and days in train and test; ship video RMSE 0.133 m): ours 99.1 % [97.9, 99.7] at unseen buoys (99.2 % DINOv3-gentle; 98.4 % out of fold at all 79 buoys): "ahead, on a harder split". No ship data, so nothing to set beside the ship-video result. — [BT], [BUOY]
- Kim et al. 2023 (78.3 % snapshot, 93.2 % 180-s video, five 0.5-m classes of mean wave height, random frames from 30 one-hour sessions at one camera): ours 69.4 % [61.8, 77.9] (arm A), 70.2 % [62.7, 78.7] (DINOv3-gentle), 68.8 % [65.7, 71.8] on a random split, 66.7 % out of fold: "**behind** on snapshots"; "the split is not the gap; their single site and session labels are". On shore video at unseen dates the same five classes score 58.4 % against 58.8 % for the majority class. — [BT], [VIDEO]
- Yang et al. 2026 (simulation; attitude cuts RMSE 0.098 → 0.073 m; attention fusion 0.032 m): on real buoys the horizon attitude makes H_s worse (+0.038 m [+0.018, +0.057]): "their gain does not carry to real buoys". No real camera + IMU data exist in the project to test the claim on a ship. — [BT], [BUOY]

**Where documents disagree (the newer one is named)**
- Kamagata's numbers: the landscape review's 2026-11-18 update quotes "Pearson 0.451, RMSE 0.562 m" from the arXiv abstract; the per-paper note (full text) shows those are full-set metrics that include training data, and the held-out values are 0.586 m / R² 0.107 / Pearson 0.436. The held-out values are the ones used in every comparison. — [LAND], [P-KAMA]
- Verification level of Choi, Kim and Yang: the literature matrix and the header of each paper note still say V2/V3; benchmark_targets.md (2026-11-18, newer) says all four are V1 after the PDFs were supplied, and each paper note has a "Full text read (V1, 2026-11-18)" section. The matrix also gives Kim's data as "~1 h/day, several days"; the full-text note says 30 days, one hour a day. — [MATRIX], [BT], [P-KIM]
- Buoy counts in the backbone report ("67 training buoys", "about 246k views") against the experiment doc (55 training + 12 validation buoys; 236,973 usable views; 165,021 views of the 55 training buoys used for pretraining). The experiment docs are authoritative. — [R-BACK], [BUOY], [PRE]
- Wind-wave RMSE of arm A appears as 0.219 m (all 6,183 test strips), 0.215 m (strips ERA5 covers, bar V2-2) and 0.220 m (Test 3's subset): different strip subsets, not conflicting results. — [BUOY]
- main's "Where we stand" table is dated 2026-11-18 and lacks the out-of-fold, stage-2 and stage-3 rows; the strips-grouped-cv worktree's version (2026-11-19) is newer. — [BT-MAIN], [BT]

### Inferences
- The only bars met are those where the camera is asked for the wind sea (Test 1, S2) or for more looks at the same record (M3, M4); every bar asking the camera for total height, swell-driven period, direction or a better encoder failed or was level.
- "Beating Kim" in his own terms would need a single-site, session-labelled, random-frame protocol; the project's strict protocols on both buoys and shore video put his task at or near chance, so the honest claim is a protocol finding, not a higher number.
- The Kamagata comparison is ahead on buoy cameras but currently behind on RMSE (0.661 vs 0.586 m) and far behind on T_p on the like-for-like Oak Island views, though with much higher R² and Pearson and instrument labels.

### Gaps
- The peak-label sensitivity and direction-control numbers are recorded only in the handoff doc; I found no saved output file for them.
- The v1 time split and the v0 time split against ERA5 on all test days are unscored (they wait for the ERA5 download after 2026-11-20).
- Per-camera and per-arm video tables live in each run's `report.md` on the video branch's run folders; I read the doc summary, not those files.
- No scored result exists for `timex`/`brt` inputs, for the Waimea learned image model, or for any external dataset (Duck, Yin, stereo).

## In flight: what is running or prepared but not yet scored on each branch or worktree, and what each waits on

### Takeaway
Nothing from 2026-11-18/19 is merged to main: four feature branches hold the newest results. Pending scored work is (1) bar P1 on the continued-pretraining weights, (2) the S2 confirmation on new weeks (set for 2026-11-27, waiting on ERA5), (3) the final shore-video run (waiting on the sample download) and uncommitted physics-v2 features, and (4) the buoy slope probe M-S1 and the other prepared measurement experiments (waiting on a review with the user and on human annotation).

### Cited Findings
- **main** (`1dfcc5f29`): one commit ahead of origin/main; uncommitted: a modified `bibliography.bib`, the untracked third report and its audit script. Its experiment docs stop at B3 and protocol check R, and its video doc is the design only. — [GIT], [BUOY-MAIN]
- **strips-grouped-cv**: B2-CV, S2 and S3 are scored and written up; open items in its status list: stage 2 confirmation, the time split against ERA5 on all test days and the v1 time split (both after an ERA5 re-download "after 2026-11-20"), and relabelling from NDBC's quality-controlled annual files when they appear. — [BUOY]
- **trusting-lalande**: the confirmation tool `scripts/buoycam_fusion_confirm.py` is built and checked (it reproduces the out-of-fold result to within 10⁻⁷ m on package strips; CPU float32 predictions differ from the GPU bf16 ones by 0.4 % on average for H_s). A dry run found 584 fresh strips at 69 buoys (2026-11-14 16:00 to 11-15 12:10 UTC) with 0 scorable because ERA5 ended 2026-11-13 01:00. The scored run is set for 2026-11-27, when ERA5 should reach 11-22; the rule is ERA5 coverage to 2026-11-21 16:00 UTC, about 6,000 strips at about 69 buoys expected. (The earlier stage-2 text says "about 2026-11-26"; the 2026-11-19 protocol's 11-27 is newer.) — [BUOY], [RUN-DRY], [FRESHPKG]
- **video-whole-clip**: the interim run is scored; "the download goes on; the final run uses the same code and config". The 18 interim dates are fixed as development data; every later date is a confirmation date whose labels stay unopened until a new feature's recipe and bars are committed. Uncommitted in the worktree: 199 added lines in `pixel_spectrum.py` implementing "Recipe v2 (bars M6-M7 ... 'Physics features v2')": common-mode removal, a rectified-rate spectrum, coherence weighting (cBathy-style cross-spectral eigenvalue share) and breaking features (foam fraction, onset rate, foam band width, foam persistence). The committed doc does not yet contain M6-M7. — [VIDEO], [PIXWIP]
- **dinov3-continued-pretraining**: the 30,000-step FINO run finished at 08:35 UTC on 2026-11-19 (about 9.1 GPU-hours; killed once at its 160 GB memory limit and resumed; data-loader workers grew about 11.5 GB per 1,000 steps, cause not found). Step 7,500 was chosen on 11 validation buoys. Status list: "[ ] P1 and the gentle comparison". The handoff's cluster snapshot at 8:12 a.m. Eastern says three project containers were running the continued encoder's frozen height evaluations. — [PRE], [HANDOFF]
- Bar P1 as fixed: the continued frozen arm must lower test RMSE of H_s and wind-wave height against B3's frozen arm (0.560 m and 0.257 m), each interval below zero; follow-up (distil into ConvNeXt-Tiny, re-check under grouped CV with fold-wise pretraining) only if the H_s gain exceeds the seed spread (at least 0.033 m). Caveat recorded: weights pretrained on the 55 training buoys are not clean under the grouped folds. — [PRE]
- **measurements** (`claude/wave-measurements-review` on top of `codex/wave-measurement-experiments`): bar M-S1 is fixed (frozen DINOv3-gentle features + wind + horizon geometry must lower out-of-fold slope RMSE against wind + geometry, interval over the 43 station groups below zero); the probe has not run. "Production slope runs are pending"; "the user asked to review the overnight runs together before launching the next set". Estimated cost: 3-5 hours on one H200 for five frozen folds, 3½-6¼ hours for the optional ten-epoch fine-tune. The fine-tune runs only if M-S1 or its validation-fit sensitivity is met. — [MEAS], [HANDOFF], [SLOPECFG]
- Other prepared, unscored measurement experiments: crest timing (eight clocked clips at three sites await independent reader timestamps); visible foam (25 buoy views await water/foam masks); Duck IR masks vs lidar height and period (1,782 waves, two day-holdouts); Yin breaker type (frozen R3D-18 vs frame ResNet-18 + GRU, six sites); ship stereo (period and elevation RMS only, peak wavelength deferred); shore Tm02 and CDIP-displacement mean height (sensitivity only, flag-2 records); ERA5 band correction (a station-day 2-D spectrum request for station 41004 prepared, not submitted). The prep "passed 329 unit tests" and CPU smoke runs. — [HANDOFF], [CHECKIN]
- The handoff notes that the expanded 1,282-clip package's CDIP and Yin overlap files still describe the 536-clip snapshot and must be refreshed before use. — [HANDOFF]

### Inferences
- The handoff names `codex/wave-measurement-experiments` as the working branch, but the worktree now sits on `claude/wave-measurements-review`, one review commit later; a reader following the handoff literally would miss the M-S1 bar and the review fixes.
- Three independent threads (P1, S2 confirmation, final video run) all resolve within roughly a week of 2026-11-19, so the "internal baseline" numbers for pretraining value, fusion confirmation and video will move soon.

### Gaps
- I could not see the GPU server, so whether the P1 containers have finished is unknown; no P1 result exists in any local file.
- No local file states how many of the roughly 4,800 sample clips remain to download.
- Whether bars M6-M7 have been drafted anywhere other than the uncommitted code comments is not confirmed.

## What the three earlier reports recommended, and which items are done, in progress, rejected or untouched

### Takeaway
The backbone report (2026-11-18) and the architecture report (2026-11-18) have been largely executed within two days: the backbone swap, gentle recipe, ViT-B test, grouped CV with uncertainty, partition fusion, spectral heads, whole-clip pooling, the pixel spectrum and one continued-pretraining pair are all run or running. Untouched are the frozen-probe ladder with DINOv2/SigLIP 2/LoRA, the Kamagata-recipe re-run, held-out-camera video tests, larger or alternative video encoders, video-domain pretraining, and whitecap features. The measurements report (third) is prepared but unscored except for its CPU checks.

### Cited Findings

**Report 1: "Vision backbones for wave cameras" (DINOv3 for strips, V-JEPA 2 for surf)**
- Recommended DINOv3 (ViT-B/16 as working model) frozen with an attentive probe over patch tokens, then LP-FT or LoRA; expected "modest gains" because published backbone swaps moved scores 0.1-2 points. — [R-BACK]
- Done: the like-for-like weights swap (B1, not met), LP-FT with a low backbone rate (B2, not met by a hair, best model), DINOv3 ViT-B/16 frozen and gentle (B3, not met; frozen +0.111 m), five-fold CV grouped by buoy (B2-CV), paired bootstrap over buoys, regression slope reported (0.83-0.86 against Kamagata's 0.12). — [BUOY], [RUN-B2]
- Revised by the second report: "frozen-probe ladders drop in value" and "the strip model stays ConvNeXt-Tiny" after B3. — [R-ARCH]
- Untouched (no mention in any experiment doc, config or script): the frozen probe ladder P1-P5 as designed (ridge / MLP / attentive probe over 6 × N patch tokens), the DINOv2-reg4 control arm, the SigLIP 2 texture arm, LoRA (rank 8-16), DINOv3 ViT-L, DINOv3-SAT, 1.5-2× input upsampling, five seeds for fine-tuned arms (three are used), whitecap fraction as a late-fused scalar. — [R-BACK], [BUOY]
- Horizon tilt as a late-fused channel: done as arm B, hurt H_s (V2-1). — [BUOY]
- Video recommendations adopted: frozen V-JEPA 2 ViT-L, 64 frames at 4 fps, multilevel attentive probe, shuffled and single-frame controls, a per-frame DINOv3 arm, record-level pooling, pixel-spectrum scalars alone and fused, Kim's five classes, Kamagata's measures on Oak Island. — [VIDEO]
- Video recommendations rejected or changed: the fixed surf-zone crop (rejected: Jennette re-points monthly and Oak Island east was replaced, so whole frames are fed as two squares). — [VIDEO]
- Video recommendations untouched: holding out whole cameras (folds are whole dates with all cameras together), re-running Kamagata's own recipe (16 frames at 10 fps, saliency crops, ImageNet ViT-S and V-JEPA 2-L), the two-rate "SlowFast" arm, the 2-fps ablation, ViT-g-384, V-JEPA 2.1, a VideoMAE V2-g or 4DS contrast, LoRA, a per-site metric scale input, timestack crest tracking, learned optical flow, physics losses as regularisers, continued self-supervised pretraining on WebCOOS footage ("the largest untested lever" in this report). — [R-BACK], [VIDEO]

**Report 2: "Wave camera architectures beyond backbones" (seven ranked additions, about 100 H200-hours for ranks 1-5)**
- Rank 1, grouped CV over the 43 groups with seed ensembles, distribution heads and buoy-level conformal intervals: **done** (three seeds, not five); B2-CV not met; coverage 0.904. — [R-ARCH], [BUOY]
- Rank 2, record-level pooling of all 16-s windows: **done in the interim run** (M3 and M4 met; the learned pooler was worse than the plain average and dropped); final run pending. — [R-ARCH], [VIDEO]
- Rank 3, partition-aware fusion with a gate, fitted out of fold, confirmed on fresh weeks: **done out of fold (S2 met)**; confirmation in progress (2026-11-27). TabPFN as a candidate combiner: not run (licence). — [R-ARCH], [BUOY]
- Rank 4, physics front ends for video: pixel spectrum **done** (M5 not met on all clips); breaking fraction and coherence features **in development** (uncommitted v2); timestack crest tracking **prepared** (reader packets), not run. — [R-ARCH], [VIDEO], [PIXWIP], [CHECKIN]
- Rank 5, period, direction and 1-D spectrum head on stills: **done** (S3 not met; wind-sea period and shape wins found; H_s harmed by joint training). The report assumed the caption carried each view's heading; it does not. — [R-ARCH], [BUOY]
- Rank 6, continued self-supervised pretraining: DINOv3 ViT-B on buoy views **pretraining done, P1 in progress**; V-JEPA 2 on unlabelled WebCOOS clips **not started** (waits for raw clips on the server, which need WebCOOS's consent). — [R-ARCH], [PRE], [VIDEO]
- Rank 7, encoder escalation (V-JEPA 2 ViT-g-384, V-JEPA 2.1, a pixel-reconstruction contrast): **not started** (conditional on rank 2 plateauing). — [R-ARCH]
- Side arm (per-frame features into a small Mamba or GRU over the full record) and AllTracker: not run. Excluded by the report: evidential heads, buoy-identity embeddings, video-generator features, monocular depth for height, a frozen 7B encoder. — [R-ARCH]
- The report's decision table against the results: "shuffled-frame video scores as well as ordered video → motion is unused; a per-frame DINOv3 replaces V-JEPA 2" matches the interim H_s result (0.533-0.535 m for video, still and shuffled; DINOv3 0.546 m); "grouped CV shows a backbone gap larger than the seed spread → encoder work moves up" is borderline (arms differ by 0.019 m, seeds span 0.016 m for H_s; 0.002 m vs 0.007-0.008 m for the wind sea). — [R-ARCH], [VIDEO], [BUOY]

**Report 3: "Wave measurements to learn next" (dated 2026-11-18; untracked on main, committed on the measurement branch)**
- Headline: "Train the next buoy model on wave slope. Train the next shore model on wave period." — [R-MEAS]
- Label check 1 (peak period): the S3 label builder averaged T_p over the ±30-min window although ADR 0010 says peak parameters come from the nearest record; 11,083 of 41,041 labels (27.0 %) are more than 0.25 s from every grid period. Follow-up **done as a sensitivity check**: under the nearest-record definition the camera scores 3.57 s and ERA5 1.95 s; ordering preserved. A matched model trained on the corrected target: not done. — [R-MEAS], [HANDOFF]
- Label check 2 (bandwidths): the S3 package inferred band widths from rounded centres; **done** in the slope labels with nominal widths (4√m0 / WVHT 0.976-0.988 at all 79 buoys); the cause of the 1-2 % offset is still open. — [R-MEAS], [MEAS]
- Order 1 (CPU): peak sensitivity table **done**; width check **done**; stronger wind/geometry slope baseline **done as a pilot** and defined as M-S1's comparator. — [R-MEAS], [MEAS]
- Order 2 (frozen slope probe, then a matched fine-tune; fresh-week replication): **prepared, bar fixed, not run**. — [MEAS]
- Order 3 (finish registered video comparisons; annotate crest times; fixed-transect vs learned timing): interim video done; crest annotation **awaiting human readers**. — [VIDEO], [HANDOFF]
- Order 4 (band energy, directional width, breaker type, individual wave height, stereo): labels and indexes **prepared**, nothing scored. — [CHECKIN]
- Sun/time/position-only direction comparator: **done**; it beats the camera (61.64° vs 75.83°). — [HANDOFF]
- Deriving Kim's actual target (mean individual height) from CDIP displacement: windows matched for 511 of 536 clips, all flag 2; **sensitivity-only, not run**. — [CHECKIN]
- Foam / whitecap fraction with reader-validated masks: packet of 25 views **awaiting masks**. — [CHECKIN]

### Inferences
- Reports 1 and 2 leaned on encoder choice and aggregation; the scored results moved the weight of evidence to "what the camera can physically see". Report 3 follows that shift, which is why its targets (slope, wind-sea bands, crest timing, breakers) are local, wind-sea or surf-zone quantities.
- The untouched items that still address an open paper comparison are few: re-running the papers' own architectures (code exists), a held-out-camera video test, and the Kim target from displacement records.

### Gaps
- The two copies of report 3 differ in one link path only (checked by diff); I did not compare the research-note folders behind reports 1 and 2.
- Whether the user has approved or declined any untouched item is not recorded in the files read.

## Open gaps and questions listed in docs/literature/research_gaps.md and docs/ROADMAP.md

### Takeaway
The standing gaps are sensor and data gaps the public sources cannot close: no real camera + platform-attitude data (G-9), no raw buoy motion from NDBC, no thermal IR with a wave instrument (G-5), spectral targets barely tried (G-6), and unmeasured label error at each shore site (G-10). The experiments added four empirical limits: swell, total period and direction are not visible to the buoy camera; calm seas are too rare at the test buoys to decide fusion questions; the buoy data are one season; and shore-camera models do not carry from one year to the next.

### Cited Findings

**Gaps table (research_gaps.md)**
- G-1 no large, diverse, synchronised real image + instrument dataset; G-2 label provenance unstated in the literature; G-3 no cross-station evaluation with instrument labels; G-4 difficult conditions excluded, not studied; G-5 thermal IR + wave instrument datasets essentially absent; G-6 spectral targets almost untried from vision; G-7 inconsistent metric reporting; G-8 no paper releases code and data; G-9 no real image + platform-attitude dataset; G-10 representativeness of a reference is assumed, not measured. — [GAPS]
- Status per the collection plan (2026-10-08): G-2, G-7, G-8 done in software; G-1/G-3 in progress via buoy cameras, Waimea and WebCOOS, with CoastCam not started; G-10 measured between buoys only (hourly H_s differs about 6 % under 10 km, a median 15 % at 10-30 km, from 4 % on open coasts to 60 % across Oahu), per camera site still open; G-5, G-6, G-9 need a self-collected site or vessel. — [PLAN], [README]

**No self-collected camera + IMU data; no raw buoy displacement**
- NDBC computes spectra on board and never transmits raw acceleration or displacement, so no NDBC product gives the hull's attitude at the instant of a photo; the horizon in the image is the only attitude proxy. — [INV], [NDBC]
- The self-collected site (FAU Harbor Branch area, CDIP 134 about 9-13 km from an ocean-side camera) is a design only: phases A (calibrated video, GPS time), B (instrument in view, stereo), C (radiometric thermal), D (IMU + GNSS on a vessel); site, permits and budget are open decisions. — [PLAN], [GAPS]
- The only held data with imagery, platform attitude and an independent wave reference on a moving platform is the Schwendeman & Thomson ship stereo set (55 bursts). — [SCAN], [EXT]
- CDIP does publish Waverider displacement (per-deployment files), which is the route to time-domain labels (H₁/₃, H_max, T_z, mean height); the ROADMAP still lists "displacement → zero-crossing labels" as to do. — [CDIP], [ROADMAP]

**Swell, peak period and direction not visible to the buoy camera**
- Q-M8 answered: the camera's calls follow wind-wave height (standardised slope 0.68) more than swell height (0.37); swell RMSE 0.572-0.597 m with R² 0.54-0.58 at the sealed buoys; ERA5 wins total H_s, peak period, energy period and direction; the horizon tilt does not fill the swell gap (correlation 0.05 with swell height). — [GAPS], [BUOY]
- Fold 0 (US West Coast, swell share 0.79, never seen in training) is the hardest: H_s 0.70 m; six West Coast buoys are under-read by 0.6-0.7 m on average. — [BUOY]
- No strip carries a compass heading, and a sun/location/time-only control beats the camera's direction output. — [BUOY], [HANDOFF]

**Calm-sea test power**
- The 12 test buoys are mostly open ocean: "smooth" has only 94 test strips, and all smooth test strips in Test 2b are at two buoys (44065: 69, 42057: 25); the docs conclude "more calm-sea buoys or months, not another combiner, would settle it". Conformal coverage is lowest in smooth seas (0.80). — [BUOY]

**Single-season buoy data**
- The package spans 2026-09-23 to 2026-11-14 ("7 weeks of autumn, one season"); the collection plan scheduled the cross-station benchmark for about 2026-12-26, after three months of archive; labels are realtime, to be rebuilt from quality-controlled annual files next year. — [PRE], [PLAN], [BUOY]

**Year-to-year transfer at shore cameras**
- Currituck: the camera beats WIS on held-out days but not on the held-out year; "smooth" recall falls from 0.75 (held-out days) to 0.49 (2025); at buoy H_s 0.3-0.4 m the share called smooth falls from 0.93 (2022-2024 validation weeks) to 0.64 (2025) and 0.56 (2026); measured haze explains only a small part; sandbars, exposure and compression are untested. — [CURR]
- Waimea: image statistics learned in 2009-2011 add nothing in 2013 ("not yet diagnosed"). — [WAIMEA]

**Other open questions in research_gaps.md**
- Still open: Q-D1 (how long the archiver must run per sea-state bin), Q-D2 (depth and exposure matching of references), Q-D4 (camera geometry stability per station), Q-D5 (self-deployed camera near CDIP 134), Q-D6/Q-D9 (WebCOOS licence and video terms; partly answered by the approved sample), Q-D7 (FRF Argus archive), Q-M1 (H₁/₃ vs H_m0 in real paired data), Q-M2 (buoy-to-surf-zone transformation), Q-M3 (directional-moment regridding), Q-M4 / hypothesis H3 (whitecap fraction with the buoy's wind), Q-G1 to Q-G4 (transfer to South Florida, drivers of domain shift, value of orthorectification, physics priors across stations), L-3/L-4/L-5 (radar and cBathy survey, screened-only sources, NDBC night-capture policy). — [GAPS]
- Answered or partly answered: Q-M5 (horizon tilt carries wind-sea information), Q-M6 (residual-to-nowcast learning: gains shrink to 0.012 m against ERA5 + WIS), Q-M7 (label rule changes RMSE by under 0.01 m), Q-M8. L-2 (full texts of Choi, Kim, Yang) is still listed as open in research_gaps.md, but benchmark_targets.md records all four papers as V1 since 2026-11-18. — [GAPS], [BT]

**ROADMAP items not done**
- v0.2.0: CDIP displacement labels; a learned image model on the Waimea splits; an always-on collector host; the USGS CoastCam adapter (five sites for leave-site-out tests); WIS/WW3 and wind as required baselines beyond ERA5 (WIS done for two sites); ERDDAP adapter; NDBC `adcp2` parser and deployment table; CF/ACDD export. — [ROADMAP]
- v0.3.0-v0.5.0: CoastCam orthorectification; a released `ndbc_buoycam_hs` dataset with Zenodo DOI and datasheet; Sentinel-1/HLS context and bathymetry; self-supervised pretraining with audited corpora (KOLOMVERSE, WebCOOS, NIMS); whitecap features; the self-collected site; camera + IMU on a moving platform; S(f) and S(f, θ) targets with stereo validation (Guimarães 2020). The ROADMAP's v0.3.0 entry still describes the buoy dataset as "ERA5 baseline and training to come", which the experiment docs supersede. — [ROADMAP]

### Inferences
- The gap list in research_gaps.md predates the November experiments in several rows (L-2, Q-M4, the G-6 row); the experiment docs are the more current statement of what is known.
- New modalities that directly attack a named gap and are already on disk but unused: `timex`/`brt` (integration over many waves), CDIP displacement (time-domain labels, Kim's target), the ship stereo set (attitude + imagery, G-9), Duck IR + lidar (G-5), night strips (45,944 archived, G-4).

### Gaps
- GitHub issues carrying the `research-question` template were not read (local files only).
- No document quantifies how many calm-sea strips or buoys would be needed to decide Test 2b's question.

## Constraints: compute, data-provider rules, storage, and anything that limits what can be tested next

### Takeaway
Compute is adequate but shared and slow to reach: one or a few GPUs of a busy 8 × H200 lab server over SSH and Docker, with the desktop GPU out of use for training. The binding limits are independent labelled sites and seasons, provider terms (WebCOOS pacing and redistribution, Yin's no-derivatives licence, DINOv3's licence), ERA5's five-day lag, thin calm and extreme samples, and a 12-buoy sealed test set that has already been used for several decisions.

### Cited Findings

**Compute**
- The lab works on its H200 server directly over SSH in Docker (no scheduler); "the server is shared and usually busy": on 2026-11-17 all eight GPUs were 70-99 % busy with 16-117 GB free each. A run gets one GPU, 24 CPU cores and 96 GB of memory. The Kubernetes route exists but is not used. — [CLUSTER]
- Measured costs: a strip run takes 40-55 min on one shared H200 (about 40 GB of GPU memory); 30 grouped-CV runs finished overnight; video feature extraction for four arms is about 1.5 s per clip; window features about 0.23 s per window on a busy GPU; the interim whole-clip run took 1.8 h (features) + 3.2 h (video probes) + 0.8 h (image probes) with about 34 GB of GPU memory; continued pretraining took about 9.1 GPU-hours for 30,000 steps at batch 256 (25 GB with activation checkpointing; 84 GB without, above the 70 GB limit set for the shared GPU). — [BUOY], [VIDEO], [PRE]
- Memory limits bite on shared containers: the pretraining container was killed at 160 GB and later needed 330 GB; a future run of that length "needs a limit of about 400 GB, or a restart from a checkpoint partway". — [PRE]
- The architecture report budgets ranks 1-5 at about 100 H200-hours and says to "budget two to three times the wall-clock" on the shared server; fully fine-tuning a 7B encoder is out (about 107 GB of optimiser state). — [R-ARCH]
- Data transfer to the server: about 1.5-2 Mbit/s through the bastion (a 3 GB package takes about 3.5 hours); the workable route is desktop → the project's R2 bucket → server at about 22 Mbit/s end to end (about 240 GB a day). The server's system disk is nearly full (95 % on 2026-11-17); work lives on the data disk, where scratch had 8.6 TiB free on 2026-11-19. — [CLUSTER], [VIDEO], [HANDOFF]
- Desktop GPU (Radeon RX 6800 XT class, 17.2 GB): "Do not train on the desktop GPU until its cooling has been checked"; it overheated and powered the desktop off on 2026-11-14 and was unchanged after cleaning. The desktop is also the primary collector host. CPU-only inference there runs at about 0.45 s per strip and model. — [CLUSTER], [COLLECT], [BUOY]

**Data-provider rules and licences**
- WebCOOS: one request at a time at the agreed pace; large downloads of data older than 90 days only with their agreement; acknowledgement text required; redistribution of images not addressed ("ask before publishing images or releasing them in a dataset"); raw clips on the GPU server need consent first. — [WEBCOOS], [VIDEO]
- Yin et al. 2025 clips: CC BY-NC-ND 4.0, derived datasets may not be redistributed; shared footage with WebCOOS must be kept together across datasets. — [REG], [CHECKIN]
- DINOv3 weights (and weights continued from them): Meta's DINOv3 licence, research use with acknowledgement in publications. V-JEPA 2 ViT-L: Apache-2.0/MIT. TabPFN: non-commercial licence, not accepted, so not run. — [BUOY], [PRE], [VIDEO]
- NDBC and USGS data are public domain; CDIP unrestricted with per-station acknowledgement; ERA5 CC BY 4.0; WIS free for scholarly use with acknowledgement; PacIOOS free reuse with acknowledgement; the three external stereo/IR sets are CC BY. — [REG], [CDIP], [ERA5], [WIS]
- SOCIB images carry no licence and are not bulk-downloaded until SOCIB agrees. — [SCAN]

**Storage and collection robustness**
- Raw WebCOOS collections live on a USB data drive; on 2026-11-19 the system disk had 192 GB free of 670 GB and the data drive 428 GB free of 916 GB, with 191 GB of raw video, 144 GB of stills, 55 GB of `timex` and 100 GB of `brt` on it. Bulk video was ruled out earlier at about 5 GB per camera-day (about 60 TB). — [WEBCOOS]; disk figures from `df`/`du` run 2026-11-19 on [VIDRAW]
- The buoy-camera stream cannot be re-collected (72-h retention); two collector hosts and a daily offsite copy to R2 exist, but there is no always-on host; the data drive's USB link dropped once (2026-11-11) and unclean resets damaged files twice (repaired). — [COLLECT], [PLAN]
- Cut video frames are about 35 MB per 10-min clip; token caches are about 2.1 MB per window for V-JEPA 2 features. — [VIDEO]

**Limits from the data themselves**
- Independent units are scarce: 55 training buoys, 12 validation, 12 test; four shore views at three sites; 18-20 local dates of video so far. — [R-ARCH], [VIDEO]
- The sealed 12 test buoys have served Tests 2, 2b and 3 and bars B1-B3; further decisions use the folds, and confirmations need new weeks. — [BUOY]
- ERA5 runs about 5 days behind real time, so any new-weeks confirmation waits; WIS ends 2025-12-31 and has no 2022 at Currituck, so 2026 shore work has only ERA5 as a model baseline (NOAA WW3 point output is the documented, not-yet-downloaded alternative). — [BUOY], [WIS]
- CDIP 200 has no data after 2026-07-20, so Oak Island video and stills after that date cannot be labelled; CDIP displacement matched so far is all provider flag 2. — [WEBCOOS], [CHECKIN]
- Rare conditions: 62 Currituck stills at 4 m or more; 299 Waimea images at 4 m or more on 27 days; 94 smooth strips at the test buoys; 154 of the 262 video clips in Kim's range fall in the lowest class. — [WEBCOOS], [WAIMEA], [BUOY], [VIDEO]
- Human annotation is a dependency for crest timing (independent readers) and foam masks. — [HANDOFF]
- Shore cameras are uncalibrated and re-point; the Currituck lens degrades year by year; so metric geometry and fixed crops are unavailable without per-view ground control. — [VIDEO], [CURR], [R-ARCH]

### Inferences
- Tests that need only CPU and saved predictions (fusion variants, label sensitivity, tabular multimodal combiners) are effectively free; tests that need new encoders on video are bounded by transfer and decoding, not GPU arithmetic; tests that need new sites, seasons, sensors or calm seas are bounded by time and permissions.
- Any multimodal experiment that puts raw WebCOOS video or Yin clips on shared infrastructure, or releases derived data, needs a provider answer first; experiments on buoy strips, CDIP, ERA5 and the CC BY external sets do not.

### Gaps
- Whether the desktop GPU's cooling has been fixed since 2026-11-14 is not recorded in the repo docs.
- No document gives a GPU-hour allowance or priority for the project on the shared server; the limits cited are observed load, not a quota.
- The R2 bucket's current size and cost are not in the files read.

[GIT]: <../../.git> "git worktree list, git branch -vv, git log main..<branch>, run 2026-11-19"
[HANDOFF]: <../../../Wave_Analysis-measurements/docs/experiments/claude_handoff_2026-11-19.md> "measurements worktree"
[CHECKIN]: <../../../Wave_Analysis-measurements/docs/experiments/measurement_checkin.md> "measurements worktree"
[MEAS]: <../../../Wave_Analysis-measurements/docs/experiments/wave_measurements.md> "measurements worktree, bar M-S1"
[SLOPECFG]: <../../../Wave_Analysis-measurements/configs/experiments/buoy_slope_v1.yaml> "measurements worktree"
[BUOY]: <../../.claude/worktrees/trusting-lalande-4a4e8b/docs/experiments/ndbc_buoycam_sea_state.md> "trusting-lalande worktree; newest buoy experiment doc, superset of strips-grouped-cv's"
[BUOY-MAIN]: <../../docs/experiments/ndbc_buoycam_sea_state.md> "main; ends at B3 and protocol check R"
[V0CFG]: <../../configs/experiments/ndbc_buoycam_sea_state_v0.yaml> "main; split comment"
[BT]: <../../.claude/worktrees/strips-grouped-cv-085fd2/docs/literature/benchmark_targets.md> "strips-grouped-cv worktree; Where we stand (2026-11-19)"
[VIDEO]: <../../.claude/worktrees/video-whole-clip-491c90/docs/experiments/webcoos_video.md> "video-whole-clip worktree"
[PIXWIP]: <../../.claude/worktrees/video-whole-clip-491c90/src/wave_analysis/processing/pixel_spectrum.py> "video-whole-clip worktree, uncommitted changes (git diff)"
[PRE]: <../../.claude/worktrees/dinov3-continued-pretraining-fc2b1a/docs/experiments/domain_pretraining.md> "dinov3-continued-pretraining worktree"
[CHANGELOG]: <../../.claude/worktrees/trusting-lalande-4a4e8b/CHANGELOG.md> "trusting-lalande worktree"
[CURR]: <../../docs/experiments/currituck_sea_state.md> "main"
[WAIMEA]: <../../docs/datasets/pacioos_beachcam.md> "main"
[WEBCOOS]: <../../docs/datasets/webcoos.md> "main"
[NDBC]: <../../docs/datasets/ndbc.md> "main"
[CDIP]: <../../docs/datasets/cdip.md> "main"
[ERA5]: <../../docs/datasets/era5.md> "main"
[WIS]: <../../docs/datasets/wis.md> "main"
[EXT]: <../../docs/datasets/external_research_datasets.md> "main"
[INV]: <../../docs/datasets/inventory.md> "main"
[COASTCAM-SRC]: <../../src/wave_analysis/sources/coastcam.py> "main; placeholder adapter"
[SCAN]: <../../docs/literature/data_source_scan_2026-10.md> "main"
[LAND]: <../../docs/literature/landscape_review.md> "main"
[GAPS]: <../../docs/literature/research_gaps.md> "main"
[P-KAMA]: <../../docs/literature/papers/kamagata2026.md> "main"
[P-KIM]: <../../docs/literature/papers/kim2023.md> "main"
[MATRIX]: <../../docs/literature/literature_matrix.md> "main"
[BT-MAIN]: <../../docs/literature/benchmark_targets.md> "main; Where we stand dated 2026-11-18"
[ROADMAP]: <../../docs/ROADMAP.md> "main"
[PLAN]: <../../docs/operations/collection_plan.md> "main"
[COLLECT]: <../../docs/operations/data_collection.md> "main"
[CLUSTER]: <../../docs/operations/cluster.md> "main"
[MODELS]: <../../docs/methodology/models.md> "main"
[EVAL]: <../../docs/methodology/evaluation.md> "main"
[README]: <../../README.md> "main"
[REG]: <../../data/registry/datasets.yaml> "main"
[R-BACK]: <../../reports/Vision backbones for wave cameras.md> "main"
[R-ARCH]: <../../reports/Wave camera architectures beyond backbones.md> "main"
[R-MEAS]: <../../reports/Wave measurements to learn next.md> "main, untracked; committed on the measurement branch"
[RUN-S1]: <../../runs/buoycam-cv-stage1-20261119/report.md> "run report"
[RUN-S2]: <../../runs/buoycam-cv-fusion-20261119/report.md> "run report"
[RUN-S3]: <../../runs/buoycam-cv-spectral-20261119/report.md> "run report"
[RUN-DRY]: <../../runs/buoycam-cv-fusion-confirm-dryrun-20261119/dry_run.json> "dry run of the S2 confirmation"
[RUN-V2]: <../../runs/v2-compare-20261118/report.md> "run report"
[RUN-B2]: <../../runs/b2-weights-20261118/report.md> "run report"
[VIDPKG]: <../../data/processed/packages/webcoos_video_v1/clips.csv> "local package, written 2026-11-19 06:28 EST; rows counted for these notes"
[VIDSUM]: <../../data/processed/video/clips_summary.json> "pairing summary, written 2026-11-19 08:04 EST"
[VIDRAW]: </mnt/wavedata/wave_analysis> "data drive holding the raw WebCOOS collections"
[FRESH]: <../../data/processed/pairs/ndbc_buoycam_fresh/summary.json> "pairing of strips archived after the package"
[FRESHPKG]: <../../data/processed/packages/ndbc_buoycam_fresh_v2cv/summary.json> "new-weeks package summary"
