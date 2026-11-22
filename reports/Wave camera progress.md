# What wave cameras can measure

Internal progress report · November 21, 2026

Across 79 buoys, our camera model estimated local wind-wave height with **26 cm error**, total significant wave height with **53 cm error**, and swell height with **72 cm error**. These are root mean square errors (RMSE), which give extra weight to large misses. The photos were most informative about locally driven seas. Their added value was modest once two wave models and the buoy’s own wind reading were already available. [Buoy height results](../docs/experiments/ndbc_buoycam_sea_state.md#grouped-cross-validation-every-buoy-a-test-buoy-once-fixed-2026-11-18-before-any-run-on-the-folds)

## Wind-wave height and total wave height

Imagine a long ocean swell rolling past a buoy while the local wind makes shorter waves on top. **Wind-wave height** describes the part of the sea being driven by the local wind. **Swell height** describes waves that have travelled away from where the wind generated them, or that remain after the wind has died down. **Total significant wave height** describes both together. It is a statistical height for a stretch of sea, roughly the average crest-to-trough height of its highest third of waves.

The buoy measures how wave energy is distributed across frequencies and separates the wind-wave and swell contributions. Their energies add, so **total height ≈ √(wind-wave height² + swell height²)**. For example, 1 m of wind waves and 2 m of swell give about **2.24 m total**. If the wind waves rise to 1.2 m and the swell stays the same, the total rises to 2.33 m. That is why a better reading of local chop can make only a small difference to total height on a swell-dominated day. [Check against 295,547 buoy records](../docs/experiments/ndbc_buoycam_sea_state.md#stage-2-partition-aware-fusion-fixed-2026-11-18-before-any-fusion-was-fitted-or-scored)

## How we tested the buoy cameras

We archived 41,091 daylight and twilight images from 79 National Data Buoy Center buoys from September 23 to November 14, 2026. Each hourly image is a strip of six views. We removed the printed station name and time, trained a model to read the views together, and checked its estimates against readings from the same buoy averaged within half an hour of the photo. [Dataset and training record](../docs/experiments/ndbc_buoycam_sea_state.md)

We held out whole regions in five rounds. Buoys within 300 km of a test buoy were excluded from both model training and the work of learning how to mix camera and forecast estimates. The main comparison had all required inputs for 34,723 images at 71 buoys.

## What the camera added

We combined **ERA5**, a reconstruction of past weather and waves, with **GFS-Wave**, a wave forecast, and the buoy’s measured wind. We then used the camera to adjust the wind-wave estimate while keeping the modelled swell. All three columns show RMSE on the same images. Lower is better.

| What we estimated | Camera alone | Wave models + measured wind | With camera added |
|---|---:|---:|---:|
| Height of locally generated wind waves | 25.44 cm | 22.58 cm | **20.79 cm** |
| Total significant wave height, roughly the average height of the highest third of waves | 51.93 cm | 26.70 cm | **26.23 cm** |

In everyday terms, the **mean absolute miss** is the distance between the estimate and the buoy reading, averaged without regard to whether the estimate was high or low. The median observed wind-wave height was **77 cm**. The camera missed by **15.8 cm** on average, the forecast by **14.1 cm**, and the combination by **12.7 cm**. For total height, the median was **1.73 m** and the corresponding misses were **37.9, 18.5 and 18.1 cm**. [Saved predictions used for these calculations](../runs/forecast-fusion-nested-v1/rows.parquet)

The wind-wave RMSE fell by **1.79 cm** (95% uncertainty interval, 0.88–2.53 cm). The camera helped in 27 of 41 geographic groups. The **8%** is the share of the forecast’s error removed. Total-height RMSE fell by just **4.7 mm** (1.8%). When each group counted equally, the total-height result was consistent with no improvement. [Score table](../docs/experiments/sensing_forecast_results.md) · [Results by location](figures/buoycam-fusion/station-groups.csv)

**Did it work on later dates?** We trained only on data from before October 29, then tested on 7,855 images taken October 31–November 13 at 67 held-out buoys. Simply averaging camera and forecast estimates cut wind-wave error from **26 cm to 23 cm**, helping 32 of 39 groups. This was still a historical test. We had examined those dates before, and ERA5 became available after the photos were taken. [Later-date scores](buoy-hpc-followup/heights/rmse.csv)

**Could the camera see swell?** We tested long waves with roughly 11–33 seconds between crests. On 38,334 images, wind readings alone gave 88.3 cm error. Adding the full image lowered that to 70.4 cm. Adding only a resized crop of the water gave 88.6 cm. The crop lost useful information, though which visual cue mattered is still open. [Swell experiment](../docs/experiments/hpc_buoy_followup.md#b-swell-from-the-part-of-the-image-below-the-horizon)

## Other buoy tests

| What we tested | Result at held-out buoys |
|---|---|
| **Wind-wave height against a single wave model** | On 36,857 matched photos, the camera alone scored **25.6 cm** against **34.3 cm** for ERA5, about 25% lower error. Adding GFS-Wave and measured wind made the comparison harder, as the main table shows. [Height comparison](<Paper outline.md>) |
| **Recognising sea states** | On the original 12 test buoys, the camera identified four **wind-wave** height bands with **80%** accuracy averaged across bands, versus **69%** for ERA5 and **49%** from measured wind alone. For four **total-height** bands the camera scored **70%** versus **74%** for ERA5. It often called a swell-dominated sea too calm. [Sea-state tests](../docs/experiments/ndbc_buoycam_sea_state.md) |
| **Period and direction** | Period is the time between crests. For the dominant wind waves the camera’s period error was **0.69 s**, versus **0.93 s** for ERA5. For the dominant waves in the whole sea the camera lost, **3.44 vs 1.70 s**. Its mean direction error was **76° vs 14°**. [Period and direction tests](../docs/experiments/ndbc_buoycam_sea_state.md) |
| **Wind, gusts and steepness** | Against the buoy’s wind sensor, camera error was **0.96 m/s** for wind and **1.14 m/s** for gusts, versus **1.27** and **1.57 m/s** for ERA5. Adding photos to wind and horizon angle reduced error in surface slope (vertical rise per horizontal distance) from **0.0102 to 0.0089**. Further training of the image model gave no measurable slope gain. [Wind and slope tests](../docs/experiments/wave_measurements.md) |
| **Which waves leave a visible signal?** | The camera explained **86%** of the variation in wave-band height for waves 3.9–6.5 seconds apart and **47%** for waves 10.5–33 seconds apart. This uses fixed ranges of wave periods. The buoy’s wind-wave and swell split can move with sea conditions. [Wave-band tests](../docs/experiments/wave_measurements.md) |
| **Published camera models** | Re-running the models on our images at unseen buoys, our six-view model scored **98.5% vs 95.5%** on Choi’s calm/rough task, **66.6% vs 52.7%** on Kim’s five height bands, and **49.8 vs 56.4 cm** total-height RMSE against Yang’s model. Telling two extremes apart was much easier than assigning a height band. [Same-data comparisons](../docs/experiments/sensing_paper_bars.md) |

**Could we improve the model itself?** Switching image models lowered total-height RMSE from 55.0 to 53.2 cm across the 79 buoys and changed wind-wave RMSE by only 0.2 cm. Extra training on unlabelled buoy images missed its target. Feeding in horizon tilt separately gave inconsistent results. The model’s nominal 90% prediction ranges covered 90% of total-height readings overall but fewer than 80% at 11 buoys. [Model and uncertainty tests](../docs/experiments/ndbc_buoycam_sea_state.md) · [Extra training](../docs/experiments/domain_pretraining.md)

**What helped at a new location?** Adjusting the camera from unlabelled images increased wind-wave error from 26.0 to 37.7 cm. Using 30 hours of actual buoy readings to correct the camera’s local bias was more useful. On later observations, total-height error fell from 56.7 to 50.7 cm. In a separate test of the combined forecast, a week of local readings gave no overall gain. [Site calibration](../docs/experiments/sensing_site_adaptation.md) · [Forecast calibration](../docs/experiments/wind_wave_reliability.md)

**When did the camera make the forecast worse?** At buoy 41040 the forecast already estimated wind waves to 7.5 cm RMSE. The camera tended to read low, and adding it raised error to 16.2 cm. An exploratory rule that varied camera weight with wind, forecast height and disagreement between the two estimates scored 19.92 cm across all locations. A fixed weight scored 20.10 cm. That extra complexity bought less than 2 mm on the historical sample. [Station review](<Paper outline.md>) · [Weighting tests](../docs/experiments/wind_wave_reliability.md)

## Shore cameras and other sensors

| What we tested | Result |
|---|---|
| **Shore photographs** | On dates in the held-out year 2025, a single still estimated total height with **26.5 cm** error, versus **61.3 cm** for an estimate based on typical conditions. Ten-minute averaged images did worse (**30.8 cm**), as did keeping the brightest value at each pixel (**31.1 cm**). In a separate four-state test at Currituck, the camera scored **73%** averaged across classes versus **77%** for the historical wave model. [Shore stills](../docs/experiments/sensing_timex_bars.md) · [Currituck](../docs/experiments/currituck_sea_state.md) |
| **Shore video** | Across four cameras, 3,685 clips and 46 dates, the whole-clip model scored **30.5 cm** for total height versus **54.5 cm** for typical conditions and **31.3 cm** using only the middle window. Shuffling the frames made no measurable difference. Measuring changes in water brightness helped with dominant period (**2.76 to 2.59 s**). Extra features describing breaking waves failed to improve height on 28 confirmation dates. [Final video tests](../.claude/worktrees/video-whole-clip-491c90/docs/experiments/webcoos_video.md#final-run-2026-11-2021-3685-clips-46-local-dates) |
| **Published shore-video model and individual waves** | On the earlier 18-date sample, our whole-clip model scored **51.3 cm** for total height versus **65.3 cm** for Kamagata’s model re-run without its physics penalty. Estimating the average time between individual waves from water brightness scored **1.29 vs 1.37 s** for typical conditions, an inconclusive difference. [Shore comparisons](../docs/experiments/sensing_per_wave_bars.md) |
| **Breaker type** | At beaches held out of training, our model distinguished spilling, plunging and surging breakers with a score of **0.84** versus **0.65** for Yin’s model. The score runs from 0 to 1 and weighs all three types equally. A simpler version that ignored frame order reached **0.85**. [Breaker tests](../docs/experiments/breaker_types.md) |
| **Thermal imagery and ground motion** | At Duck, a thermal image showing where waves broke improved individual-wave height on one test day (**81.6 to 71.8 cm**) and worsened it on the other (**50.4 to 55.8 cm**). At Waimea, ground motion alone gave **53.5 cm** height error, worse than the **45.5 cm** monthly-average estimate on 91 test dates. The seismic pilot stopped there. [Thermal pilot](../docs/experiments/sensing_duck_pilot.md) · [Seismic pilot](../docs/experiments/sensing_seismic_bar.md) |

## What is still open

The fresh-photo buoy test uses models and scoring rules fixed in advance. Scoring is expected around November 27, once ERA5 covers the required dates. A later 14-day test will check whether adjusting the camera’s weight to local conditions beats a fixed blend. On shore, the five-band wave-height test now has 430 eligible storm clips from 22 dates and is awaiting a score. [Buoy confirmation](../.claude/worktrees/trusting-lalande-4a4e8b/docs/experiments/fresh_strip_confirmation.md) · [Blend follow-up](../docs/experiments/wind_wave_reliability.md#independent-two-week-replication) · [Storm comparison](../../Wave_Analysis-sensing-shore/docs/experiments/sensing_per_wave_bars.md)
