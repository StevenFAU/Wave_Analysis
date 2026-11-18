# Published results to compare against, and the models they suggest

Compiled 2026-11-18 from the [landscape review](landscape_review.md) (§2, §9), the
[literature matrix](literature_matrix.md), the per-paper notes in [papers/](papers/)
and searches of 2026-11-18. Evidence levels as in the landscape review: **V1** full
text read, **V2** abstract or publisher record, **V3** project notes, **S** screened
only. The full texts of Choi 2020, Kim 2023 and Yang 2026 sit behind publisher access
(ScienceDirect, SSRN refused automated and browser access on 2026-11-18); their
numbers were V2/V3 until Steven supplied the PDFs on 2026-11-18; all four are now V1.

**How to read a comparison.** A published number belongs to its own data, labels and
split. A model "improves on" it only where the task and measure match, and a number
from a random split of one site (test images minutes from training images) is not
the same claim as one at unseen sites. Each row says which of our tests matches it.

## Camera methods

| Work | Platform, input | Labels | Split | Their result | Our matching test |
|---|---|---|---|---|---|
| Choi et al. 2020, *Ocean Eng.* 201 (**V1**) | **NDBC buoy cameras**, one view (lower half, 224²), 20 buoys, 2018-03/05 | same buoy's H_s 30 min earlier | 35,696 / 8,924 examples, same buoys and days in both | **84.05 % ± 0.14** calm (< 1.52 m) vs rough (> 3.05 m) | same task at 12 unseen buoys: v0 classifier 98 % [97, 99] (indicative); v2 heights model |
| Choi et al. 2020, ship part (**V1**) | container-ship video, 16 frames at 1 s | shipboard wave radar | 2,675 / 344 clips, not described | RMSE 0.133 m, MAE 0.089 m | none (no ship data) |
| Kim et al. 2023, *Ocean Eng.* 288 (**V1**) | one fixed camera on a WEC, grayscale, 1 h a day for 30 days; snapshot or up to 300-s video | seafloor AWAC, **average** wave height (0.631 × H_s), one value per day | random frames / clips (same hour in train and test) | **78.3 %** snapshot, **93.2 %** 180-s video, 5 classes of 0.5 m (0.75-3.25 m) | Kim's bins on v2 heights (buoy cameras); at a shore camera with video (Currituck, Jennette, Oak Island) once the WebCOOS video sample is in |
| Kamagata et al. 2026, arXiv:2607.11998 (**V1**) | shore video, 16 frames at 10 fps, V-JEPA + SlowFast | **not stated** (traceable to no instrument) | 6 scenes train; 5 held-out videos (Oak Island E/W, Xishawan) | H_s **RMSE 0.586 m, R² 0.107, Pearson 0.436**; T_p 1.42 s; direction 37° | v2 heights at unseen buoys; Oak Island stills (and video) with CDIP 200 labels, the same views |
| Kamagata et al. 2026, arXiv:2606.13302 (**V1**) | shore video, ViViT | expert + timestack + optical-flow pseudo-labels | 4 held-out scenes | T_p RMSE 0.80 s | period from WebCOOS video against CDIP T_p (needs video) |
| Yang et al. 2026, *Ocean Eng.* 352 (**V1**) | ship camera + IMU attitude at 2 Hz, **simulation** (Unity/Crest, 47 sea states) | virtual buoys | K-fold over pooled samples; unseen states separately | RMSE 0.032 m (simulated); image only 0.098 → + attitude 0.073 | Bar V2-1: horizon attitude + images vs images, real data |
| Xu et al. 2024, *JMSE* 12:2003 (V1 via MDPI) | one shore camera, Qingdao, 1 frame/min, 1920 × 1080 | Waverider MKIII nearby, hourly, interpolated to minutes | **random 6:2:2 of frames** | MAE 0.04 m, MAPE 8.5 % with wind, temperature, humidity, pressure and tide as inputs; image-only CR 57.7 %; wind-sea vs swell images 94 % | a random split of our data beside our station split shows what that protocol adds |
| Buscombe et al. 2020, *Coast. Eng.* 155 (V2) | IR / RGB at Duck and Santa Cruz | instrument-derived | not verified, single sites | RMSE 0.14 m (IR), 0.11 m (oblique RGB) | single-site shore tests (Currituck day split) |
| Vorkapic et al. 2024, *Ocean Eng.* 309 (V2); Pobar et al. 2026, *IEEE OJ-ITS* 7 (S) | ship bridge camera, Beaufort 1-8, real + synthetic | **observers** | not verified | 96 % within one class; mean F1 62 % | Beaufort from the buoy's own wind (an instrument) on buoy cameras |
| Visual-range sea-state dataset, *Symmetry* 14:1487 (2022) (S) | 100,800 images, 4 sea states | not verified | not verified | 81.8 % (NASNet-Mobile) | none until read |
| Ocean Dynamics 2025, doi:10.1007/s10236-025-01743-x (S) | shore video **timestacks**, no learning | not verified | two months, one site | H_s RMSE 0.12 / 0.09 m, r 0.76 / 0.80 | the physics baseline for shore video |

## Other sensors (context: what an image of the sea surface can reach)

| Work | Sensor | Their result |
|---|---|---|
| Kwon et al. 2023, *PLOS ONE* (V1) | X-band radar sequences, 3-D CNN, buoy labels | RMSE 0.358 m, R 0.970 (public data) |
| SWH-WindNet, *Ocean Eng.* 2026 (S) | X-band radar + wind | r 0.946, RMSE 0.389 m |
| Quach et al. 2021, *IEEE TGRS* (V2) | Sentinel-1 wave mode, NN | RMSE 0.3 m vs altimeter (operational since 2022) |
| WV-Net, arXiv:2406.18765 (S) | SAR self-supervised, frozen + linear probe | RMSE 0.50 m (vs 0.60 supervised) |
| OceanSAR-2, arXiv:2601.07392 (S) | SAR, DINOv2 pretraining | H_s among its tasks |
| SWOT KaRIn (V2) | altimetry | RMSE 0.43 m vs buoys |

## What the sources suggest building (ranked by value for this data)

1. **Heights, not classes, with the motion as an input** (Yang 2026's attitude fusion;
   Xu 2024's separate wind-sea and swell models). Built: v2 (`MultiViewRegressor`,
   arms with and without the horizon).
2. **Foundation backbones**: a frozen or fine-tuned DINOv2/v3 or V-JEPA 2 encoder in
   place of ConvNeXt-Tiny (Kamagata 2026 used V-JEPA; OceanSAR-2 and WV-Net show
   self-supervised features beat supervised ones for SAR wave height). Cheap: same
   package, one backbone swap.
3. **The camera with the buoy's own wind** (Xu 2024's multi-factor inputs;
   SWH-WindNet). The anemometer sits on the same hull; the v1 control shows wind alone
   at balanced accuracy 0.49 against the camera's 0.80 for the wind sea, so the
   question is what the pair reaches.
4. **Video for shore cameras** (Kim, Kamagata, Choi's ship model): WebCOOS keeps
   10-min (Oak Island) and 4-min (Currituck, 20 fps, 2688 × 1520) clips back to at
   least March 2024; a sample request is drafted. Time-domain labels (mean wave height,
   as Kim) can be computed from CDIP displacement records.
5. **Synthetic + real training** (Pobar 2026; Kamagata's Airy clips) for rare seas:
   later, evaluated only on real data.
6. **Ordinal-aware losses** for any class task (sea-state classes are ordered).

## Protocol notes for every comparison

- Report the strict split (unseen buoys or sites) as the result; a random split, run
  only to show what it adds, beside it.
- Average several seeds (one seed moved v0's test macro-F1 by 0.12).
- Give intervals from a bootstrap over whole buoys (or buoy-days).
- State the label's instrument and definition (H_s from the spectrum vs H₁/₃ vs mean
  height) for every number.
