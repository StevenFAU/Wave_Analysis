# Research Gaps and Open Questions

Tracked as GitHub issues with the `research-question` template. IDs are stable
so that commits and results can reference them.

## Gaps

| ID | Gap | Evidence | What this project does about it |
|---|---|---|---|
| **G-1** | No large, diverse, **synchronised real-world** image/video + **instrument** dataset | All four papers (literature matrix) | NDBC camera archiver (prospective; 91 cameras, all with directional spectra); CoastCam, WebCOOS, and CDIP pairing (v0.2–0.3) |
| **G-2** | **Label provenance** is often unstated or non-physical | Kamagata 2026 labels untraceable; Yang 2026 virtual buoys | Every label carries reference station, sensor, estimator, Δt, and distance (`MultimodalSample`) |
| **G-3** | **No cross-station evaluation with instrument labels** | Literature matrix row "held-out" | Cross-station is the primary benchmark (`configs/experiments/baseline_hs.yaml`) |
| **G-4** | **Difficult conditions excluded** rather than studied | Choi (brightness), Kim (07–11 h, weather) | Annotate-don't-delete; illumination, glare, and quality strata |
| **G-5** | **Thermal IR + wave instrument** datasets essentially absent | Only small close-range IR sets exist (Buscombe & Carini 2019; Buscombe et al. 2020 OWG, Duck); no multi-site IR + spectra dataset ([landscape review](landscape_review.md)) | Schema ready (`ThermalMetadata`); candidate for self-collection |
| **G-6** | **Spectral targets** S(f), S(f,θ) almost untried from vision | Papers predict bulk scalars only | Spectra preserved end-to-end; energy-conserving regridding; spectral-consistency loss |
| **G-7** | **Metric reporting inconsistent** (NRMSE normaliser, linear direction RMSE, in-sample "full-set" metrics, i.i.d. CIs) | Kamagata 2026 details (V1) | Required reporting set (`methodology/evaluation.md`) |
| **G-8** | **Reproducibility**: no paper releases code and data | Literature matrix | Open code, manifests, provenance, RO-Crate, DOIs |

## Open questions

### Data availability
- **Q-D1** How long must the NDBC camera archiver run to reach N samples per
  sea-state bin at M stations? (Estimate from the H_s climatology in the
  inventory.)
- **Q-D2** Which calibrated CoastCam / WebCOOS / FRF cameras have a wave
  instrument within a few km and similar depth? What is the nearest physically
  meaningful reference distance? *Partly answered (2026-09-26):* nearest CDIP
  Waveriders are 6–44 km from the public cameras (Waimea ↔ CDIP 106, 6.3 km;
  USGS Dorado has an on-site Spotter); table in
  [landscape_review.md §5.3](landscape_review.md). Depth and exposure matching
  still open.
- **Q-D6** WebCOOS imagery licence and archive depth per camera (API token
  required; terms not found in public docs).
- **Q-D7** FRF Duck Argus imagery: is the archive available on request, now
  that the OSU CIL FTP host is gone and `argus_temp/` on THREDDS is empty?
- **Q-D8** PacIOOS Waimea benchmark: how much do the 2009–2013 images and CDIP
  106 overlap after QC, and how do surf-zone views relate to a buoy 6 km
  offshore in ~200 m of water? (Wave transformation versus offshore-state
  estimation.)
- **Q-D3** Which public sources expose complete S(f), and which have directional
  moments sufficient for S(f, θ)? (NDBC: 127 stations with all five files.)
- **Q-D4** NDBC camera geometry: stability of field of view and heading per
  station/deployment; the fraction of each strip that is sea.
- **Q-D5** CDIP 134 (Fort Pierce, FL; WMO 41114; Datawell DWR-M3) lies near FAU
  Harbor Branch. Is a self-deployed camera (RGB and/or radiometric thermal)
  viewing its location feasible? This would be a co-located dataset designed
  for this project.

### Methodology
- **Q-M1** How large is the H₁/₃ vs H_m0 label difference in real paired data,
  and does mixing estimators measurably hurt models?
- **Q-M2** Buoy-to-shore propagation: when does a group-velocity lag
  correction improve pairing, and when do shoaling and refraction dominate?
- **Q-M3** How should spectra from different grids be combined for training
  without distorting energy? (Energy-conserving rebinning implemented; directional
  moments pending.)
- **Q-M4 (hypothesis H3)** On NDBC buoy cameras, does image whitecap fraction
  combined with the buoy's measured wind predict H_s better than wind alone?
  (Whitecap coverage follows a wind-wave Reynolds number combining wind stress
  and H_s; Brumer et al. 2017.)
- **Q-M5** Does per-view horizon tilt (six-view plane fit, one sample per hour)
  carry sea-state information, since NDBC directional buoys themselves infer
  waves from hull pitch and roll? (Horizon tracking: Schwendeman & Thomson 2015.)
- **Q-M6** Does predicting the residual (buoy − WIS/WW3/ERA5 nowcast) from
  imagery beat direct regression, and does any camera model beat the nowcast
  at all?

### Generalisation
- **Q-G1** Can a model trained on one coast transfer to South Florida?
- **Q-G2** Which environmental factors dominate domain shift (illumination, water
  colour, camera height, wave climate)?
- **Q-G3** How much does camera calibration (orthorectification) improve
  cross-station performance?
- **Q-G4** Do physics priors improve cross-station performance rather than only
  in-distribution fit? Does a steepness-band prior beat the fixed-steepness
  prior of Kamagata et al.?

### Literature
- ~~**L-1** Review arXiv:2606.13302~~ **Done (V1, 2026-09-26).** Same group as
  Kamagata 2026. T_p only. Labels are expert visual plus timestack FFT (13
  gold scenes) and optical-flow pseudo-labels on stock footage (20 silver
  scenes). Held-out RMSE 0.80 s on 4 scenes. Data and code on request. See
  [landscape_review.md](landscape_review.md) and
  [source_verification.md](source_verification.md).
- **L-2** Obtain full texts of Choi 2020, Kim 2023, and Yang 2026 to raise V3
  claims to V1 (`source_verification.md`).
- **L-3** Survey X-band radar wave inversion and Argus-based depth/wave
  inversion (e.g. cBathy-type methods) as physics-based vision baselines.
  *First pass done* in the landscape review (§2.1, §2.5). Kwon et al. 2023
  provides public radar + buoy data.
- **L-4** Read the items marked **S** (screened only) in the landscape review
  before citing any of them.
- **L-5** NDBC's BuoyCAM FAQ says photos are taken "typically only during
  daylight hours". Hourly night images were observed on 2026-09-26. Ask NDBC
  whether night capture is policy or configuration.
