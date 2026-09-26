# Landscape Review: Imaging-Based Wave Estimation, Models, Industry, and Open Data

**Search date:** 2026-09-25/26. **Scope:** academic and industry work on estimating
ocean-wave parameters from images, video and related remote sensing, the model
families in use, and **publicly available data that could train or evaluate**
this project's models. Companion to the core-paper review
([literature matrix](literature_matrix.md), [paper notes](papers/README.md)).

**How to read the evidence levels.** Levels V1–V3 are defined in
[source_verification.md](source_verification.md). This review adds two more:

| Level | Meaning |
|---|---|
| **V1** | Full text read. |
| **V2** | Abstract, publisher metadata, or the authors' dataset/code record read. |
| **S** | *Screened only*: identified by title and venue in a search. The content is **not verified** and must be read before it is cited for any claim. |
| **Q** | *Queried*: the data service was accessed programmatically in this session and the stated facts (counts, dates, endpoints) come from the response. |

---

## 1. Summary for this project

1. **The bottleneck is labelled data, not architectures.** Every camera paper
   we found uses one of the following: a single private site (Kim 2023; Buscombe
   2020, whose splits are unverified), simulated data (Yang 2026), or labels that
   are not traceable to an instrument (Kamagata 2026; Kamagata et al.
   arXiv:2606.13302, whose labels are expert visual judgement plus timestack
   FFT and optical-flow pseudo-labels on stock footage). **No public benchmark pairs
   imagery with instrument wave spectra across several sites.** G-1 and G-3 in
   [research_gaps.md](research_gaps.md) are confirmed as open.
2. **Deep learning on images already works operationally for H_s, but on SAR,
   not cameras.** Since 2022-06-07, Sentinel-1 Wave Mode OCN products carry a
   neural-network "total" H_s (Quach et al. 2021). That model was trained on
   SAR–altimeter collocations from 2015–2017, tested on the held-out year 2018,
   and cut RMSE from about 0.6 to 0.3 m (V2). This is the template to copy:
   labels from an **independent sensor at scale** and a **temporal hold-out**.
3. **Height is the hardest target for a single camera, while period and
   direction are better posed.** Physics-based methods reliably get period,
   direction, celerity, and depth from video (timestacks, cBathy). Height needs
   metric geometry (stereo, calibrated breaking-height estimators near shore, or
   lidar). Deep models that regress H_s from uncalibrated monocular frames
   generalise poorly across sites (Kamagata 2026 held-out R² = 0.107, V1).
4. **The NDBC buoy cameras carry two signals with a physical link to the
   wave state:** horizon tilt and whitecap coverage.
   - *Tilt.* NDBC's own directional buoys infer wave direction from hull pitch
     and roll. Horizon tracking recovers roll and pitch to about 0.5° (S), and
     Schwendeman & Thomson (2015) used horizon tracking to stabilise shipboard
     video (V2). One frame per hour is a heavily undersampled sample of the same
     slope process, but it is physically grounded.
   - *Whitecaps.* Whitecap coverage scales with a wind-wave Reynolds number that
     combines wind stress and H_s (Brumer et al. 2017, V2). NDBC buoys **also
     measure wind**, so image whitecap fraction plus measured wind constrains
     H_s. This is testable as new hypothesis **H3**
     ([research_gaps.md](research_gaps.md)).
5. **Most public shore cameras have a CDIP Waverider 6–45 km away. NDBC-owned
   directional buoys are typically 20–400 km away** (§5.3, computed from the
   inventory and CDIP metadata). CDIP is therefore the primary label source for
   shore cameras, and NDBC for buoy cameras.
6. **Immediately usable historical pairs exist.**
   - PacIOOS Waimea Bay cameras (2009–2013, about 17,900 hourly daytime images,
     free reuse) are 6.3 km from CDIP 106 (Q).
   - USGS CoastCams have a public, listable S3 archive of hourly snapshots since
     2023 (Q), calibrated data releases, and CDIP buoys 15–40 km away.
   - USGS Dorado (Puerto Rico) has a Spotter buoy of its own about 2 km offshore (Q).
7. **Industry runs the largest camera fleets but publishes no validation.**
   Surfline states that it computes wave height with computer vision on 1,400
   cameras (V2, vendor pages) but gives no instrument comparison. Radar vendors
   (WaMoS II, Miros Wavex) do publish buoy comparisons. An open, instrument-referenced,
   cross-site benchmark would be a contribution that industry is not positioned to make.
8. **Reviewed camera papers lack strong non-vision baselines.** None of them
   compares against the obvious alternatives: a wave-model nowcast at the site
   (WW3, ERA5, WIS), persistence of the nearest buoy, or a propagated offshore
   buoy. A camera model has value only where it beats these. We add them as
   required baselines (§7), and propose **residual learning**: predicting the
   buoy-minus-model residual from imagery.

---

## 2. Techniques by sensing modality

### 2.1 Fixed shore cameras (monocular)

**Physics and signal processing (Argus lineage).**
- *Review.* Holman & Haller (2013, *Annu. Rev. Mar. Sci.* 5:95–113) surveys
  nearshore remote sensing (V2).
- *Timestacks.* A pixel array is sampled through time to give period, celerity,
  and runup. Recent refinements include an Ocean Dynamics (2025) timestack wave
  parameter study (S) and U-Net crest detection in timestacks (S).
- *Bathymetry and direction from celerity.* The linear dispersion relation is
  inverted, as in cBathy (Holman, Plant & Holland 2013, *JGR Oceans*, V2),
  including from ordinary coastal webcams (*Remote Sensing* 17:2274, 2025, S).
- *Breaking wave height.* Almar et al. (2012, *Coastal Engineering*) estimate
  breaker height from roller extent on the wave face after geometric
  rectification (V2). This works only in the surf zone and needs calibration.
- *Spectral video analysis and texture motion.* Peak period from surface
  texture motion (*Acta Oceanol. Sin.*, 2024, S).
- *Open tooling.* The CIRN toolboxes (QCIT; cBathy), ULISES, COSMOS,
  Beachkeeper Plus, and SIRENA provide calibration and rectification.

**Deep learning.**

| Work | Input → target | Labels | Evaluation | Level |
|---|---|---|---|---|
| Buscombe et al. 2020, *Coast. Eng.* 155:103593 ("optical wave gauging") | IR close-range, RGB orthomosaic, oblique RGB → H, T | Instrument-derived. Split protocol not verified | IR: RMSE 0.14 m / 0.41 s. Oblique RGB: 0.11 m / 0.81 s. Single-site sets | V2 (code and data on GitHub) |
| Kim et al. 2023, *Ocean Eng.* | Grayscale snapshots and sequences → 5 H classes | ADCP (AWAC) | Single site, daytime | V2/V3 |
| Kamagata et al. 2026, arXiv:2607.11998 | 16-frame clips → H_s, H_max, T_p, T_z, θ | **Unstated** | Held-out H_s R² 0.107 | V1 |
| Kamagata et al., arXiv:2606.13302 (v2 2026-07-19) | 60-frame clips → T_p | Expert visual + timestack FFT (gold, 13 scenes); optical-flow pseudo-labels on **stock footage** (silver, 20 scenes); Airy synthetic (>1,000 clips) | 4 held-out scenes: LtViViT RMSE 0.80 s, SI 0.10. Data and code "on request" | **V1** |
| *Remote Sensing* 18:1385 (2026): UAV waterline + timestack | UAV video → swash-zone T | Timestack | n/a | S |
| *Ocean Eng.* (2024): three-stage wave element extraction from offshore-infrastructure monocular video | Video → wave elements | n/a | n/a | S |
| *Int. J. Nav. Archit. Ocean Eng.* (2023): CNN wave detector + pixels-per-metre | Images → H | n/a | n/a | S |

**Lesson.** The strongest results come from **close-range, single-site,
calibrated** setups (IR at Duck; orthomosaics). Unsurprisingly, cross-site,
uncalibrated, oblique setups are the weakest. For this project,
**orthorectification (v0.3) is on the critical path** for shore cameras
(Q-G3).

### 2.2 Cameras on moving platforms (ships, buoys, USVs)

- **Choi et al. 2020** (NDBC buoy cameras and a container ship; V2/V3) and
  **Yang et al. 2026** (simulated ship with IMU attitude fusion; V2/V3) are
  covered in the [literature matrix](literature_matrix.md).
- **Vorkapic, Pobar & Ivašić-Kos (2024)**, *Ocean Eng.* 309 (arXiv:2407.03755):
  a bridge-mounted camera on an ocean-going vessel, Beaufort classes by
  manual annotation, ResNet-101, NASNet, MobileNetV2, and ViT-B/32 (V2). The
  labels are observer-based, not instrument-based. Data availability is not stated.
- **Horizon geometry.**
  - Roll and pitch from visual horizon recognition to under 0.5° (Springer
    2014, S).
  - Schwendeman & Thomson (2015, *JTECH* 32:164–176) use horizon tracking for
    shipboard video stabilisation and rectification (V2).
  - Wave period estimated from lidar-buoy pitch and roll (*Sensors* 2021, S).
- **Ship as a wave buoy.** Nielsen's wave-buoy analogy and its ML successors
  infer sea state and directional spectra from vessel motion (several
  *Ocean Eng.* and *Appl. Ocean Res.* papers, S). For camera platforms,
  platform motion is **signal**, not only nuisance. Yang et al. already fuse
  attitude, and NDBC camera tilt is a free attitude proxy.
- **Whitecaps from ships.**
  - Brumer et al. (2017, *JPO* 47:2211): more than 600 twenty-minute shipboard
    video records at 20 Hz; whitecap coverage modelled with a wind-wave
    Reynolds number (V2).
  - Schwendeman & Thomson (2015, *JGR Oceans*): whitecap coverage against wind
    stress, wave slope, and dissipation measured by SWIFT drifters (V2).
- **Maritime vision datasets without wave labels.** These are useful for
  pretraining (§5.2).

### 2.3 Stereo imaging (geometric ground truth)

- **WASS** (Bergamasco et al. 2017) is an open-source stereo reconstruction
  pipeline that gives space–time surface elevation η(x, y, t). Faster variants
  are WASSfast and CNN-based reconstruction (V2).
- **Guimarães et al. 2020**, *Sci. Data* 7:145 (V1 via PMC full text).
  - Twelve ~30-min stereo records at 10–15 Hz from four platforms: Black Sea
    Katsiveli (~30 m depth), Acqua Alta (~17 m), Socheongcho (~50 m), and La
    Jument lighthouse.
  - H_s 0.30–10.03 m and T_p 2.63–13.33 s, with co-located wave wires, AWAC,
    and X-band radar.
  - Licence CC BY 4.0; data on IFREMER Sextant (doi:10.12770/af599f42-2770-4d6d-8209-13f40e2c292f).
- **Recent work (S):** neural 3-D wave reconstruction from cameras
  (arXiv:2512.06024), DSGP-Net for sparse stereo (*Ocean Eng.* 2026),
  floating-platform stereo trained in synthetic environments (*Ocean Eng.* 2025),
  and deep stereo matching for sea-level monitoring (*JMSE* 12:961).
- **Role here:** physics validation. Stereo gives spectra and steepness
  from imagery alone, so the imagery → S(f) pipeline can be checked without
  point-sensor collocation error.

### 2.4 Wave breaking, whitecaps, and thermal infrared

- **Stringari et al. 2021**, *Sci. Rep.* 11:3604: active-breaking
  classification with deep networks. Code, 10k and 20k labelled training sets,
  and a 1k test set are available (BSD-3; GitHub `caiostringari/deepwaves`) (V2).
- **Buscombe & Carini 2019**, *Remote Sensing* 11:859: breaker type from close-range
  **thermal IR** at Duck, 89–93 % accuracy (V2). This fills part of G-5: IR
  imagery with wave context exists, although it is not a large paired dataset.
- **Carini et al. 2021**, *JGR Oceans*: lidar and IR fusion at breaking onset (S).
- **Yin et al. 2025** (VWBT-9000): breaking type only; licence CC BY-NC-ND (V1, see matrix).

### 2.5 Marine X-band radar

- **Commercial systems.** WaMoS II (Rutter Sigma S6) is reported at about 0.30 m (20 %)
  H_s accuracy against buoys, with correlation 0.89 (V2, vendor and early papers).
  Miros Wavex is a DNV-certified "virtual sensor" on standard navigation radar
  (V2, vendor).
- **Deep learning.** Kwon, Chang & Yang (2023), *PLOS ONE* 18:e0292884: a 3-D
  CNN on 72,180 radar image sequences from Sokcho, Korea (June–August 2021)
  against buoy H_s. RMSE 0.358 m and R = 0.970, compared with R = 0.665 for
  the SNR method (V1). **The data are public on figshare**
  (doi:10.6084/m9.figshare.24242791).
- **Synthetic training.** Synthetic X-band radar data train spatio-temporal
  deep networks for H_s estimation and prediction (*Ocean Eng.* 2025, S).
- **Relevance.** Radar is the closest mature analogue to camera-based
  sensing: an image sequence of the sea surface, calibrated against buoys. Its
  evaluation practice (buoy comparison over months, reported bias and RMSE)
  is the standard camera papers should meet.

### 2.6 Satellite remote sensing

| Sensor / product | Wave information | Level |
|---|---|---|
| Sentinel-1 SAR Wave Mode (OCN) | swell spectra; **NN total H_s since 2022-06-07** (Quach et al. 2021, *IEEE TGRS* 59:1859–1867; online 2020) | V2 |
| WV-Net (arXiv:2406.18765) | self-supervised foundation model on about 10 M S-1 WV images | S |
| Sentinel-2 MSI sun glitter | directional spectra from glitter brightness gradients (Kudryavtsev et al. 2017, *JGR* Parts 1–2) | V2 |
| SWOT KaRIn | wide-swath H_s. Against buoys, bias 0.11 m and RMSE 0.43 m (about 0.25–0.35 m near nadir) | V2 |
| CFOSAT SWIM | directional spectra for 70–500 m wavelengths (L2SPEC; AVISO, registration required) | V2 |
| ESA Sea State CCI | multi-mission altimeter H_s (L2P, L3, L4; v3 covers 2002–2021; CEDA DOIs) | V2 |

Satellites provide **context and independent labels offshore**, but revisit
times (hours to days) rule them out as per-image labels for shore cameras.

### 2.7 Lidar and in-situ references

The FRF at Duck runs a pier-mounted **lidar wave-gauge array** (x = 80–140 m)
and a runup lidar (2015–2026), alongside AWACs, pressure arrays, and Waveriders
(Q: THREDDS catalogue). This is the densest public nearshore wave record,
though the Argus imagery is not currently public (§5.1).

---

## 3. Model trends and what transfers

| Trend | Evidence | Implication here |
|---|---|---|
| CNN → ConvLSTM/BiLSTM → video transformers | Choi 2020 → Kim 2023 → Kamagata 2026 and 2606.13302 (LtViViT best) | Temporal context helps where video exists. NDBC buoy cameras are single frames, so snapshot models plus multi-view fusion apply there |
| **Self-supervised video backbones** | V-JEPA (Bardes et al. 2024), used by Kamagata 2026; **V-JEPA 2** (arXiv:2506.09985, >1 M h video pretraining, V2) | Freeze and probe first (linear/MLP heads) as a data-efficient baseline before fine-tuning |
| **Sim-to-real pretraining** | 2606.13302 (Airy synthetic clips), stereo in synthetic scenes (S), synthetic radar (S), synthetic marine perception (arXiv:2609.20680, S) | Physically parameterised synthetic sea surfaces (JONSWAP + cos-2s from `physics.spectra_models`) can pretrain spectral heads. Evaluate strictly on real, instrument-labelled data |
| **Physics-guided losses** | Kamagata 2026 (fixed-steepness prior, V1); 2606.13302 (band-limited FFT penalty, V1) | Prefer priors that permit the observed range (steepness band, ordering, spectral consistency; `models/losses.py`) over point constraints |
| **Earth-system foundation models** | Aurora (*Nature* 2025) fine-tuned for ocean waves, matching or beating operational wave forecasts in 86 % of comparisons (V2); ECMWF AIFS surface-ocean work (arXiv:2604.25559, S) | Model nowcasts are strong priors: learn **residuals** from imagery; report skill relative to the nowcast |
| **LLMs for H_s** | "Orca" (arXiv:2407.20053, S) | Low relevance |

**Recurring evaluation weaknesses** (reinforcing G-7):
- in-sample or random-split metrics;
- held-out sets of a few scenes;
- labels without instrument provenance;
- no model-nowcast or persistence baseline;
- NRMSE normalisers that are not stated.

---

## 4. Industry and operational use

| Actor | What they do | Validation published? | Level |
|---|---|---|---|
| **Surfline** (Smart Cams, SurfZone AI, Coastal Intelligence) | CV on a network of about 1,400 cameras: wave height, wave counts, ridden/unridden, crowds. Reported archive of 24 B frames per year. Patent-protected pipeline | **None found** against instruments | V2 (vendor pages, trade press) |
| **Rutter (WaMoS II), Miros (Wavex)** | X-band radar wave and current sensing on ships and platforms. Wavex is DNV-certified | Yes (buoy comparisons in the literature) | V2 |
| **Sofar Ocean** | Spotter drifting and moored buoys. Spectra are assimilated into operational forecasts (Houghton et al. 2022, *GRL*, S). Open archive 2019–03/2022 | n/a (instrument) | V2 |
| **WebCOOS** (SECOORA / IOOS) | Public-sector webcam network with ML products (rip currents, overwash, flooding, beach use). API with token | Products documented. No wave-height product found | V2 / Q |
| **USGS CoastCams / HIVIS** | Calibrated coastal cameras: shorelines, runup, bathymetry. Images via the NIMS API and S3 | Peer-reviewed products | Q |
| **SOCIB** (Balearic Islands) | SIRENA 5-camera beach monitoring at Cala Millor since 2011, with wave and water-depth measurements | Sci. Data 2023 | S/V2 |
| **Microsoft (Aurora)** | Foundation-model wave forecasting | Nature 2025 | V2 |

**Observation.** Commercial camera analytics exist at scale, but their claims
cannot be verified. Radar and satellite products are validated. An **open,
instrument-referenced, cross-site camera benchmark** would be a real
contribution.

---

## 5. Public data for training and evaluation

Registry entries (`data/registry/datasets.yaml`) record access, licence, and
verification status for each source below.

### 5.1 Tier 1: imagery paired with in-situ wave instruments

| Source | Imagery | Instrument and distance | Period / volume | Access and licence | Level |
|---|---|---|---|---|---|
| **NDBC buoy cameras** (this project's archiver) | 6-view strips, hourly day and night | Same hull: directional spectra (NDBC) | Prospective from 2026-09-26. About 1,900 images a day. NDBC keeps superseded images for about **72 h** | Public domain | Q |
| **PacIOOS beach cameras, Waimea Bay** (`beachcam_003`, `beachcam_004`) | Oblique RGB of the beach and of the bay offshore, hourly 06/07–17 HST | **CDIP 106 (WMO 51201) at 6.3 km**, record 2001–2026 | 2009-02 to 2013-10. 8,935 + 8,919 images | ERDDAP `files/`. Free reuse and redistribution with acknowledgement | **Q** |
| PacIOOS beach cameras, Waikīkī (`beachcam_001`, `beachcam_002`) | Oblique and nadir | Nearest south-shore Waverider to be assessed | 2009–2013. 15,564 + 15,590 images | as above | Q |
| **USGS CoastCams** (14 sites: FL, PR, MA, CA, AK, Guam) | Calibrated snapshot, timex, variance, and timestacks in data releases. Hourly daylight snapshots in the NIMS S3 archive since 2023 (e.g. Sand Key 11,911; Madeira 15,875; Dorado 7,905 images) | CDIP 214 (26–40 km, Tampa Bay sites), CDIP 156 (18–24 km, Monterey Bay), CDIP 221 (31 km, Cape Cod), CDIP 121 (15 km, Guam). **Dorado: a USGS Spotter ~2 km offshore** (geoport ERDDAP, 2022-10 to 2024-12) | Releases: Madeira 2017–2022, Head of the Meadow 2019–2025, Marconi 2021/2023/2024 | Public domain. The NIMS "overlay" images carry burned-in text (mask it) | **Q** / V2 |
| **SOCIB Cala Millor** (Mallorca) | SIRENA, 5 cameras. Snapshot, timex, variance, timestack from the first 10 min of each daylight hour, 7.5 fps | Hourly wave and depth measurements (MOBIMS) | 2011 onward | Beamon app. Licence to confirm | V2 (Sci. Data 2023) |
| **WebCOOS** | Still images and video archive (SE US) | CDIP 200/150 (25–47 km, Oak Island); CDIP 433/430 (22–25 km, Currituck and Duck) | Multi-year | API token (the API returned 403 without one); licence to confirm | Q (API) / V2 |
| **Guimarães et al. 2020 stereo** | Stereo video, 10–15 Hz | Wave wires, AWAC, X-band radar; stereo η itself | 12 × ~30 min | CC BY 4.0 | V1 |
| **Buscombe et al. 2020 OWG** | IR (Duck), RGB orthomosaics and oblique (Santa Cruz) | Instrument-derived H, T | CSV training sets on GitHub | Repository licence unclear | V2 |
| **Cienfuegos et al. 2025** (Zenodo 17192781) | Fixed camera (2 sessions, rectified) and UAV video | Bottom pressure sensors, ADVs | 2023-08-01/02, 6.4 GB | CC BY 4.0 | V2 |
| **Salin 2023** (Zenodo 7942014) | Video of wind waves on Lake Ladoga | Spar buoy in frame | 4.6 GB | CC BY 4.0 | V2 |
| **Kwon et al. 2023** (figshare) | X-band radar sequences (radar, not camera) | Buoy H_s | 72,180 images | figshare | V1 |
| USACE FRF, Duck | Argus imagery **not found** on the public THREDDS (`argus_temp/` empty). The legacy OSU CIL FTP host no longer resolves. North Head Argus merged products (2018) exist | Waveriders, AWACs, 8-m array, lidar wave gauges (to 2026) | Instruments: decades | THREDDS, public | Q |

### 5.2 Tier 2: imagery without instrument labels (pretraining, pseudo-labels)

| Source | Content | Licence | Level |
|---|---|---|---|
| KOLOMVERSE (Nanda et al. 2024, *IEEE T-ITS* 25:20832) | ~2.15 M 4K frames from 5,845 h of video in 21 Korean sea areas, with object labels | CC BY 4.0 | V2 |
| USGS NIMS S3 (`usgs-nims-images`) | Hourly CoastCam snapshots (above) plus 1,300+ hydrologic cameras | Public domain | Q |
| Stringari 2021 deepwaves | Labelled breaking and non-breaking crops (10k and 20k training sets, 1k test set) | BSD-3 (code) | V2 |
| Yin 2025 VWBT-9000 | 9,000 breaking-type clips | CC BY-NC-ND | V1 |
| Vorkapic 2024 | Bridge-camera Beaufort dataset | not stated | V2 |
| Salin 2023; Mac Conamhna (Zenodo 4295611, storm wave impacts) | Wave video | CC BY | V2/S |

### 5.3 Tier 3: labels without imagery (references, baselines, context)

| Source | Use | Level |
|---|---|---|
| **CDIP** (99 active stations; THREDDS, `latest_3day.nc` carries station metadata) | Primary **nearshore** reference for shore cameras (full a1–b2 directional moments) | Q |
| NDBC (1,797 stations; 127 with full directional spectra) | Offshore reference; buoy-camera labels | Q |
| Sofar Spotter Archive (AWS `sofar-spotter-archive`, 2019 to 2022-03) | Hourly spectra and bulk parameters from drifting Spotters. Sofar data-access agreement | V2 |
| Copernicus Marine in-situ waves (INSITU_GLO_WAV_*_013_045) | Global buoy aggregation including spectra; updated twice a year | S/V2 |
| **USACE Wave Information Study** (hourly hindcast along all US coasts, 1980–2024; 2-D spectra) | **Nowcast baseline** at each camera site | V2 |
| ERA5, WAVERYS, WW3 | Model baselines and priors | registry |
| ESA Sea State CCI, SWOT, CFOSAT, Sentinel-1 WV | Independent offshore validation | V2 |

**Nearest wave references for public camera sites** (computed 2026-09-26 from
`data/registry/stations.parquet` and CDIP `latest_3day.nc` metadata). The machine-readable
form is `data/registry/camera_sites.yaml`. The dashboard recomputes the distances from
NDBC station-table positions, which can differ from these by 0.1–0.2 km:

| Camera | Nearest CDIP (WMO), km | Nearest NDBC directional buoy, km |
|---|---|---|
| PacIOOS Waimea Bay | **106 (51201), 6.3** | 51003, 384 |
| USGS Dorado, PR | 249 (41121), 43.9, plus on-site USGS Spotter | 41043, 327 |
| USGS Madeira Beach / Sand Key, FL | 214 (42098), 26.5 / 39.9 | 42036, 175–185 |
| USGS Head of the Meadow / Marconi, MA | 221 (44090), 31 | 44018, 18–38 (through 2024) |
| USGS Santa Cruz / Sunset SB, CA | 156 (46236), 23.5 / 17.7 | 46042, 39–53 (through 2023) |
| USGS Adelup Point, Guam | 121 (52200), 15.3 | none within 5,000 km |
| WebCOOS Oak Island, NC | 200 (41108), 24.5 | 41013, 64 |
| WebCOOS Currituck, NC | 433 (44056), 21.6 | 44014, 93 |
| FAU Harbor Branch, Fort Pierce, FL | **134 (41114), 14.6** | 41009, 110 |

Distance is not the only criterion. Depth, exposure, and sheltering determine
whether a buoy represents the camera's field of view (Q-M2). Shoaling and
refraction between a 20–40 km buoy and the surf zone are first-order effects.
For this reason, **offshore H_s labels for surf-zone imagery should carry a
transformation model or be evaluated as offshore-condition estimation**, and
never presented as a local measurement.

### 5.4 Tier 4: synthetic data

No public synthetic camera wave dataset with physical labels was found. The
project can generate one: sea surfaces from `physics.spectra_models` (JONSWAP ×
cos-2s, random phases) rendered with sky reflection and Fresnel optics. Such data
serves pretraining and controlled ablations only (§3).

---

## 6. Where this project is distinctive

1. **Instrument-traceable labels with estimator provenance** (H_m0 vs H₁/₃,
   sensor, Δt, distance). No reviewed camera paper provides this.
2. **Cross-site evaluation as the primary protocol**, with leakage audits.
3. **The only systematic buoy-camera archive** paired with directional spectra
   from the same hull. NDBC keeps images for only about 72 h and nobody else archives them.
4. **Spectral targets**, validated against stereo (Guimarães 2020) where
   geometry is known.

---

## 7. Recommendations (feed into the roadmap)

1. **v0.2: build the Waimea benchmark first.** PacIOOS `beachcam_003/004`
   paired with CDIP 106 gives about 17,900 images from 2009–2013, a Waverider
   6 km away, a two-camera site (a within-site cross-view test), and big-wave
   North Shore conditions. It is available now and needs no collection.
2. **v0.2: USGS CoastCam adapter.** Use the NIMS S3 listing for snapshots and the
   data releases for IO/EO calibration. Pair with CDIP 214, 156, 221, and 121,
   and with the Dorado Spotter. Five geographically distinct sites give a genuine
   leave-site-out test.
3. **v0.2: WebCOOS.** Register for a token and confirm the licence. Pair Oak
   Island and Currituck with CDIP.
4. **Required baselines for every experiment:** climatology, persistence of
   the nearest buoy, **WIS/WW3/ERA5 nowcast at the site**, and propagated
   offshore buoy. Report camera-model skill *relative to the nowcast*.
   Consider residual targets (buoy − nowcast).
5. **Buoy-camera physics features (H3):** per-view horizon tilt (six-view plane
   fit) and whitecap fraction, combined with the buoy's own measured wind.
   Test these before end-to-end deep models.
6. **Pretraining:** freeze-and-probe V-JEPA 2 or DINO-family backbones on
   KOLOMVERSE, WebCOOS, and NIMS imagery, then fine-tune on Tier 1 data.
   Synthetic pretraining only with real-data evaluation.
7. **Physics validation:** process the Guimarães stereo records into spectra
   and verify that our spectral pipeline reproduces the published statistics.
8. **Self-collected site:** CDIP 134 at Fort Pierce is 14.6 km from FAU
   Harbor Branch. A camera (RGB plus radiometric thermal) viewing the buoy's
   wave climate would fill G-5 (Q-D5).

---

## 8. Sources

Core papers and standards are in [bibliography.bib](bibliography.bib). The
entries added by this review are keyed below. Search-result-only items (S) are
listed separately so that they cannot be mistaken for reviewed evidence.

**V1 / V2 (read):** Kamagata2026Tp (arXiv:2606.13302) · Guimaraes2020 ·
Kwon2023 · Buscombe2020 · Quach2021 · Vorkapic2024 · Stringari2021 ·
BuscombeCarini2019 · Brumer2017 · SchwendemanThomson2015a/b ·
HolmanHaller2013 · Holman2013cBathy · Almar2012 · Kudryavtsev2017 · Bodnar2025 (Aurora) ·
Assran2025 (V-JEPA 2, arXiv) · Nanda2024 (KOLOMVERSE) · CienfuegosZenodo2025 · SalinZenodo2023.
DOIs and author lists were checked against Crossref on 2026-09-26.

**Data services queried (Q):** NDBC image retention (`/images/buoycam/`),
PacIOOS ERDDAP `beachcam_00{1–4}`, USGS NIMS API and `usgs-nims-images` S3,
USGS geoport ERDDAP `waves-Dorado`, USACE FRF THREDDS (`chldata.erdc.dren.mil`),
CDIP THREDDS (`106p1`, `latest_3day.nc`), WebCOOS API (403 without token),
UW SWIFT public data listing (processed products, no images).

**Screened only (S), to read before citing:**
- Ocean Dynamics (2025) timestack wave parameters, doi:10.1007/s10236-025-01743-x
- *Remote Sensing* 18:1385 (2026) UAV waterline, doi:10.3390/rs18091385
- *Remote Sensing* 17:2274 (2025) webcam depth inversion, doi:10.3390/rs17132274
- *Ocean Eng.* (2024) three-stage monocular offshore extraction
- *Int. J. Nav. Archit. Ocean Eng.* (2023) CNN wave detector + PPM
- arXiv:2512.06024; arXiv:2609.20680; arXiv:2406.18765 (WV-Net); arXiv:2407.20053 (Orca); arXiv:2604.25559 (AIFS ocean)
- DSGP-Net (*Ocean Eng.* 2026); floating-platform synthetic stereo (*Ocean Eng.* 2025); synthetic X-band DNN (*Ocean Eng.* 2025)
- *Acta Oceanol. Sin.* (2024) T_p from texture motion
- Lidar-buoy pitch/roll period (*Sensors* 2021); horizon roll/pitch (Springer 2014)
- Wave-buoy-analogy ML papers (*Ocean Eng.*, *Appl. Ocean Res.*)
- Houghton et al. 2022, *GRL* (Spotter assimilation); Carini et al. 2021, *JGR Oceans*
- Review: "Non-contact sensing methods for wave measurement" (*Intell. Transp. Infrastruct.* 2025, doi:10.1093/iti/liaf003)

**Discrepancy noted.** NDBC's BuoyCAM FAQ states that cameras "typically take photos only
during daylight hours". On 2026-09-26 we observed hourly images posted at night
(for example 41010 at 01:10 UTC, fully dark). The archive keeps them; `illumination`
annotation handles them.
