# Camera-based ocean-wave measurement: published state of the art by measurement (as of 2026-11-19)

**How to read the evidence tags.** Every number below carries one tag.

| Tag | Meaning |
|---|---|
| **FT** | Full text read in this session (PDF or HTML from the publisher, arXiv, or an author/agency repository). |
| **ABS** | Abstract only (publisher record, repository record, or Semantic Scholar/Crossref abstract field). |
| **META** | Title, authors, venue only. The abstract was not reachable (Elsevier/ScienceDirect and SSRN were not accessed, per instruction; Springer redirected to a login). **No result is known for these.** |
| **SEC** | Secondary source: a number quoted from another paper's text or table. |
| **SNIP** | Seen only in a web-search result summary. **Unverified**; read before citing. |
| **PROJ-V1** | Taken from the project's own full-text notes (`docs/literature/benchmark_targets.md`, `papers/*.md`); not re-read here. |

Method notes: the work ran in two passes on 2026-11-19. In the first, web search ran out part-way (a shared limit), so discovery used citation lists (Semantic Scholar, by DOI) for Choi 2020, Buscombe 2020, Xu 2024, Yang 2026, Yin 2025, Stringari 2021, Sáez 2021, Collins 2021, Buckley 2024, Almar 2021, WASS 2017, cBathy 2013, plus direct reading of open PDFs. The second pass added about 30 targeted searches (wind from imagery, buoy cameras, surf cams, timestack CNNs, polarimetry, Λ(c), setup, learned optical flow, foundation models, public datasets, monocular-from-stereo), OpenAlex abstracts for some Elsevier papers, and the Kim 2023 citing list. What is still thin is listed under each Gaps heading.

---

## Key question 1. For each measurement, what are the leading published camera/video methods, their reported error, reference instrument, and train/test split?

### Takeaway
The lowest published camera errors for wave height and period (Buscombe 2020: 0.08–0.14 m, 0.41–0.81 s; Xu 2024: MAE 0.04 m) all come from **one site and a random split of images taken on the same days**; every instrument-referenced result on a temporal or site hold-out is in bathymetry, runup, rip currents or currents, not in height, period, direction or spectra. Metric wave fields (elevation, spectra, crest statistics) are still measured only by stereo, lidar or nadir UAV methods, with stereo H_s agreeing with gauges and buoys to about 0.10–0.14 m.

### Cited Findings

#### 1a. Significant wave height (offshore, nearshore, breaking)

| Work (year, venue) | Input → target | Reference instrument | Split | Reported result | Tag |
|---|---|---|---|---|---|
| Buscombe et al. 2020, *Coastal Eng.* 155:103593 | Single image, 128×128 grayscale, CNN regression. (i) IR close-range, Duck: 9,400 images from 10.5 h on 7–8 Nov 2016, **individual** wave height and period; (ii) RGB orthomosaic and (iii) oblique RGB, Sunset State Beach CA: 980 images, 6 Dec 2017–30 Jan 2018, H_s and T_p | (i) Riegl VZ-400 lidar along a profile in the IR field of view; (ii, iii) Nortek Signature 1000 on a tripod in 12–14 m depth just offshore (H_s 0.39–2.56 m, T_p 7–23 s) | Top and bottom 5 % of the target withheld as "out-of-calibration"; of the middle 90 %, **random 60 % train / 40 % validate** (54 % / 36 % of all). Reported RMS is on that 40 %, which is also the set used for early stopping; no separate test subset is described. Same days in both halves | IR 0.14 m / 0.41 s; rectified RGB 0.08 m / 0.79 s (out-of-calibration 0.14 m / 3.44 s); oblique RGB 0.11 m / 0.81 s | **FT** — [author manuscript](https://par.nsf.gov/servlets/purl/10201658); [USGS record](https://pubs.usgs.gov/publication/70208193) |
| Xu et al. 2024, *JMSE* 12:2003 | One shore camera, Qingdao, 1 frame/min | Waverider nearby, hourly, interpolated | Random 6:2:2 of frames | MAE 0.04 m with met and tide inputs; image-only correct rate 57.7 % | **PROJ-V1** — `docs/literature/benchmark_targets.md` |
| Kim et al. 2023, *Ocean Eng.* 288:116002 | Fixed camera on a WEC, 5 height classes | Seafloor AWAC | Random frames/clips | 78.3 % snapshot, 93.2 % video | **PROJ-V1** |
| Choi et al. 2020, *Ocean Eng.* 201:107129 | NDBC buoy-camera stills (2 classes); ship video (regression) | Same buoy; shipboard wave radar | Same buoys and days in both | 84.05 %; RMSE 0.133 m (ship) | **PROJ-V1** |
| Kamagata et al. 2026, arXiv:2607.11998 | 16-frame shore clips, V-JEPA + SlowFast | Not stated | 5 held-out videos at other sites | H_s RMSE 0.586 m, R² 0.107 | **PROJ-V1** |
| Liu, Qin & Ti 2026, *Ocean Eng.* 352:124561 | Monocular video, 3D-CNN, pretrain on simulated waves then fine-tune on "a small set of in-situ measurements", POD denoising; H_s and T_p | Not in abstract | Not in abstract | "High accuracy" only; no numbers. Full text embargoed in the repository until 2028-04-15 | **ABS** — [PolyU record](https://ira.lib.polyu.edu.hk/handle/10397/118413?mode=full) |
| Y. Liu, G. Xu et al. 2026, *Ocean Eng.* 366:127543, "Nearshore wave height estimation using coastal cameras: comparing physics-informed spectral and conventional image representations" | Coastal cameras → wave height | Unknown | Unknown | Unknown | **META** — [DOI](https://doi.org/10.1016/j.oceaneng.2026.127543) |
| Ngene et al. 2026, *Appl. Ocean Res.* 173:105161, "Swell-dominated wave parameter retrieval from oblique coastal video under variable camera geometry" | **Not learned.** One oblique camera, no ground control: vanishing-point geometry, sun-angle shadow correction, spectral decomposition of pixel brightness → H_s and T_p of shoaling swell | Co-located pressure sensor at Praia de Mira, Portugal (UAV and fixed camera); a second site (Port of Sines, about 1,700 m standoff) has no sensor | n/a | T_p MAPE below 10 %; the shadow-based H_s estimate beat the elevation-reconstruction one by 5.9 % relative error (absolute H_s error not in abstract); at Sines H_s was "robust" mainly above 1.5 m | **ABS** — [DOI](https://doi.org/10.1016/j.apor.2026.105161) (abstract from the [OpenAlex record](https://api.openalex.org/works/doi:10.1016/j.apor.2026.105161)) |
| Egan, undated Stanford course report (data from 2020), "Wave Statistic Estimation from Surfline Video Data Using ConvNets" — **not peer reviewed** | Surfline camera at Scripps Pier; 1,600 ten-minute clips from 20 days (17 Apr–7 May 2020), downsampled | Pier pressure sensors, hourly, upsampled to 10 min | **Random 80 / 10 / 10 of clips** from the same 20 days | ConvNet test RMS 0.25 m / 1.18 s (r² −0.10 / −5.88); ConvLSTM test RMS 0.17 m / 0.4 s (r² 0.51 / 0.21) | **FT** — [report PDF](https://galenegan.github.io/files/final_report.pdf) |
| Hwang et al. 2023, *Int. J. Nav. Archit. Ocean Eng.* (article 100542) | CNN "wave detector" on ocean images + pixels-per-metre calculator → wave height | "Two types of wave height measurement sensors" | Not in abstract | "Highly consistent"; no numbers in abstract | **ABS** — [DOI](https://doi.org/10.1016/j.ijnaoe.2023.100542) |
| Ahammed & Dang 2026, *Ocean Eng.* 367:127753, "Polarization-aware multi-scale vision transformer for significant wave height estimation" | Unknown (cites Choi 2020 and Yang 2026) | Unknown | Unknown | Unknown | **META** — [DOI](https://doi.org/10.1016/j.oceaneng.2026.127753) |
| Li et al. 2026, *Ocean Eng.* 364:126780, "Multi-domain deep learning architectures for vision-based sea wave parameter estimation utilizing reconstructed 3D datasets" | Unknown | Unknown | Unknown | Unknown | **META** — [DOI](https://doi.org/10.1016/j.oceaneng.2026.126780) |
| Zhang & Li 2024, ISAIMS proceedings (ACM); related *Ocean Eng.* 313:119356 (META) | Monocular video from offshore infrastructure incl. night IR; ConvLSTM + VGG + self-attention memory → H_s and peak frequency | Not in abstract | Not in abstract | MAPE 3.15 %, R² 98.84 % | **ABS** — [ACM DOI](https://doi.org/10.1145/3669721.3669739) |
| Venkateswarlu et al. 2025, *Ocean Dynamics*, timestack wave parameters | Shore video timestacks, no learning | Unknown here | Unknown here | The project's screened figures (H_s RMSE 0.12 / 0.09 m) remain unverified | **META** — [DOI](https://doi.org/10.1007/s10236-025-01743-x) |
| Andriolo et al. 2020, *Remote Sens.* 12:204 | Timex images from a surfcam → **breaking** wave height from the cross-shore length of the foam signature | Method 1 against **modelled** breaking height; method 2 against "different wave sources" at five sites | Not a learned method | RMSE 0.2 m for breaking heights 1–3 m over two days; normalised RMSE 18 % over about 900 records, 0.1–3.8 m | **ABS** (from full-text PDF) — [MDPI](https://www.mdpi.com/2072-4292/12/2/204) |
| Salatin et al. 2024, *Coastal Eng.* 192:104542 (V-BeachNet) | Video segmentation of the swash edge → swash spectra, significant swash height, setup | Scanning lidar on one transect | Network trained on 16 hand-segmented images; four 1-h cases | Vertical swash RMSE vs lidar 0.33, 0.27, 0.15, 0.14 m | **FT** — [NSF PAR](https://par.nsf.gov/servlets/purl/10513109) |

- Laboratory-only learned height estimators exist (wave-flume video with domain adaptation; correlation 0.98/0.90 regular and 0.95/0.85 irregular waves against wave gauges) — **ABS**, [Kim et al. 2021, *Sci. Rep.*](https://doi.org/10.1038/s41598-021-01157-x).
- A "Fusion-Wave" dataset of real infrared wave videos plus simulated clips, labelled with wave height and period, was described in 2025; the abstract gives no accuracy numbers and no availability statement — **ABS**, [Yang et al. 2025, ICAIRC](https://doi.org/10.1109/ICAIRC68035.2025.11385244).
- Two action cameras as a low-cost stereo pair in shallow water and the surf zone: "very small deviations" in H_s against a co-located pressure gauge; no numbers in the abstract — **ABS**, [Vieira et al. 2025, *Coastal Eng.* 197:104694](https://doi.org/10.1016/j.coastaleng.2024.104694).
- Classical breaker-height estimator of Almar et al. 2012 (*Coastal Eng.* 61:42–48): 9 % mean and 18 % RMS error on individual breaker heights, validated on a **wave-basin** dataset — **SNIP** (search summary of the HAL record; the record itself refused access).
- Two further non-learned video methods were seen only in search summaries: pixel-intensity amplitude spectra against a buoy (maximum RMSE 0.37 m, MAPE 16 %, H_s 0.5–3.6 m; *SN Computer Science* 2021, doi:10.1007/s42979-021-00727-0) and a ground-control-free timestack method for estuaries (errors under 6 % against a sonic wave gauge; *Estuarine, Coastal and Shelf Science* 2020) — **SNIP**.
- Searches for other work on NDBC buoy-camera images returned nothing besides Choi 2020, and searches for a published instrument validation of a commercial surf-camera wave-height product returned only the Egan course report above (see Gaps).

#### 1b. Peak, mean and zero-crossing period

- OWG period RMS: 0.41 s (IR, individual waves, lidar), 0.79 s (rectified RGB, T_p), 0.81 s (oblique RGB); out-of-calibration period error rises to 3.44 s — **FT**, [Buscombe 2020](https://par.nsf.gov/servlets/purl/10201658).
- Kamagata 2026 held-out: T_p RMSE 1.42 s, T_z 0.98 s; Kamagata (arXiv:2606.13302) T_p RMSE 0.80 s on 4 held-out scenes with expert and pseudo-labels — **PROJ-V1**.
- Smartphone stereo at Leme beach, Brazil, reports "perfect estimation of the peak period" against a pressure sensor at 8.3 m depth — **ABS** (from full-text PDF), [Vieira et al. 2020, *JMSE* 8:831](https://www.mdpi.com/2077-1312/8/11/831).
- Runup-timestack timing on a dissipative Gulf of Mexico beach: T_m-1,0 RMSE 3.88 s (colour contrast) and 3.51 s (CNN) against hand-digitised waterlines — **ABS**, [van der Grinten 2024, TU Delft thesis](https://repository.tudelft.nl/record/uuid:8882db80-2627-4762-89ab-19daaeb163a0).
- UAV video + DeepUNet waterline → swash-zone wave period and spectra; compared only "qualitatively" with a buoy 2.7 km offshore in 18 m depth — **FT** (highlights and methods), [Chuang et al. 2026, *Remote Sens.* 18:1385](https://www.mdpi.com/2072-4292/18/9/1385).
- Low-grazing-angle single camera, no learning: sea-surface texture motion by image matching, then spectral analysis; T_p deviates by at most 8 % from an X-band radar inversion and at most 14 % from the period obtained by tracking a buoy's motion in the same video (mean 5 % in a second region); H_s by a pinhole model is "preliminary" — **FT**, [Yu et al. 2024, *Acta Oceanol. Sin.* 43(9):136–144](https://castjournals.cast.org.cn/joweb/aos/EN/PDF/10.1007/s13131-024-2359-y).
- Geometry-and-spectrum single-camera retrieval: T_p MAPE below 10 % against a co-located pressure sensor — **ABS**, [Ngene et al. 2026](https://doi.org/10.1016/j.apor.2026.105161).
- ConvLSTM on surf-camera clips, random split within 20 days: period RMS 0.4 s but r² only 0.21 on the test set — **FT**, [Egan course report](https://galenegan.github.io/files/final_report.pdf).

#### 1c. Wave direction and directional spread

- Kamagata 2026 held-out direction RMSE 36.98°, bias +17.9°, linear (not circular) statistics, labels untraceable — **PROJ-V1**.
- Bayesian Directional Method on rectified pixel brightness (each pixel treated as a wave probe) gives directional spectra compared with a SWAN model; "good agreement", no numbers in the abstract, no instrument — **ABS**, [Zikra et al. 2015, *Jurnal Teknologi* 74(5)](https://journals.utm.my/jurnalteknologi/article/view/4651).
- UAS image sequences + snapshot POD + 2-D autocorrelation give wavelength, propagation angle and period of the dominant wave; the abstract reports better signal-to-noise than an FFT, no instrument error — **ABS**, [Bheeroo et al. 2024, *Exp. Fluids*](https://doi.org/10.1007/s00348-024-03922-w).
- "High-accuracy wave direction measurement using monocular vision on UAV platforms" — **META**, [Guo et al. 2026, *Ocean Eng.* 344:123659](https://doi.org/10.1016/j.oceaneng.2025.123659).
- Stereo with deep stereo matching (PSM-Net) and SIFT tracking: direction error 0.24–1.26°, height within 2 %, period within 4 %, **indoor motion platform only**, no sea data — **FT**, [Wu et al. 2024, *Front. Mar. Sci.* 11:1508233](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2024.1508233/full).
- The same method was first presented at Hasaki Beach, Japan, with results "compared with a numerical model and field measurements"; the abstract gives no numbers and does not name the instruments — **ABS**, [Zikra et al. 2012, *Coastal Engineering Proceedings* 33](https://icce-ojs-tamu.tdl.org/icce/article/view/6555).
- No paper was found that validates camera-derived **directional spread** against a buoy; a 2024 Hasaki study reporting that the peak direction is little changed by breaking while the spread broadens over the sandbar was seen only in a search summary, and its page returned 404 — **SNIP**.

#### 1d. One- and two-dimensional spectra

- Regularised inversion of nadir UAV video spectra to the 3-D wavenumber–frequency elevation spectrum (contrast assumed proportional to steepness); bulk parameters and spectra "in good agreement" with buoys and bottom-mounted acoustic sensors across several sea trials; no numbers in the abstract — **ABS**, [Solodoch et al. 2025, *IEEE TGRS* 63:4201914](https://doi.org/10.1109/TGRS.2025.3536378).
- Single-image methodology for surface elevation from drone and satellite images, compared with an ADCP; energy bias depends on the azimuth between view and crest, with a proposed correction — **ABS**, [Almar et al. 2021, *Remote Sens.* 13:679](https://www.mdpi.com/2072-4292/13/4/679).
- Smartphone stereo yields η(x,y,t), E(f) and E(f,θ); H_s RMSE about 11 % against a pressure sensor — **ABS** (from full-text PDF), [Vieira et al. 2020](https://www.mdpi.com/2077-1312/8/11/831).
- PIV on near-field surf-zone imagery reproduces wave-orbital velocity spectra at co-located ADVs with mean nRMSE 10 % (16 % for time series) — **ABS**, [McCormack et al. 2024, *Coastal Eng.* 195:104631](https://doi.org/10.1016/j.coastaleng.2024.104631).
- V-BeachNet swash spectra agree with lidar spectra (figure comparison, no scalar error) — **FT**, [Salatin et al. 2024](https://par.nsf.gov/servlets/purl/10513109).

#### 1e. Individual-wave height, maximum/rogue height, crest statistics

- The only learned individual-wave result found is OWG's IR set: lidar-measured individual wave height and period, RMS 0.14 m and 0.41 s, random split within 10.5 h of data — **FT**, [Buscombe 2020](https://par.nsf.gov/servlets/purl/10201658).
- Kamagata 2026 held-out H_max RMSE 1.035 m, R² 0.096 — **PROJ-V1**.
- Ship-mounted stereo on an icebreaker in the Southern Ocean recorded surface elevation and found heavy-tailed ("rogue sea") statistics during wave growth — **ABS**, [Toffoli et al. 2024, *PRL* 132:154101 (arXiv:2310.01841)](https://arxiv.org/abs/2310.01841).
- Ship-based stereo video: all-wave statistics "compare well" with buoys; 103 breaking-wave profiles show local steepness extremes near the crest — **ABS**, [Schwendeman & Thomson 2017, *JPO*](https://doi.org/10.1175/JPO-D-16-0187.1).
- Automated detection and tracking of individual whitecaps (area, speed, direction) on three stereo-camera datasets, evaluated in the Phillips Λ(c) framework — **ABS**, [Peach et al. 2025, *IEEE TGRS*](https://doi.org/10.1109/TGRS.2025.3555851).

#### 1f. Celerity, wavelength and depth inversion

| Work | Method and data | Split | Result | Tag |
|---|---|---|---|---|
| Holman, Plant & Holland 2013, *JGR Oceans* 118:2595 | cBathy (original): cross-spectral phase → frequency–wavenumber pairs → depth, Kalman-filtered | n/a (physics) | Duck, 16 surveys over 2 years: bias 0.19 m, RMS error 0.51 m | **ABS** — [USGS record](https://pubs.usgs.gov/publication/70118239) |
| Holman & Bergsma 2021, *Remote Sens.* 13:3996 | cBathy V1.0 vs V2.0 against 39 Duck surveys (2015–2019), 624 collections | n/a (physics) | Kalman-filtered bias / RMS / 95 % exceedance: V1.0 0.15 / 0.47 / 0.96 m; **V2.0 0.08 / 0.38 / 0.78 m**. Unfiltered phase 2: V1.0 0.19 / 0.64 / 1.27 m; V2.0 0.16 / 0.56 / 1.19 m | **FT** (abstract page) — [MDPI](https://www.mdpi.com/2072-4292/13/19/3996) |
| Brodie et al. 2018, *Coastal Eng.* 136:147 | cBathy at Duck against surveys and altimeters, H_s 0.3–4.3 m | n/a (physics) | RMSE 0.75 m, bias 0.26 m against surveys; depth overestimated by up to 2 m when H_s > 3 m; sandbar placed 18–40 m too far onshore when waves break on the bar | **FT** (abstract page) — [PDF](https://www2.whoi.edu/staff/elgar/wp-content/uploads/sites/153/2018/03/135.pdf) |
| Collins et al. 2020, *Remote Sens.* 12:3364 | FCNN on **synthetic** timex + snapshot imagery | Synthetic test set | RMSE 0.39 m | **ABS** — [MDPI](https://www.mdpi.com/2072-4292/12/20/3364) |
| Collins et al. 2021, *Remote Sens.* 13:4907 | FCNN, transfer from synthetic to Duck tower imagery + bulk wave statistics | **Train 2015–2018 (41 surveys, 3,036 images); test 2019 (11 surveys, 306 images)** | RMSE 0.37 m, bias 0.06 m, MAE 0.31 m, 90th-percentile error 0.79 m; waves up to 3.5 m | **FT** — [MDPI](https://www.mdpi.com/2072-4292/13/23/4907) |
| Lange et al. 2023, *Coastal Eng.* 185:104375 | Hovering-UAV video; a deep network marks individual crests in timestacks, surf-zone depth from crest celerity, joined to cBathy offshore (cBathyCT) | 8 test days, 112 transects, 17-min hovers, H_s 0.55–2.15 m; the split used to train the crest network is not in the abstract | Surf zone: mean RMSE 0.17 m, bias 0.06 m (after a heuristic nonlinear speed correction). Surf-zone edge to 10 m depth: RMSE 0.87 m, bias 0.58 m. (Salatin 2026, Table 3, quotes the range as 0.17–0.81 m) | **ABS** — [DOI](https://doi.org/10.1016/j.coastaleng.2023.104375) (abstract from the [OpenAlex record](https://api.openalex.org/works/doi:10.1016/j.coastaleng.2023.104375)) |
| Chen, Wang & Chen 2023, *Coastal Eng.* (article 104337) | Physics-informed network: celerity field + sparse wave heights → depth and wave-height field together | **Synthetic** cases only (barred beaches, circular shoal) | No field error; no scalar in abstract | **ABS** — [DOI](https://doi.org/10.1016/j.coastaleng.2023.104337) |
| Salatin, Elgar & Raubenheimer 2026, *JGR: Machine Learning and Computation* 3:e2025JH000906 | Physics-informed conditional generative model: PIV surface flows from video → surf-zone depth, Duck | 6 bathymetries (8 flow realisations) with random train/test of flow–depth pairs; one **unseen bathymetry** | In-distribution RMSE 0.18–0.50 m; unseen bathymetry 0.66 m (coverage 30 %) | **FT** — [PDF](https://www2.whoi.edu/staff/elgar/wp-content/uploads/sites/153/2026/01/166.pdf) |
| O'Dea et al. 2025, *Coastal Eng.* | cBathyShortDwell on 30–60 s Planet SkySat video at Duck | n/a | RMSE 0.50–0.59 m, bias −0.19 to 0.16 m | **ABS** — [ERDC record](https://doi.org/10.21079/11681/49956) |
| Andriolo et al. 2025, *Remote Sens.* 17:2274 | Webcam timestacks; separate shoaling and breaking celerity | Two days, rocky bottom | Bias 0.05 m, RMSE 0.43 m to 10 m depth | **FT** (abstract page) — [MDPI](https://www.mdpi.com/2072-4292/17/13/2274) |
| Goessling & Leon 2025, *Drones* 9:761 | Drone video (UBathy) vs spectral (Stumpf) | Two beaches | 0.22–0.41 m vs 0.30–0.71 m RMSE | **ABS** — [MDPI](https://www.mdpi.com/2504-446X/9/11/761) |
| Kim et al. 2026, *JGR Oceans* | Nonlinear depth inversion coupled to a spectral wave model | n/a | Linear inversion's depth overestimate reduced by up to 90 % | **ABS** — [DOI](https://doi.org/10.1029/2025JC023876) |
| LeClair 2025 (UW thesis); LeClair, Thomson & Hegermiller, *Coastal Eng.* 214:105194 | ML detection of breaking crests in timestacks at Duck → bore speed → depth | Not in abstract | "Low but systematic bias" in the surf zone; no numbers in the abstract | **ABS** (thesis) / **META** (paper) — [thesis](https://hdl.handle.net/1773/55181), [paper](https://doi.org/10.1016/j.coastaleng.2026.105194) |

#### 1g. Breaking detection, breaker type, whitecap coverage, dissipation

- Active-breaking classifier: balanced accuracy ≈ 90 % on the test set; 19,000 training and 1,300 test images cut from the Guimarães stereo records (Black Sea, Adriatic, Yellow Sea, La Jument), with **100 test samples drawn from each of the same records used for training** — **FT**, [Stringari et al. 2021, arXiv:2101.11478](https://arxiv.org/abs/2101.11478).
- The original Duck U-Net for wave-by-wave breaking matched "nearly 71 %" of the breaking instances found by a lidar–IR sensor-fusion reference; the authors judge it better than the reference in many cases by eye — **ABS**, [Sáez et al. 2021, *Coastal Eng.* (article 104021)](https://doi.org/10.1016/j.coastaleng.2021.104021).
- Wave-by-wave breaking identification with a U-Net trained at Duck, extended by transfer learning and human-in-the-loop labelling to the Maipo river mouth (Chile) and Palavas-les-Flots (France): accuracy 80–90 % depending on site; seven rectified Argus videos (3.3 GB, CC BY 4.0) released — **ABS**, [Sáez & Catalán, Zenodo record for *Coastal Eng.* 206:104954](https://zenodo.org/records/17956740).
- Breaker-type classification on the Duck IR data with pretrained CNN features plus optical flow cuts errors by up to 60 %, and the authors "provide corrections to a methodology in the literature from which the data originates to achieve a more accurate assessment of model performance" — **ABS**, [Smith et al. 2023, *Eur. J. Remote Sens.*](https://doi.org/10.1080/22797254.2022.2163707).
- Foam segmentation with DeepLabv3+ on a low-tide-terrace beach, **train/validation/test split by time period**: overall accuracy 96.4 %, foam 96.2 % — **ABS**, [Yin et al. 2024, *Remote Sens.* 16:4616](https://www.mdpi.com/2072-4292/16/24/4616).
- Surf detection during typhoons (YOLOv6n): 99.3 % mAP50 on 2,855 labelled images from five typhoons at one tide-gauge station — **ABS**, [Shi et al. 2025, *Remote Sens.* 17:1039](https://www.mdpi.com/2072-4292/17/6/1039).
- Shipboard active-whitecap fraction: horizon-based stabilisation and a U-Net, applied to 48 h of Gulf of Mexico cruise video; whitecap fraction vs wind speed "showed substantial variability and differed from previous research", with SST, swell and wave age as secondary factors; no segmentation score in the abstract — **ABS**, [Yang et al. 2025, *JTECH* 42:827](https://repository.library.noaa.gov/view/noaa/71822).
- Shipborne whitecap segmentation (EMA-SE-ResUNet), northwest Pacific fishery survey: whitecap IoU 73.32 %, F1 84.60, pixel absolute error 0.081 % — **ABS**, [Chen et al. 2025, *Electronics* 14:4286](https://www.mdpi.com/2079-9292/14/21/4286).
- Dissipation from video rests on the classical Λ(c) (breaking-crest length per unit area per speed) method — Kleiss & Melville 2011 (*JTECH*), Thomson et al. 2009 (*GRL*), Schwendeman et al. 2014 (*JPO*) — and a 2016 preprint notes that the three extraction methods in use assign different speeds to the same front and so give different Λ(c) — **SNIP** (search summary only; [arXiv:1602.06649](https://arxiv.org/abs/1602.06649) not opened). The only recent learned component found is the whitecap-track filter in [Peach et al. 2025](https://doi.org/10.1109/TGRS.2025.3555851) — **ABS**.
- Flume-only breaker-type classifiers: [Robertson et al. 2025, *Coastal Eng.*](https://doi.org/10.1016/j.coastaleng.2025.104902) and [Yun et al. 2026, *Appl. Ocean Res.*](https://doi.org/10.1016/j.apor.2026.105284) — **META**.

#### 1h. Runup, swash, total water level, setup

- More than 7,000 hand-digitised runup timestacks from six sites; a fully convolutional model trained from scratch is compared with a transformer (SegFormer) trained by transfer learning. Contoured at 50 % wet probability, both find runup maxima better than minima at every site, giving accurate R2% but **under-predicted significant swash and over-predicted setup**; weighting the contour by wet probability fixes most of this. The transformer agrees best on R2%, significant swash and setup, is "within the uncertainty range of hand-digitization" on a random subset, and runs in under 10 s per timestack against more than 5 min by hand. No scalar errors in the abstract — **ABS**, [Buckley et al. 2024, *Coastal Eng.* 193:104600](https://pubs.usgs.gov/publication/70257858) (full abstract from the [OpenAlex record](https://api.openalex.org/works/doi:10.1016/j.coastaleng.2024.104600)). The released weights are a model trained on Madeira Beach and Duck and a version fine-tuned on Waiakāne, with train/validation file lists but no stated proportions — **ABS**, [Zenodo 11167347](https://zenodo.org/records/11167347).
- Lidar (not camera) runup with an FCNN: 966 thirty-minute linescan series for training and validation, **99 separate series for test**; RMSD 0.02 m for the 3-cm-depth contour and 0.06 m for R2% against manual digitisation — **ABS**, [Collins et al. 2023, *IEEE TGRS*](https://doi.org/10.1109/TGRS.2023.3244488).
- CNN on colour-contrast timestacks, nine 1.5-h series cross-validated: waterline RMSE 0.10 m (r 0.96), R2% RMSE 0.09 m; the CNN "occasionally failed" on other datasets — **ABS**, [van der Grinten 2024](https://repository.tudelft.nl/record/uuid:8882db80-2627-4762-89ab-19daaeb163a0); journal version [*Coastal Eng.* 2025](https://doi.org/10.1016/j.coastaleng.2025.104757) is **META**.
- CCTV runup during the 1 Jan 2024 Noto tsunami (classical luminance-variance edge + land–water mask): RMSE 0.094 m and 0.191 m at two sites after removing periods under 180 s — **ABS**, [Shirai & Arikawa 2025, *Sci. Rep.*](https://doi.org/10.1038/s41598-025-28874-x).
- V-BeachNet setup estimates "agree" with lidar (no scalar) — **FT**, [Salatin et al. 2024](https://par.nsf.gov/servlets/purl/10513109).

#### 1i. Overtopping

- CCTV overtopping detector (CNN + filtering) in Korea: AP50 59.6 % at 70 fps, including night and severe weather — **ABS**, [J. Navig. Port Res. 46(3), 2022](https://koreascience.kr/article/JAKO202220257621387.pdf).
- CNN semantic segmentation of beach and back-beach flooding on the Texas coast: water-pixel accuracy about 90 % from over 500 labelled images — **ABS**, [*Earth Surf. Dynam.* 12:1, 2024](https://esurf.copernicus.org/articles/12/1/2024/).
- Semantic segmentation with added foam and wet-sand classes; overtopping is flagged when sea or foam reaches predefined critical areas; validated on "a dedicated overtopping dataset" with "improved reliability" over a generic baseline; no numbers in the abstract — **ABS**, [Sanfiel-Reyes et al. 2026, *Coastal Eng.* 211:105091](https://doi.org/10.1016/j.coastaleng.2026.105091).
- Laboratory overtopping volumes from image clustering, validated on focused-wave dike tests — **ABS**, [Formentin et al. 2024, *Phys. Fluids*](https://doi.org/10.1063/5.0207486).

#### 1j. Rip currents and surf-zone currents

- Faster R-CNN trained on 2,440 still images and tested on **23 separate videos (18,042 frames)**: average accuracy 0.884, 0.984 with frame aggregation, against 0.760 for human observers — **FT**, [de Silva et al. 2021, arXiv:2102.02902](https://arxiv.org/abs/2102.02902).
- RipVIS: 184 videos (212,328 frames), **video-level split 112 / 36 / 36**, inter-annotator κ 0.82 — **FT**, [Dumitriu et al., CVPR 2025](https://arxiv.org/abs/2504.01128).
- RipBench: 304 videos (303,491 frames; 51,011 annotated for six tasks), standardised video-level splits with video-level bootstrap intervals; frame classification F2 0.93, best detector 0.72, best instance segmenter 0.66 — **FT** (abstract and tables), [arXiv:2611.15574, NeurIPS 2026](https://arxiv.org/abs/2611.15574).
- Optical flow on wave-averaged movies at Duck: RMSE reduced to 0.1 m/s against surf-zone drifters after filtering — **ABS**, [Anderson et al. 2021, *Remote Sens.* 13:690](https://www.mdpi.com/2072-4292/13/4/690).
- Optical flow on low-pass-filtered frames at Anglet (H_s 0.8–3.3 m): r² 0.5–0.8, RMSE 0.12–0.24 m/s, bias −0.09 to −0.17 m/s against a current profiler — **ABS**, [Rodríguez-Padilla et al. 2021, *Remote Sens.* 13:1874](https://www.mdpi.com/2072-4292/13/10/1874).
- Wavelet-based optical flow ("Typhoon") on shore-based and UAV video tracks foam to give dense instantaneous currents; compared with an ADV, it captures wave-to-wave and low-frequency fluctuations; no error number in the abstract — **ABS**, [Dérian & Almar 2017, *IEEE TGRS* 55:5790](https://www.documentation.ird.fr/hor/fdi:010071324).
- No surf-zone or rip-current study using a learned optical-flow network (RAFT class) was found; the search returned only river and laboratory uses — **SNIP**.
- Nadir-view PIV, 10-min means, against in-situ current meters across the surf zone: alongshore r² 0.94, RMSD 0.07 m/s; cross-shore r² 0.51, RMSD 0.16 m/s (optical current meter: 0.88 / 0.08 and 0.44 / 0.22) — **ABS**, [Dooley et al. 2025, *JTECH* 42:33](https://doi.org/10.1175/JTECH-D-24-0015.1).

#### 1k. Sea-state / Beaufort classification from ship or platform imagery

| Work | Labels | Split | Result | Tag |
|---|---|---|---|---|
| Vorkapic et al. 2024, *Ocean Eng.* 309 (arXiv:2407.03755) | Beaufort 1–8 **estimated by human experts per video**; the paper states anemometer readings "are not completely consistent" with them | About 750 training and 300 test/validation **frames per class**, "not included in the training set"; no statement that test videos or voyages are separate | ResNet-101 accuracy 0.881, macro-F1 0.881; ViT-B/32 0.868 | **FT** — [arXiv](https://arxiv.org/abs/2407.03755) |
| Pobar et al. 2025, Int. Workshop on Pattern Recognition (SPIE) | Same group, **newly created test set** | New test set | Swin-Base F1 46.62 %; calibrated GPT-4o F1 27 %; MAE 0.738 (ResNet101d) vs 0.842 (GPT-4o) Beaufort | **ABS** — [DOI](https://doi.org/10.1117/12.3077196) |
| Pobar et al. 2026, *IEEE OJ-ITS* 7 | Real + synthetic training images, "substantially enlarged" test set, voting over frames | Enlarged test set | 96 % within one class, F1 62 %, maximum error 2 Beaufort | **ABS** — [DOI](https://doi.org/10.1109/OJITS.2025.3650057) |
| Ivašić-Kos et al. 2026, *IEEE J. Oceanic Eng.* | Images + **anemometer wind speed** as a second input | Not in abstract | Image-only ensemble F1 73 %; image + wind F1 97 %; six-class image-only 94 % | **ABS** — [DOI](https://doi.org/10.1109/JOE.2026.3674388) |
| Umair et al. 2022, *Symmetry* 14:1487 (MU-SSiD) | Four sea states; ground truth from a **half-hourly mean wind speed** (anemometer) plus Beaufort visual features; DSLR at one offshore (cruise ship) and four onshore sites | Random; the authors note that for sea state 4 (five videos) "similar image features are present both in the training and testing splits" | 100,800 images; best benchmark accuracy 81.8 % (NASNet-Mobile) | **FT** — [MDPI](https://www.mdpi.com/2073-8994/14/7/1487) |
| Zhang et al. 2026, *Ocean Eng.* 367:128212, "Beaufort-geometry ordinal learning for cross-domain sea-state image classification" | Unknown | Unknown ("cross-domain" in title) | Unknown | **META** — [DOI](https://doi.org/10.1016/j.oceaneng.2026.128212) |

#### 1l. Wind speed from sea-surface imagery

- Four differently worded searches returned no paper that regresses wind speed from optical images or video of the sea surface against an anemometer. What they returned instead: wind speed from video of a flag and a tree (0.75–11 m/s against a cup anemometer; arXiv:1905.13290), whitecap coverage from sea-surface photographs fitted against measured wind (Stramska & Petelski 2003, *JGR*), Sentinel-2 whitecap coverage against buoy wind, and SAR or buoy-spectrum wind retrievals — all **SNIP**.
- The closest camera work is MU-SSiD, whose class labels are derived from measured wind speed (so it is in effect a four-class wind-from-image task, 81.8 %) — **FT**, [Umair et al. 2022](https://www.mdpi.com/2073-8994/14/7/1487) — and Ivašić-Kos et al. 2026, where wind is an **input** — **ABS**, [DOI](https://doi.org/10.1109/JOE.2026.3674388).

### Inferences
- The published "accuracy bar" for camera H_s and period is set by protocols the project already treats as leaky: OWG's headline numbers are computed on a random 40 % of images from the same days (and that subset also drove early stopping); Xu, Kim and Choi are random-frame splits. No instrument-referenced camera H_s number on unseen sites was found other than the project's own.
- OWG's often-quoted 0.14 m / 0.41 s (IR) refers to **individual waves with lidar truth**, not H_s; the H_s/T_p numbers are the Santa Cruz RGB ones (0.08–0.11 m, 0.79–0.81 s) against an ADCP at the seaward edge of the view.
- OWG's own out-of-range test is the only published evidence on period extrapolation from stills, and it fails (0.79 s → 3.44 s), which is consistent with the project's finding that learned probes do not recover peak period.
- The Rijeka group's sequence — 88 % on a frame-level split (2024), F1 47 % on a new test set (2025), F1 62 % after synthetic data and voting (2026) — is a published within-group demonstration of how much a frame split overstates sea-state accuracy. The three papers do not state this comparison themselves.
- Where splits are honest (Collins 2021 held-out year; de Silva separate videos; RipVIS/RipBench video-level; Salatin 2026 unseen bathymetry), reported skill is moderate and errors are of the same order as the classical method (learned depth 0.37 m on a held-out year against cBathy V2.0's 0.38 m Kalman-filtered RMS at the same site; unseen bathymetry 0.66 m).
- For period, the non-learned single-camera methods (Ngene 2026, Yu 2024) report about 10 % of T_p against instruments, which for a 10 s swell is close to the best learned random-split figure; no learned method has been shown to beat a spectral or texture-motion estimate on the same video.
- The Egan course report reproduces the OWG pattern on a surf camera: RMS errors that look like OWG's (0.17 m, 0.4 s) with low explained variance (r² 0.51 and 0.21), a reminder that RMSE alone flatters results when the target range is narrow.

### Gaps
- Five 2026 *Ocean Engineering* papers that look like direct competitors (Liu Y. 127543; Ahammed 127753; Li 126780; Zhang G. 128212, all title-only; Liu H. 124561, number-free abstract) could not be read. Neither Semantic Scholar, Crossref nor OpenAlex holds their abstracts, and two exact-title searches returned nothing. Their datasets, instruments, splits and results are **unknown**; they must be read from the PDFs before any "we improve on" claim.
- Still not verified after the second pass: Almar et al. 2012 breaker-height errors (search summary only; the repository refused access); Benetazzo et al. 2012 and Bergamasco et al. 2017 stereo error budgets (hosts refused); per-site error tables in Buckley 2024; the segmentation score in Yang 2025; numeric agreement in Solodoch 2025 (a blog summary gives two conflicting figures, 0.15 m and 30 cm, so neither is recorded here); Venkateswarlu 2025 timestack errors (Springer redirected to a login); Zhang & Li 2024 and Zhou et al. 2025 (*Measurement*, "Wave parameters identification based on surveillance cameras") remain title-only.
- Wind speed from sea-surface imagery, other buoy-camera work, and a published validation of a commercial surf-camera height product were each searched specifically and nothing was found; a search of ScienceDirect, IEEE Xplore and CNKI by someone with access would be the remaining check.
- Wave setup: the only camera-derived setup results read are Buckley 2024 and Salatin 2024; a search for a dedicated video setup validation returned shoreline-elevation and toolbox papers (CoastCams, Didier et al. 2017) that were not opened — **SNIP**.
- Directional spread: the Hasaki surf-zone study (2024) could not be opened (404), so the only camera result on spread remains a search summary.
- Seen only in search summaries (**SNIP**, unverified): RipScout drone detector 93.1 % (*IEEE JSTARS* 2025); drone PIV for tidal currents RMSE 0.32 m/s; video–ADV vector-mean wave direction within 5° in a physical model; video PIV within 12–16 % of an ADV.

---

## Key question 2. Which 2024–2026 papers use foundation models or video transformers for these tasks, what did they report, and on what split?

### Takeaway
Only the two Kamagata preprints apply a self-supervised video encoder to wave height, period or direction, and neither has instrument-traceable labels; elsewhere, pretrained transformers appear in runup extraction, Beaufort classification and rip-current benchmarks, where DINOv2/DINOv3 backbones already lead and a general multimodal model (GPT-4o) was tested and lost to fine-tuned vision models.

### Cited Findings
- V-JEPA ViT-S + SlowFast transformer on shore video: held-out (5 videos at other sites) H_s RMSE 0.586 m, T_p 1.42 s, direction 37°; labels not stated — **PROJ-V1**, [arXiv:2607.11998](https://arxiv.org/abs/2607.11998). Semantic Scholar lists **0 citations** for it and for arXiv:2606.13302 — [citation record](https://api.semanticscholar.org/graph/v1/paper/ARXIV:2607.11998?fields=title,citationCount).
- Fine-tuned ResNet101d, XCiT-small and Swin-Base against OpenAI GPT-4o for Beaufort 1–8 on a new test set: Swin-Base F1 46.62 % vs calibrated GPT-4o 27 % — **ABS**, [Pobar et al. 2025](https://doi.org/10.1117/12.3077196).
- DeiT III, Swin, XCiT, CoAtNet and ResNet-101d with synthetic + real training and temporal voting: 96 % within one Beaufort class, F1 62 % — **ABS**, [Pobar et al. 2026](https://doi.org/10.1109/OJITS.2025.3650057).
- Ensemble of data-efficient ViT and EfficientNet members: image-only F1 73 %, with anemometer wind 97 % — **ABS**, [Ivašić-Kos et al. 2026](https://doi.org/10.1109/JOE.2026.3674388).
- Runup timestacks: a transformer segmentation model trained by transfer learning is compared with a convolutional model trained from scratch on 7,000+ timestacks from six sites — **ABS**, [Buckley et al. 2024](https://pubs.usgs.gov/publication/70257858).
- Swash: a video-object-segmentation network adapted from an urban-flood model, trained with **16** labelled images, RMSE 0.14–0.33 m against lidar — **FT**, [Salatin et al. 2024](https://par.nsf.gov/servlets/purl/10513109).
- Rip currents: NTIRE 2026 entrants used DINOv3 detectors and Mask2Former with a DINOv3 backbone; the report's abstract says general-purpose pretrained models help but task-specific structure still has room — **FT**, [NTIRE 2026 RipDetSeg report](https://arxiv.org/abs/2604.17070). RipBench reports a DINOv2-based RF-DETR leading detection — **FT**, [arXiv:2611.15574](https://arxiv.org/abs/2611.15574).
- Stereo: an attention-augmented pyramid network trained on WASS/SGBM disparities (Black Sea), temporal 10:1:1 split, test H_s 0.36–0.65 m: elevation R² 0.89, RMSE 0.03 m on one test condition, H_s errors within 10 % — **FT**, [Liu et al. 2025, arXiv:2512.06024](https://arxiv.org/abs/2512.06024). The abstract's "millimetre-level" claim is not the same as the 0.03 m RMSE in the text.
- "Polarization-aware multi-scale vision transformer" for H_s — **META**, [Ahammed & Dang 2026](https://doi.org/10.1016/j.oceaneng.2026.127753).
- Seven vision-language models were evaluated on more than 1,000 densely labelled coastal images (18 classes, Oahu) in text-to-mask, mask-to-mask and mask-to-text tests; coastal concepts were the hardest group, and the abstract names **no wave, sea-state, breaking or rip-current task** — **ABS**, [Knoop et al. 2026, arXiv:2609.10855 (IEEE OCEANS 2026)](https://arxiv.org/abs/2609.10855).
- Segment Anything combined with dynamic mode decomposition and DEM monoplotting has been used for water–land boundary and flooding monitoring (Wang et al. 2025, *Sci. Rep.*, doi:10.1038/s41598-025-94791-8), not for wave parameters — **SNIP**.
- A targeted search for DINOv2, Segment Anything or vision-language models applied to surf-zone wave parameters returned no paper — **SNIP** (negative result of one search).

Split used by each learned camera paper read here:

| Split type | Papers |
|---|---|
| Random frames or clips from the same site and days | Buscombe 2020 (FT); Xu 2024, Kim 2023, Choi 2020 (PROJ-V1); Egan course report (FT); Vorkapic 2024 (FT, frames per class); Stringari 2021 (FT, same records); Umair 2022 (FT) |
| Held-out time (days, periods, year) | Collins 2021 (year); Yin H. 2024 (time periods); Liu J. 2025 (time segments); Collins 2023 (separate series); Pobar 2025/2026 (new test set) |
| Held-out videos, scenes, sites or bathymetries | de Silva 2021; RipVIS 2025; RipBench 2026; Kamagata 2026 (labels untraceable); Salatin 2026 (one unseen bathymetry); Sáez 2026 (new sites, with fine-tuning) |
| Unknown | Liu Y. 2026; Liu H. 2026; Ahammed 2026; Li 2026; Zhang G. 2026; Hwang 2023; Buckley 2024 (train/validation lists released, proportions and site hold-out not stated); Lange 2023 (crest network) |
| Not learned (no split) | Ngene 2026; Yu 2024; Andriolo 2020/2025; cBathy family; Solodoch 2025; stereo methods |

### Inferences
- No published work applies DINOv2/v3, V-JEPA 2 or VideoMAE-class encoders to wave height, period, direction or spectra with instrument labels and a leakage-free split. The project's frozen/gently tuned DINOv3 and V-JEPA 2 experiments appear to be ahead of the published record on protocol, whatever the scores.
- The one published multimodal-model test (GPT-4o on Beaufort) suggests zero-shot general models are not yet a competitive baseline for sea state, but a fine-tuned vision model at F1 47 % on a fair test set shows the task itself is far from solved.

### Gaps
- Whether any of the META-only 2026 *Ocean Engineering* papers uses a foundation backbone is unknown.
- Buckley 2024's architecture is confirmed as SegFormer from the model release, but whether its evaluation is leave-site-out or pooled was not read.
- No follow-up to Buscombe's optical wave gauging was found among its 39–41 citing papers that re-tests it on a held-out period; the citing list contains mostly breaking, runup and bathymetry work — [citation list](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.coastaleng.2019.103593/citations?fields=title,year,venue&limit=1000).

---

## Key question 3. Stereo, multi-camera, polarimetric, UAV and lidar-supervised methods: what accuracy for elevation fields and spectra, and are there learned monocular methods trained on stereo/lidar truth?

### Takeaway
Stereo reaches about 0.10–0.14 m (roughly 10 %) on H_s against gauges and buoys and is the only optical route to elevation fields, directional spectra and crest statistics in the field; learned work so far replaces parts of the stereo pipeline. No field-validated monocular model trained on stereo or lidar truth was found apart from OWG's lidar-labelled IR set.

### Cited Findings
- WASS reprocessing of the Black Sea records in the Guimarães dataset against a wave gauge: H_s RMSE 0.10 m, mean absolute error 0.086 m, maximum deviation 0.165 m. The same paper's ship-borne WASS trial: H_s error "did not exceed 0.14 m" against a concurrent buoy; best baseline-to-height ratio 0.1–0.3; 10 Hz sampling preferred — **FT**, [Zhu et al. 2026, *Sensors* 26:993](https://www.mdpi.com/1424-8220/26/3/993).
- Expected stereo errors by installation, from quantisation analysis in the public stereo dataset paper: vertical RMS 1 cm (Black Sea platform and Acqua Alta tower), 3 cm (Yellow Sea), 9 cm (La Jument lighthouse); usable frequency limit 0.5–1.5 Hz. Co-located references are wire wave gauges (2 mm accuracy) at the Black Sea site and an AWAC at Acqua Alta; the gauge-versus-stereo spectrum statistics are in a figure, not in the text — **FT**, [Guimarães et al. 2020, *Sci. Data* 7:145](https://pmc.ncbi.nlm.nih.gov/articles/PMC7228941).
- Two unsynchronised smartphones (audio cross-correlation for sync): H_s RMSE about 11 %, directional spectrum recovered, against a pressure sensor — **ABS** (from PDF), [Vieira et al. 2020](https://www.mdpi.com/2077-1312/8/11/831). Follow-up "Nearshore space-time ocean wave observation using low-cost video cameras", [*Coastal Eng.* 197:104694](https://doi.org/10.1016/j.coastaleng.2024.104694) — **META**.
- CNN surface completion from sparse stereo points with the dispersion relation built in, **trained on purely synthetic data**, 5 frames/s — **ABS** (from PDF), [Pistellato et al. 2021, *Remote Sens.* 13:3780](https://www.mdpi.com/2072-4292/13/18/3780). Diffusion-model interpolation of stereo point clouds, where the spread of generated surfaces "strongly correlates" with the absolute error — **ABS**, [Pistellato et al. 2026, *IEEE TGRS*](https://doi.org/10.1109/TGRS.2026.3730661).
- Thermal (long-wave IR) stereo with a fine-tuned stereo network: mean bias under 2.1 % against wave probes, **wave flume** — **ABS**, [Li et al. 2025, *IEEE TIM* (arXiv:2408.10670)](https://arxiv.org/abs/2408.10670).
- Monocular: "WaveFormer" (teacher trained in simulation, self-training with pseudo-labels on real footage, zero-shot comparison with WASS) has a project page with **no quantitative results**, and paper, code and data marked "Coming Soon" — **FT** (project page), [WaveFormer](https://zhenhuang123987shu.github.io/WaveFormer/). "Measuring near-field wave elevation with monocular vision using self-supervised learning techniques" — **META**, [Wei et al. 2024, *MSSP* 208:111086](https://doi.org/10.1016/j.ymssp.2023.111086).
- Monocular on a moving vessel, **field data**: pixel depth is split into a mean-water-plane term from vessel motion and a wave-induced term from a learning model (keyworded self-supervised) trained on images taken in the South China Sea near a typhoon; the reconstructed 3-D field's spectrum, dominant frequency and H_s are compared with X-band radar, and the method is said to beat other vision methods on the same data. No numbers in the abstract, and the reference is radar, not stereo or lidar — **ABS**, [Wei et al. 2023, *IEEE Sensors J.*](https://doi.org/10.1109/JSEN.2023.3319028).
- Polarimetric slope sensing: the founding paper (Zappa et al. 2008, *Meas. Sci. Technol.*) recovers the 2-D slope field of short gravity waves from the polarisation of reflected skylight, tested in a wave tank and from a pier on the Hudson River; shipboard polarimetry later resolved wavelengths from 0.21 m down to 0.003 m (Laxague et al. 2015, *JGR Oceans*). No numeric accuracy was seen and both papers refused access — **SNIP**.
- Lidar-supervised camera learning: OWG's IR model (lidar-measured individual height and period) is the only example found — **FT**, [Buscombe 2020](https://par.nsf.gov/servlets/purl/10201658). Related, title only: lidar–IR fusion at breaking onset ([Carini et al. 2021](https://doi.org/10.1029/2020JC016934)) and a lidar–camera fusion system for the swash zone ([Matsuba et al. 2025](https://doi.org/10.1016/j.coastaleng.2025.104724)) — **META**.
- UAV: nadir-video spectral inversion ([Solodoch 2025](https://doi.org/10.1109/TGRS.2025.3536378), **ABS**); two-camera stereo from a single UAV ([Sawada et al. 2024, *J. Mar. Sci. Technol.*](https://doi.org/10.1007/s00773-024-01005-3), **META**); stereo-GoPro photogrammetry at the break point compared with multibeam lidar and pressure sensors ([Jaud et al. 2024, *ISPRS Open J.*](https://doi.org/10.1016/j.ophoto.2024.100077), **META**).
- Moving platforms: motion-compensated stereo ([Lowell & Irani 2026, *Measurement*](https://doi.org/10.1016/j.measurement.2026.121990)); stereo on a floating platform in a synthetic environment ([Jebari et al. 2025, *Ocean Eng.* 327:120958](https://doi.org/10.1016/j.oceaneng.2025.120958)); DSGP-Net for sparse stereo ([Wang et al. 2026, *Ocean Eng.* 356:125086](https://doi.org/10.1016/j.oceaneng.2026.125086)) — all **META**.
- A 2026 community review states that "detailed small-scale wave physics can now be explored with measurement techniques using light, including stereo-imaging and polarimetry" — **ABS**, [Collins et al. 2026, *Reviews of Geophysics*, "Measuring Ocean Surface Waves"](https://doi.org/10.1029/2025RG000888).

### Inferences
- A monocular model trained on stereo-derived elevation or spectra and tested on held-out bursts with a reported error would be new: the published monocular attempts are unevaluated (WaveFormer), radar-referenced with no numbers in the abstract (Wei 2023), or unread (Wei 2024, Li 2026). A dedicated search for monocular models supervised by stereo or lidar truth returned nothing else.
- Stereo's roughly 0.10 m H_s agreement, and the 1–9 cm expected vertical error of the public stereo records, are the accuracy ceiling to quote when stereo-derived quantities are used as training truth.
- Surface slope has an optical instrument of its own (polarimetry) at wavelengths below about 0.2 m; no learned estimate of buoy-band mean-square slope from ordinary imagery was found.

### Gaps
- Polarimetric slope imaging: known only from a search summary; no accuracy figure was read (IOP, Wiley and the mirror all refused access).
- Benetazzo et al. 2012 and Bergamasco et al. 2017 (the WASS accuracy papers) were refused on two attempts each.
- The *Reviews of Geophysics* full text and its preprint were not reachable (preprint host returned 403); it is the obvious next read for classical accuracy figures.
- Quantitative error for nadir-UAV spectra (Solodoch 2025) and for the UAV stereo and GoPro studies is unknown.

---

## Key question 4. Which measurements have a credible physical signal in monocular imagery but no strong learned baseline on real field data with a leakage-free split?

### Takeaway
Height (total and wind-sea), period, direction and spread, band-resolved spectra, surface slope, wind speed and individual-wave height all lack a published, instrument-referenced, leakage-free learned baseline; rip-current detection, runup extraction and video bathymetry are the areas where such baselines already exist.

### Cited Findings
- H_s and period: every instrument-referenced learned result read uses a random split at one site — **FT**, [Buscombe 2020](https://par.nsf.gov/servlets/purl/10201658); **PROJ-V1** for Xu 2024, Kim 2023, Choi 2020. The only cross-site test has untraceable labels — **PROJ-V1**, [Kamagata 2026](https://arxiv.org/abs/2607.11998).
- Direction: the only instrument-free comparisons are with a wave model — **ABS**, [Zikra et al. 2015](https://journals.utm.my/jurnalteknologi/article/view/4651) — or with lab motion — **FT**, [Wu et al. 2024](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2024.1508233/full).
- Spectra: field spectra from imagery come from stereo or nadir-UAV inversion, not from learned monocular models — **ABS**, [Solodoch 2025](https://doi.org/10.1109/TGRS.2025.3536378); [Vieira 2020](https://www.mdpi.com/2077-1312/8/11/831).
- Whitecap coverage: segmentation models exist (IoU 73 %) but each is trained and scored on one cruise — **ABS**, [Chen et al. 2025](https://www.mdpi.com/2079-9292/14/21/4286); [Yang et al. 2025](https://repository.library.noaa.gov/view/noaa/71822).
- Sea state from ships: fair-test F1 is 47–62 % with observer labels — **ABS**, [Pobar 2025](https://doi.org/10.1117/12.3077196), [Pobar 2026](https://doi.org/10.1109/OJITS.2025.3650057).
- Wind speed: four searches found no optical sea-surface wind regression against an anemometer (see 1l) — **SNIP** (negative search result).
- Physics-based single-camera retrieval is the stronger current baseline for swell period: T_p MAPE below 10 % against a co-located pressure sensor without any training — **ABS**, [Ngene et al. 2026](https://doi.org/10.1016/j.apor.2026.105161); T_p within 8–14 % of radar and buoy-motion references — **FT**, [Yu et al. 2024](https://castjournals.cast.org.cn/joweb/aos/EN/PDF/10.1007/s13131-024-2359-y).
- The one learned shore-video result with a reported skill score, on a random split, explains little variance: ConvLSTM r² 0.51 for height and 0.21 for period — **FT**, [Egan course report](https://galenegan.github.io/files/final_report.pdf).
- Already benchmarked with held-out videos or periods: rip currents — **FT**, [RipBench](https://arxiv.org/abs/2611.15574); bathymetry — **FT**, [Collins 2021](https://www.mdpi.com/2072-4292/13/23/4907); runup across six sites — **ABS**, [Buckley 2024](https://pubs.usgs.gov/publication/70257858).

### Inferences
Ranked by how well the project's data fit and how empty the published record is:
1. **Wind-sea height and total H_s at unseen buoys, same-hull spectra.** No published competitor uses instrument partitions or a station hold-out. The project's 0.215 m wind-wave RMSE against ERA5's 0.341 m has no published counterpart to be compared with.
2. **Peak and mean period from shore video on held-out days and sites, against a buoy.** The published learned numbers (0.79–0.81 s random split; 3.44 s out of range; 1.42 s with untraceable labels; r² 0.21 in a course report) leave a rigorous negative or positive result publishable either way. The comparison a reviewer will expect is against a non-learned spectral or texture-motion estimate (Ngene 2026, Yu 2024: roughly 10 % of T_p at their sites), which supports the project's plan to score a fixed-transect crest-timing method beside the video probe.
3. **Band-resolved spectrum / spectral shape from stills or video.** No learned monocular baseline was found at all.
4. **Direction and directional spread.** No instrument-validated learned result; spread has no camera validation of any kind in what was read.
5. **Wind speed (and Beaufort from the buoy's own anemometer) from buoy imagery.** Published sea-state work uses observer labels or wind-derived classes with random splits.
6. **Surface slope / whitecap fraction against same-hull spectra and wind.** Whitecap segmentation exists; a cross-platform, instrument-referenced benchmark does not.
7. **Individual-wave and maximum height from monocular video with lidar or stereo truth, held-out runs.** OWG is the single precedent, on a random split of 10.5 h.

### Gaps
- Items 3, 4 and 6 still rest on limited search coverage: no search was run specifically on learned spectra-from-imagery or on mean-square slope from ordinary cameras beyond the polarimetry query. Item 5 (wind) was searched four ways with no hit.
- The five unread 2026 *Ocean Engineering* papers could change items 1, 2 and 5.
- Chinese-language journals (for example *Chinese J. Ship Res.*, cited in the project's review) were not searched.

---

## Key question 5. Are there public benchmarks, challenges or leaderboards (with data) for any of these tasks?

### Takeaway
Rip-current detection is the only task with running public challenges and standardised splits; breaking has released datasets without a leaderboard; nothing comparable was found for wave height, period, direction or spectra from cameras.

### Cited Findings
- **RipVIS** (CVPR 2025): 184 videos, video-level train/val/test, benchmark site ripvis.ai — **FT**, [arXiv:2504.01128](https://arxiv.org/abs/2504.01128).
- **AIM 2025 RipSeg** challenge (ICCV workshops): 75 registered, 5 valid submissions; winner F1 0.72, F2 0.74, AP50 0.69, AP[50:95] 0.33, composite 0.68 — **FT**, [arXiv:2508.13401](https://arxiv.org/abs/2508.13401).
- **NTIRE 2026 RipDetSeg** challenge (CVPR workshops): 159 registered, 9 valid submissions; top composite score 55.79 — **FT**, [arXiv:2604.17070](https://arxiv.org/abs/2604.17070).
- **RipBench** (NeurIPS 2026): six tasks on the same videos and splits, project site ripbench.ai — **FT**, [arXiv:2611.15574](https://arxiv.org/abs/2611.15574).
- **Active breaking**: code, trained models and labelled sets released with the paper — **FT**, [Stringari et al. 2021](https://arxiv.org/abs/2101.11478). **Wave-by-wave breaking**: seven rectified Argus videos from three sites, CC BY 4.0 — **ABS**, [Zenodo 17956740](https://zenodo.org/records/17956740).
- **Surf-zone surface currents**: "PVLAB: Remotely sensed surface currents from field experiments (2013, 2018, 2021, 2022)", Zenodo 10.5281/zenodo.11581091 — **SEC** (reference-list entry in [Salatin et al. 2026](https://www2.whoi.edu/staff/elgar/wp-content/uploads/sites/153/2026/01/166.pdf); record not opened).
- **Sea state**: MU-SSiD, 100,800 images in four classes — **FT**, [Umair et al. 2022](https://www.mdpi.com/2073-8994/14/7/1487) (public availability not checked).
- **Rip currents, second public set**: RipAID, 2,815 images from eight SIRENA cameras at Cala Millor and Son Bou (Balearic Islands), oriented boxes with a separate "doubt" class, CC BY 4.0, 6.5 GB, **no split provided** — **ABS**, [Zenodo 15082427](https://zenodo.org/records/15082427).
- **Runup**: SegFormer weights and train/validation file lists for the Buckley 2024 models, CC BY 4.0 — **ABS**, [Zenodo 11167347](https://zenodo.org/records/11167347).
- A search for a public image or video set labelled with instrument wave height (Zenodo, Kaggle, benchmark) returned only the Guimarães stereo records and buoy-only collections such as the Free Ocean Wave Dataset — **SNIP** (negative result of one search).
- Already in the project's registry and not re-checked: Yin 2025 VWBT-9000 (breaker type), Guimarães 2020 stereo, OWG data on GitHub, Kwon 2023 radar — `docs/literature/landscape_review.md` §5.

### Inferences
- A multi-site, instrument-referenced camera wave-parameter benchmark with fixed station/site splits would be the first of its kind; the rip-current group's path (dataset paper → workshop challenge → unified benchmark with video-level confidence intervals) is a ready template, including its practice of reporting intervals over whole videos.

### Gaps
- Kaggle and Hugging Face were not browsed directly; one web search covered them and Zenodo.
- Whether Fusion-Wave (2025), MU-SSiD or the Rijeka UNIRI-SeaState sets are downloadable is unknown.
- Yin 2025's own baseline scores on VWBT-9000 were not re-read (the project holds the paper).

---

## Key question 6. How are the four target papers cited and extended; has anyone already claimed to beat them?

### Takeaway
No claim to beat Choi 2020, Kim 2023, Yang 2026 or Kamagata 2026 was found, but the check is incomplete: the papers most likely to contain one are five 2026 *Ocean Engineering* articles that cite them and whose text could not be read.

### Cited Findings
- **Choi 2020**: 72 citing papers on Semantic Scholar (73 on Crossref). Most are wave *forecasting* papers. Camera-relevant citers: Xu 2024; Zhang & Li 2023/2024 (three papers); Hwang et al. 2023 (*IJNAOE*, CNN wave detector + pixels-per-metre); Yang 2026; Liu H. 2026; Liu Y. 2026; Li 2026; Ahammed & Dang 2026; Zhang G. 2026; Yang R. 2025 (Fusion-Wave) — [citation list](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.oceaneng.2020.107129/citations?fields=title,year,venue&limit=1000); [Crossref count](https://api.crossref.org/works/10.1016/j.oceaneng.2020.107129).
- **Kim 2023**: Crossref counts 10 citing works; Semantic Scholar indexes the paper under its preprint record and lists 8, all camera papers: Xu 2024; Zhang & Li 2024 (three-stage extraction); Yang R. 2025 (Fusion-Wave); Liu H. 2026; Liu Y. 2026; Li 2026; Zhang G. 2026; Kamagata 2026 — [citation list](https://api.semanticscholar.org/graph/v1/paper/efd3b7b75748aa091b9f83937aff48b4f1dcc38d/citations?fields=title,year,venue&limit=100); [Crossref count](https://api.crossref.org/works/10.1016/j.oceaneng.2023.116002).
- **Yang 2026**: 9 citing papers on Semantic Scholar (10 on Crossref), including Li 2026 (multi-domain, reconstructed 3-D datasets), Ahammed & Dang 2026 (polarization-aware ViT), Lowell & Irani 2026 (motion-compensated stereo from a moving platform), and an X-band ship-motion framework — [citation list](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.oceaneng.2026.124517/citations?fields=title,year,venue&limit=1000).
- **Kamagata 2026** (both preprints): 0 citations — [record](https://api.semanticscholar.org/graph/v1/paper/ARXIV:2606.13302?fields=title,citationCount).
- **Xu 2024** (the project's fifth comparator) is cited by Liu Y. 2026, Liu H. 2026, Zhang G. 2026, Li 2026, both Kamagata preprints, Yang R. 2025 and Shi et al. 2025 — [citation list](https://api.semanticscholar.org/graph/v1/paper/DOI:10.3390/jmse12112003/citations?fields=title,year,venue&limit=1000).
- **Yin 2025** (VWBT) is cited by Kamagata 2026 and a flume breaker-type paper only — [citation list](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1038/s41597-025-06005-5/citations?fields=title,year,venue&limit=1000).

### Inferences
- The real competitors beyond the four targets, in order of how directly they overlap the project's shore-camera and buoy-camera tasks: **Liu Y., Xu et al. 2026** (coastal cameras, physics-informed spectral image representations — the spectral-input idea the project's video branch also uses; Guangjun Xu, Yuli Liu and Hongxia Chen also co-author the 2025 typhoon surf-detection paper, which suggests a continuing Chinese coastal-camera programme); **Liu H., Qin & Ti 2026** (3D-CNN with simulation pretraining, H_s and T_p); **Buscombe 2020** (still the lowest published learned errors); **Ngene et al. 2026** and **Yu et al. 2024** as the non-learned single-camera baselines for period (about 10 % of T_p against a pressure sensor, radar or buoy motion); **Xu 2024**; **Ahammed & Dang 2026** and **Li 2026** on the ship/3-D side; and the **Rijeka group** (Vorkapic, Pobar, Ivašić-Kos) for sea state.
- Kamagata 2026 has no uptake yet; beating it carries little weight on its own, whereas a protocol-matched comparison with Buscombe 2020 and with the 2026 *Ocean Engineering* papers would.
- The same four 2026 papers (Liu Y., Liu H., Li, Zhang G.) cite Choi 2020, Kim 2023 and Xu 2024 together, so they form one cluster that positions itself against exactly the project's targets; reading those four PDFs settles most of this question.

### Gaps
- Whether any of the unread 2026 papers reports a comparison against Choi, Kim, Yang or Xu on their own data is unknown; this is the single most important follow-up and needs the PDFs (ScienceDirect).
- Two of Kim 2023's ten Crossref-counted citing works are not in the Semantic Scholar list and were not identified.
- arXiv was listed by keyword on 2026-11-19 (wave height / sea state / wave period with camera or video): the only wave-parameter preprints returned were the two Kamagata papers and Vorkapic 2024, so no unpublished challenger was visible there, but the query set was small.
