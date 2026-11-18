# Ocean/EO foundation models and physics, flow and timestack front ends as feature extractors for camera wave estimation

Scope: whether ocean/SAR and general Earth-observation (EO) foundation models transfer to oblique sea-surface camera imagery, and which classical or physics-based front ends (optical flow, timestacks, spectral/cBathy, horizon tilt, whitecap fraction, physics losses) are worth fusing with a learned RGB backbone. The two datasets in question are (a) NDBC buoy-camera strips: six 480x270 RGB **snapshots** per hour, about 3 m above the sea, horizon visible, labelled with Hs, wind-wave and swell height from the same buoy, tested on unseen buoys; and (b) WebCOOS shore video: 2688x1520 at 20 fps in 4-10 min clips, labelled from CDIP buoys. The notes were compiled on 2026-11-18, and release dates are given where known.

Notation: "verified" means the number was read from the primary page or paper in this session. Numbers taken from a search-engine summary and not opened are marked "(search summary)".

---

## Q1. Which of these have evidence of improving camera-based wave estimates, with numbers against instruments? (This includes whether EO/ocean foundation models transfer to oblique camera imagery.)

### Takeaway
None of the ocean-SAR, general-EO or weather foundation models has been tested on optical camera imagery of the sea, oblique or otherwise. I found no study of such transfer. The only methods with instrument- or survey-validated numbers from cameras are domain-specific video and timestack methods: CNN "optical wave gauges", U-Net crest tracking on timestacks, and cBathy or celerity depth inversion. Almost all of these are surf-zone or nearshore results, and most are validated on depth, not on buoy Hs or Tp. The ocean-SAR foundation models still teach one transferable lesson: frozen or lightly probed self-supervised features regress Hs well, and a generic DINOv3 gets close to the ocean-specific model.

### Cited Findings

**Ocean / SAR foundation models (all Sentinel-1 Wave Mode, i.e. radar, nadir-ish, 20x20 km vignettes, labels from altimeters)**
- OceanSAR-2 (Tuel, Kerdreux, Febvre, Mouche, Grouazel, Miadana, Audras, Wang, Chapron) was submitted on 12 Jan 2026 and accepted at EUSAR 2026. It is described as the "second generation" of an SSL foundation model for Sentinel-1 WV, with improved SSL and dynamic data curation. Its tasks are geophysical pattern classification, wind vector, significant wave height and iceberg detection. The authors say they release "standardized benchmark datasets". — [arXiv 2601.07392](https://arxiv.org/abs/2601.07392)
- OceanSAR-2 architecture (verified): a ViT with 16x16 patches, 384-dim embedding and about 21M parameters (i.e. ViT-S/16 size). It takes calibrated sigma0 input and uses dynamic pruning that periodically selects the most diverse samples. Its pretraining effort is put at "roughly three orders of magnitude" less than DINOv3's. — [arXiv 2601.07392 HTML](https://arxiv.org/html/2601.07392v1)
- OceanSAR-2 evaluation protocol (verified): zero-shot kNN on **frozen** features, then fine-tuning with a 3-layer MLP head for 50 epochs (learning rate swept 5e-5 to 5e-3). — [arXiv 2601.07392 HTML](https://arxiv.org/html/2601.07392v1)
- OceanSAR-2 zero-shot kNN results (verified). SWH error is in m; the page does not say RMSE explicitly, but the reference is altimeter-based.

  | Model | TenGeoP acc. | SWH (m) | Wind speed (m/s) |
  |---|---|---|---|
  | OceanSAR-2 | 94.0% | 0.52 | 1.32 |
  | DINOv3 | 91.9% | 0.55 | 1.68 |
  | WV-Net | 91.5% | 0.64 | 1.71 |
  | TerraMind | 74.9% | 0.73 | 1.95 |

  — [arXiv 2601.07392 HTML](https://arxiv.org/html/2601.07392v1)
- OceanSAR-2 results with the MLP head (verified).

  | Model | SWH (m) | Wind speed (m/s) | Wind direction | Iceberg F1 |
  |---|---|---|---|---|
  | OceanSAR-2 | 0.40 | 1.01 | 16.9° | 0.865 |
  | DINOv3 | 0.39 | 1.12 | 17.9° | n/a |
  | WV-Net | 0.427 | 1.23 | 21.4° | 0.855 |
  | TerraMind | 0.67 | 1.82 | n/a | n/a |

  On SWH, the generic DINOv3 matches or slightly beats the ocean-specific model after an MLP head. — [arXiv 2601.07392 HTML](https://arxiv.org/html/2601.07392v1)
- OceanSAR-2 authors on generalist EO models: TerraMind, a generalist multimodal EO foundation model, performs worst even after fine-tuning. They attribute this to no ocean-SAR data in its pretraining and to its MAE objective, which gives "less linearly separable features". They also argue that multimodal alignment with optical data is a poor fit for SAR ocean signatures. — [arXiv 2601.07392 HTML](https://arxiv.org/html/2601.07392v1)
- OceanSAR-1 (Galeio with Ifremer; Hugging Face org last updated 28 Apr 2025) ships four checkpoints at 23.6M (probably parameters): OceanSAR-1 for features, plus -tengeop, -wind and -wave heads. With linear probing it reaches wind-speed RMSE 1.37 m/s and wave-height RMSE 0.63 m, best with ViT-B/8. TenGeoP accuracy is 75.5% (ResNet50) to 83.6% (ViT-B/8). The ViT-S/16, S/8 and B/8 variants are "on-demand", and no datasets are public. — [Hugging Face galeio-research](https://huggingface.co/galeio-research)
- OceanSAR-1 used the original DINO, and OceanSAR-2 moved to the DINOv2 formulation (iBOT patch loss plus KoLeo) (search summary). — [arXiv 2601.07392 PDF](https://arxiv.org/pdf/2601.07392)
- WV-Net (Glaser, Stopa, Wolniewicz, Foster, Vandemark, Mouche, Chapron, Sadowski; arXiv, 26 Jun 2024) is a ResNet50 trained with SimCLR on about 9.9M unlabelled S-1 WV images (about 165 months). Training took 200 epochs, batch 1024, on 8x V100-32GB for about 12 days. Input is 8-bit grayscale at 400-500 px a side, replicated to 3 channels. — [arXiv 2406.18765](https://arxiv.org/html/2406.18765v1)
- WV-Net results (verified). Wave-height labels come from S-1/altimeter colocation (200k images: 50k train, 50k val, 100k test).
  - Linear probe, Hs: RMSE 0.500 m versus 0.601 m for an ImageNet ResNet50 (MAE 0.370 vs 0.447).
  - End-to-end fine-tuning, Hs: RMSE 0.530 m, worse than the linear probe.
  - Air-temperature RMSE: 0.902 vs 0.974.
  - Multilabel AUROC: 0.958 vs 0.952.
  - kNN, MLP-probe and fine-tuning were also reported, and the model was "more data-efficient" when labels were scarce.

  — [arXiv 2406.18765](https://arxiv.org/html/2406.18765v1)
- The supervised predecessor, Quach et al. 2020 (IEEE TGRS), was trained on more than 750,000 SAR-altimeter colocations. It takes the real and imaginary parts of the SAR image spectra (plus CWAVE features) as input. Tested on 2018, it cut Hs RMSE from 0.6 m to 0.3 m against altimeters. This algorithm has supplied "Total Hs" (oswTotalHs) in the Sentinel-1 WV OCN products since 2022-06-07. — [Caltech resolver](https://resolver.caltech.edu/CaltechAUTHORS:20200724-100654154); [ESA Sentinel news](https://sentinel.esa.int/web/sentinel/-/total-significant-wave-height-is-now-provided-for-the-sentinel-1-wave-mode-acquisitions)

**General EO foundation models (Prithvi-EO-2.0, DOFA, TerraMind, DINOv3-SAT, etc.)**
- No source was found that evaluates Prithvi, DOFA, SatMAE, Scale-MAE, Clay, SkySense, CROMA or DINOv3-SAT on oblique, ground-level or camera images. The closest robustness benchmark, RSPDBench (2026), tests Prithvi-EO-2.0, DOFA and DINOv3 only on overhead imagery at about 1-30 m GSD (search summary). — [RSPDBench arXiv 2609.23427](https://arxiv.org/pdf/2609.23427)
- Pretraining in the remote-sensing domain beats ImageNet on remote-sensing scene classification by only "up to 2%" — [arXiv 2307.01645](https://arxiv.org/pdf/2307.01645). A 2022 study found that a self-supervised ImageNet encoder "still frequently wins" against RS-specific methods — [Remote Sensing 14(21):5500](https://www.mdpi.com/2072-4292/14/21/5500). Neither tests the reverse direction (RS weights applied to ground imagery), which is the direction that matters here (search summary).
- Prithvi-EO-2.0 is a multispectral, multi-temporal model (sizes: tiny-TL, 100M, 300M, 600M, with and without temporal/location embeddings), distributed through TerraTorch. — [Hugging Face Prithvi-EO-2.0-600M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M); [arXiv 2412.02732](https://arxiv.org/abs/2412.02732)
- DOFA is a hypernetwork that generates patch-embedding weights conditioned on each band's central wavelength (as described in the Prithvi-EO-2.0 paper). — [arXiv 2412.02732](https://arxiv.org/html/2412.02732v1)
- On Sen1Floods11 flood mapping, the reported IoUs are Prithvi-CAFE 83.41, Prithvi 82.50, TerraMind 82.90, DOFA 81.54 and SpectralGPT 81.02, so the generalist EO models are within about 2 IoU of each other on their home tasks (search summary). — [ResearchGate flood GFM benchmark](https://www.researchgate.net/publication/397280320_Assessing_the_value_of_Geo-Foundational_Models_for_Flood_Inundation_Mapping_Benchmarking_models_for_Sentinel-1_Sentinel-2_and_Planetscope_for_end-users)
- DINOv3 comes in two satellite variants pretrained on SAT-493M: a ViT-7B trained from scratch and a ViT-L distilled from it. A community copy of the model card gives GEO-Bench averages of 79.6 (ViT-L) and 81.1 (ViT-7B) (indicative, not from the official card). — [Hugging Face facebook/dinov3-vitl16-pretrain-sat493m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m); [community card copy](https://huggingface.co/demonwizard0/dinov3/blob/main/README.md)

**Camera, video and timestack methods with instrument or survey numbers**
- Buscombe, Carini, Harrison, Chickadel and Warrick, "Optical wave gauging using deep neural networks" (Coastal Engineering, 2019/2020), uses ImageNet CNNs on nearshore images. Inception-ResNetV2 was best and MobileNetV2 worst. The best gauge reached about 0.14 m RMS error for height and 0.41 s for period on the infrared dataset, "capturing up to 98% of the variance". Other findings:
  - Height extremes extrapolated reasonably well, but period extremes did not.
  - Inference takes under 100 ms per image on a CPU.
  - A separate model must be trained for each site.

  — [USGS publication page](https://www.usgs.gov/publications/optical-wave-gauging-using-deep-neural-networks)

  The NSF preprint gives different numbers: best model 0.08 m / 0.79 s, and 0.11 m / 0.81 s on oblique RGB (search summary) — [NSF PAR preprint](https://par.nsf.gov/servlets/purl/10201658). **These sources conflict** and were not reconciled. An ESIP follow-up reports preliminary values of 18 cm and 1.2 s (search summary) — [ESIP](https://www.esipfed.org/wp-content/uploads/2024/01/Buscombe.pdf).
- Lange, Fiedler, Merrifield and Guza (Coastal Engineering, 2023), "UAV video-based estimates of nearshore bathymetry", use a U-Net with a ResNet18 ImageNet encoder (about 17.8M parameters) to detect individual crests in timestacks, separating crests from the breaking toe. It was trained on only 2 hand-annotated timestacks (1 for validation) using 256x256 tiles, and overfits after about 17 epochs. Results, using linear phase speed with a heuristic nonlinear correction:
  - Surf-zone depth from crest celerity: RMSE about 0.17 m, bias about 0.06 m, over 112 transects of 17-minute hovers.
  - From the surf-zone edge to 10 m depth: RMSE about 0.87 m, bias about 0.58 m.
  - Crest tracking "improves cBathy estimates near the surf zone".

  — [GitHub AthinaLange/WaveCrestDetection](https://github.com/AthinaLange/WaveCrestDetection)
- Andriolo, Azevedo, Gonçalves and Taborda (Remote Sensing 17:2274, 2025) estimate wave celerity from wave trajectories on timestacks from an online streaming coastal webcam, and invert for depth (linear theory for shoaling waves, nonlinear for breaking). On a rocky bottom, with 2 days at Hs 1.7-3.5 m: mean bias 0.05 m, mean RMSE 0.43 m, depth-normalised RMSE under 20%. — [Remote Sensing 17(13):2274](https://doi.org/10.3390/rs17132274) ([PDF](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-17-02274/article_deploy/remotesensing-17-02274.pdf))
- cBathy accuracy, reported against depth surveys and not buoy wave parameters:
  - cBathy 2.0 (Holman and Bergsma): RMSE 0.56 m, or 0.38 m with Kalman filtering. Wave period is the most important input in deeper water. — cited in [Coastal Engineering 2025, satellite-video kinematics bathymetry](https://www.sciencedirect.com/science/article/pii/S0378383925000584) (search summary)
  - Duck, NC blending study: 1.06 m and 1.34 m on different days. — [arXiv 2311.01085](https://arxiv.org/html/2311.01085)
  - UAS: 0.21 m overall, 0.13-0.34 m per transect. — [NOAA repository](https://repository.library.noaa.gov/view/noaa/67534/noaa_67534_DS1.pdf)
  - Errors grow in storms (overestimation of up to 2 m with Hs above 3 m) and at spring tides, with errors at camera seams. — [FRF test bed](https://www.researchgate.net/publication/345714406_FRF_WAVE_TEST_BED_AND_BATHYMETRY_INVERSION)
  - Short-record variants have large RMSE below 2 m and above 7 m depth. — [ERDC/CHL MP-25-15](https://erdc-library.erdc.dren.mil/server/api/core/bitstreams/c1b1725c-13f7-404f-902d-2faaa83b9220/content)

  (all search summary)
- Ocean Dynamics 2025, doi:10.1007/s10236-025-01743-x ("Advancing nearshore wave monitoring: estimating parameters through video time stack imagery", Venkateswarlu et al.). Per the search summary, this is **not a deep-learning paper**. It filters a 20-minute cross-shore timestack, estimates height from intensity transitions, and gets period by zero-crossing. This corrects the brief, which grouped it with deep learning on timestacks. Its accuracy numbers could not be retrieved (Springer authentication redirect). — [Springer](https://link.springer.com/article/10.1007/s10236-025-01743-x)
- Kamagata, Jat, Gamundani, Srivastava and Saravanakumar (arXiv 2606.13302, 11 Jun 2026) estimate peak period directly from monocular shore video. Method:
  - Models: ViViT (LtViViT), attention-CNN, LSTM and ConvLSTM.
  - Data: Airy-theory synthetic pretraining (more than 1,000 sequences), then optical-flow pseudo-labels (Silver set, 10,655 windows), then expert plus timestack labels (Golden set, 6,926 windows).
  - Loss: Huber plus a physics penalty outside an Airy-valid period window.

  **No instrument ground truth was used.** LtViViT reaches RMSE 0.77 s and MAE 0.56 s against expert labels. The paper is internally inconsistent: the valid-period range is quoted as 2-20 s, 8-20 s and 0.05-0.5 Hz, and the ablation baseline RMSE of 2.50 s conflicts with other tables. The optical-flow labeller is never validated, and Hs is left to future work. — [arXiv 2606.13302](https://arxiv.org/html/2606.13302v1)

**Flow and breaking**
- WaveBreakingNet: a lightweight spatiotemporal model whose ConvNet on multi-frame dense optical flow "achieves very good performance" in breaking-type classification despite limited training data. The companion dataset has 9,000 labelled clips (spilling, plunging, surging) (Scientific Data, 2025). Breaking type is not checked against any instrument. — [ResearchGate WaveBreakingNet](https://www.researchgate.net/publication/396905496_WaveBreakingNet_A_Lightweight_Spatiotemporal_Deep_Learning_Model_for_Video-based_Nearshore_Wave_Breaking_Type_Classification); [Sci. Data](https://www.nature.com/articles/s41597-025-06005-5)
- Learned flow (FlowNet variants, RAFT) has been applied to coastal video mainly for rip currents. — [PMC rip-current UAV study](https://pmc.ncbi.nlm.nih.gov/articles/PMC13718950)
- In the surf zone, raw optical flow captures foam "moving at wave celerity". To get currents, one method low-pass filters the frames over a window tied to the dominant wave period before tracking. In other words, flow on unfiltered frames carries wave-phase motion, which is the signal wanted for celerity and period. — [Remote Sensing 13:1874 (2021)](https://doi.org/10.3390/rs13101874)

**Horizon/attitude and whitecaps**
- Horizon tracking for shipboard video (JTECH 32(1), 2015) uses a Hough transform horizon and pinhole geometry. It gives pitch/roll errors of about 0.3° and 0.6°, and "rectification errors are dominated by the uncertainty in camera height", which changes with platform heave. — [JTECH 2015](https://journals.ametsoc.org/view/journals/atot/32/1/jtech-d-14-00047_1.xml)
- An invariant EKF for autonomous surface vessels (2025) fuses horizon-derived roll/pitch with GPS heading. It discards near-vertical segments (clouds, wave crests) and corrects pitch for horizon depression. — [arXiv 2506.10850](https://arxiv.org/html/2506.10850v1)
- Wave period has been estimated from the pitch and roll of a floating lidar buoy (Sensors, 2021). — [PMC7918650](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7918650/)
- "Ship as a wave buoy" methods invert heave, pitch and roll for directional spectra using RAOs or deep learning. — [ResearchGate 343547921](https://www.researchgate.net/publication/343547921_Ship_As_A_Wave_Buoy_Estimating_Full_Directional_Wave_Spectra_From_In-Service_Ship_Motion_Measurements_Using_Deep_Learning)
- No paper was found that estimates wave parameters from **horizon-line motion in camera images** on a buoy (search summary).
- Whitecap extraction:
  - The standard is AWE (Automated Whitecap Extraction, NUI Galway), an adaptive brightness threshold. A buoy- or mast-camera study combined AWE with explainable deep learning and found coverage "modulated by various atmospheric and wave parameters", not wind alone (search summary). — [ResearchGate 303830085](https://www.researchgate.net/publication/303830085_Automated_measurements_of_whitecaps_on_the_ocean_surface_from_a_buoy-mounted_camera)
  - Deep-learning active-whitecap extraction finds "wind speed alone is insufficient". — [ResearchGate 393674679](https://www.researchgate.net/publication/393674679_Remote_Measurement_of_Active_Whitecaps_Using_Deep_Learning)
  - Published wind exponents for whitecap coverage range from 1 to 5. — [Kubryakov et al., RSE 2021](https://data-ww3.ifremer.fr/BIB/Kubryakov_etal_RSE2021.pdf)
  - Brooks et al. used a still camera on a free-floating spar buoy (search summary). — [NOAA repository 60518](https://repository.library.noaa.gov/view/noaa/60518/noaa_60518_DS1.pdf)

**Physics-informed learning and simulation-to-real training**
- A physics-driven CNN (Remote Sensing 13:3780, 2021) reconstructs the 3D sea surface from triangulated stereo points over 3 consecutive frames. It uses the linear dispersion relation to densify sampling and enforce temporal smoothness. — [doi:10.3390/rs13183780](https://doi.org/10.3390/rs13183780)
- PINNs for phase-resolved assimilation of nonlinear waves (Jan 2025) put free-surface boundary-condition residuals in the loss, using sparse point measurements and no imagery. — [arXiv 2501.08430](https://arxiv.org/html/2501.08430v1)
- NWnets encode wave energy balance and dispersion for nearshore processes. — [ICCE paper](https://icce-ojs-tamu.tdl.org/icce/article/download/12653/11926)
- A monocular pose estimator for shipborne UAVs, trained on Blender-rendered seas, became **biased in camera-height estimation when trained on images with an infinite sea horizon**. Adding horizon-free images helped. — [arXiv 2406.09260](https://arxiv.org/pdf/2406.09260)
- FFT ocean surfaces driven by a JONSWAP spectrum reproduce the input Hs to within about 7% (search summary). — [ResearchGate "On Modeling and Rendering Ocean Scenes"](https://www.researchgate.net/publication/233680624_On_Modeling_and_Rendering_Ocean_Scenes)
- No paper was found that trains a camera Hs regressor on JONSWAP/Airy renders and validates it against buoys. Simulated sea images do appear in X-band radar work, which has been tested on simulated and real radar (search summary). — [Academia figure](https://www.academia.edu/figures/33912797/figure-11-significant-wave-height-estimation-using-svr)

### Inferences
- **The SAR foundation models will not transfer as features to RGB oblique snapshots.** Their inputs are single-channel calibrated sigma0 at 5 m (downsampled) over 20 km near-nadir vignettes, and the SAR sea signal comes from radar imaging physics (tilt and hydrodynamic modulation, velocity bunching). An oblique RGB view at 3 m height sees specular sky reflection, glitter and foam, with strong perspective foreshortening. There is no shared low-level statistic to rely on. Their value is methodological:
  1. A frozen backbone with a kNN or linear probe gets within about 0.1 m of fine-tuning on Hs. In WV-Net, fine-tuning was even worse than the linear probe.
  2. A generic DINOv3 is within 0.03 m of the domain-specific OceanSAR-2 after an MLP head, and ahead of a generalist EO model (TerraMind).
  3. MAE-type objectives give poorly linearly-separable sea-state features.

  This supports DINO-family RGB backbones with frozen probes for the buoy strips, and argues against MAE-pretrained EO models.
- **General EO models are a poor bet here.** They have no evidence on oblique imagery, they expect nadir multispectral or GSD-tagged inputs (Prithvi-EO-2.0 is built around multispectral HLS bands, and DOFA conditions on wavelengths), and on their own home tasks they beat ImageNet by only about 2%. If any is tried, DINOv3-SAT ViT-L is the only cheap test: it shares the DINOv3 API, so it is a one-line swap for a frozen-probe comparison against DINOv3-web on the buoy strips. The expected result is that it is no better.
- **Motion front ends apply only to the WebCOOS video.** The NDBC strips are single snapshots, so optical flow, timestacks and frequency spectra are impossible there. For the buoy strips, the only physics-based front ends available are horizon tilt (an attitude, i.e. slope, sensor) and whitecap fraction. Both are per-image scalars and cheap to add.
- **Why snapshots miss swell, and why video can recover it.** Low-steepness swell barely modulates image texture, but it is periodic. Pixel-intensity time series (timestacks, cBathy-style cross-spectra) pick out the swell frequency band even at small amplitude, which is why cBathy works from swell-dominated video. A spectral or timestack front end on WebCOOS video is therefore the natural complement to a texture backbone that sees wind sea. This inference has not been tested against a buoy in the sources found.
- **Deep-water physics gives period from celerity without depth.** Linear dispersion gives c = gT/(2π), so T ≈ 0.64·c (s, with c in m/s). Celerity measured by flow or timestack in deep water converts directly to period. In shoaling water near piers, depth is needed, which is where cBathy and Lange-style errors grow (0.4-0.9 m depth RMSE beyond the surf zone). The CDIP buoys are offshore, so the pier video sees shoaled waves; any conversion from celerity to period should prefer frequency, which is conserved during shoaling, over phase speed.
- **The horizon tilt signal measured here (within-buoy Spearman 0.56 with spectral slope) is a slope-sensor channel.** Ship horizon trackers reach about 0.3-0.6° pitch/roll accuracy. On a buoy, the six views per hour give six samples of hull attitude: enough for a crude estimate of tilt variance, not enough to resolve period (each sample is a snapshot, and they are minutes apart). The camera-height error that dominates ship rectification affects geometric rectification but not the horizon angle itself.

### Gaps
- No study was found that tests any EO foundation model (Prithvi, SatMAE, Scale-MAE, Clay, DOFA, SkySense, CROMA, DINOv3-SAT) on oblique, ground or camera images of any kind, sea or otherwise.
- It is not stated which DINOv3 variant (web LVD-1689M or SAT-493M, and which size) OceanSAR-2 compared against. Its weight release and licence are unconfirmed.
- The Ocean Dynamics 2025 paper's accuracy numbers (behind Springer authentication) were not retrieved.
- The Buscombe OWG numbers conflict (0.14 m / 0.41 s versus 0.08 m / 0.79 s and 0.11 m / 0.81 s). Their label instrument and exact imagery type were not confirmed.
- No study was found that validates RAFT, SEA-RAFT, FlowFormer, UniMatch or Farneback-derived celerity or period **against a buoy**. Kamagata 2026 uses flow only as unvalidated pseudo-labels. Kamagata 2026 may be a different paper from the one whose Oak Island test cameras the project already pairs with CDIP 200; check the author list before citing.
- No direct cBathy-versus-buoy comparison of period or celerity was found. cBathy is validated on depth.
- No camera-based buoy-tilt-to-wave-parameter paper was found. The JTECH 2015 authors were not confirmed from the fetched text.
- Whitecap segmentation numbers from a shipborne EMA-SE-ResUNet (IoU 73.32%) appeared only in a search summary, without an identifiable paper, and are not cited above.
- No JONSWAP-rendered sim-to-real training for camera Hs validated on real buoys was found.

---

## Q2. Which can realistically be combined with a learned RGB backbone (late fusion of features), and at what cost?

### Takeaway
For the buoy strips, the realistic late-fusion add-ons are cheap scalar channels: horizon tilt or attitude per view, and a whitecap fraction (AWE-style threshold or a small segmenter). Physical-model priors can be added too: the project's existing ERA5 baseline, with AIFS or Aurora wave fields as possible extras for swell. For WebCOOS video, the worthwhile front end is spectral or timestack: pixel time-series spectra give peak frequency and swell-band energy, and crest-track celerity gives period. All of these are CPU-cheap. Dense learned flow (SEA-RAFT) is feasible at reduced resolution and frame rate, but it has no buoy-validated evidence. Ocean-SAR models are not useful as fused features. Weather foundation models are useful only as gridded scalar priors or teachers, not as image features.

### Cited Findings
- Aurora (Bodnar et al., Nature 2025, doi:10.1038/s41586-025-09005-y; arXiv 2405.13063):
  - It was fine-tuned on ECMWF HRES-WAM analysis at 0.25°, with variables Hs, mean wave period, mean direction, peak period and 10 m neutral wind.
  - It "exceeds numerical models on 86% of targets" for 10-day global wave forecasts.
  - It includes a Typhoon Nanmadol case.

  — [arXiv 2405.13063](https://arxiv.org/pdf/2405.13063); [Aurora wave example docs](https://microsoft.github.io/aurora/example_wave.html); [ScienceDaily](https://sciencedaily.com/releases/2025/05/250522124851.htm)
- A 2026 hybrid-AI Hs paper cites Aurora as state-of-the-art for 10-day global wave forecasts (search summary). — [Ocean Engineering 2026](https://www.sciencedirect.com/science/article/abs/pii/S0029801826011054)
- ECMWF's AIFS Marine (blog, 12 Oct 2026):
  - It is a **prototype**, not operational, that adds Hs, mean period and direction plus ocean and sea-ice variables.
  - It is trained on ERA5, a dedicated wave hindcast and the ORAS6 ocean reanalysis.
  - Its Hs errors are about 10% lower than ecWAM's against satellite altimeters (May-Aug 2024).
  - The blog states that waves are already in operational AIFS v2.

  — [ECMWF AIFS Marine blog](https://www.ecmwf.int/en/about/media-centre/aifs-blog/2026/aifs-marine); [arXiv 2604.25559](https://arxiv.org/html/2604.25559v1)
- ECMWF open data lists Hs for AIFS Single and AIFS ENS, including period-band breakdowns such as 10-12 s and 21-25 s (search summary). — [ECMWF open data](https://www.ecmwf.int/en/forecasts/datasets/open-data)
- SEA-RAFT (ECCV 2024 Oral, best-paper candidate; arXiv 2405.14793) claims to be "at least 2.3x faster than existing methods", with a Spring EPE of 3.69 and best cross-dataset generalisation (KITTI, Spring). The checkpoint name (spring540x960-M) suggests training at 540x960. — [GitHub princeton-vl/SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT)
- RAFT builds all-pairs correlation volumes and updates flow recurrently. — [arXiv 2003.12039](https://arxiv.org/pdf/2003.12039)
- MATLAB's RAFT documentation says RAFT is more accurate than Farneback, especially in low texture and motion blur, but needs more time and memory. — [MathWorks opticalFlowRAFT](https://www.mathworks.com/help/vision/ref/opticalflowraft.html)
- Lange's crest U-Net (about 17.8M parameters, 256x256 tiles) was trained from only 2 annotated timestacks, showing that crest detection on timestacks needs very little labelled data. — [GitHub WaveCrestDetection](https://github.com/AthinaLange/WaveCrestDetection)
- Buscombe's CNN wave gauge runs in under 100 ms per image on a CPU but must be trained per site. — [USGS](https://www.usgs.gov/publications/optical-wave-gauging-using-deep-neural-networks)
- CoastCams is an open-source MATLAB toolkit that derives wave height, wave period, water level and morphology from camera timestacks (search summary). — [IRD PDF](https://horizon.documentation.ird.fr/exl-doc/pleins_textes/2023-10/010090198.pdf)
- Horizon trackers that predict the horizon location and search a narrow band run in real time and reduce false detections. — [ScienceDirect, Optik 2015](https://www.sciencedirect.com/science/article/abs/pii/S0030402615009171)
- A ship horizon tracker based on the Hough transform is "robust and computationally efficient". — [JTECH 2015](https://journals.ametsoc.org/view/journals/atot/32/1/jtech-d-14-00047_1.xml)
- In the OceanSAR-2 MLP-head setting, the head is a 3-layer MLP trained for 50 epochs on frozen or fine-tuned features, the same shape as a late-fusion head. — [arXiv 2601.07392 HTML](https://arxiv.org/html/2601.07392v1)

### Inferences
- **Buoy strips (six 480x270 snapshots per hour).** The late-fusion design is: per-view backbone embedding (frozen DINO-family), plus per-view horizon features (tilt angle, horizon height in px), plus per-view whitecap fraction (AWE threshold below the horizon). Aggregate across the six views with mean, variance and range, which turns tilt into a slope-variance estimate. Then concatenate the model-prior scalars (ERA5 now; AIFS or Aurora swell bands if obtainable) and feed an MLP or GBM head.
  - Cost: horizon detection and AWE are CPU-only and run in milliseconds per image.
  - Risk: horizon detection fails in fog, glare, rain and at night; a per-view validity flag is needed.
  - Swell, which the camera cannot see, should come from the model prior. The camera adds the wind-sea correction, matching the measured +0.04-0.06 macro-F1 over hindcasts.
- **Model priors and leakage.** Aurora and AIFS fields are trained on analyses (HRES-WAM, ERA5 and a wave hindcast) that assimilate altimeter data. Whether NDBC buoy Hs enters those analyses was not verified. If it does, using them as features against same-buoy labels is mild leakage, and the evaluation should note it.
- **Model priors are archive-limited.** AIFS open data is operational forecast output, so for 2025 training labels, ERA5 (already in the package) remains the consistent reanalysis prior. Aurora's wave checkpoint would have to be run on HRES-WAM or IFS initial conditions to reproduce past hours, which costs GPU time and needs data access. That is not worth it unless the ERA5 prior is shown to be the bottleneck.
- **WebCOOS video front end, in order of value per cost:**
  1. *Pixel time-series spectra* (cBathy-style or simple FFT over sampled pixels or timestack lines), giving peak frequency, swell-band and wind-band energy ratios, and dominant direction from cross-spectral phase. This needs no training and minimal CPU: a few hundred pixel series x 12,000 frames. It directly supplies the swell information that snapshots lack, and fuses as a handful of scalars per clip.
  2. *Timestack crest tracking* (Lange U-Net or a Radon or cross-correlation celerity), giving wave-by-wave period and celerity. The U-Net needs little labelling (2 stacks in Lange) and is small.
  3. *Dense learned flow* (SEA-RAFT at about 540x960 on frame pairs subsampled to a few Hz), giving flow-magnitude and orientation statistics, and flow-derived crest speed. GPU cost is non-trivial at 2688x1520 x 20 fps x 4-10 min (4,800-12,000 frames per clip), so downscaling and temporal subsampling are mandatory. There is no buoy-validated evidence that it beats intensity spectra for period, and foam advection contaminates the signal in the surf zone.
- **Physics losses are cheap to add to any head.**
  - Bound Tp to a plausible range (Kamagata-style penalty).
  - Enforce consistency between predicted Tp and measured celerity via dispersion (deep water: c = gT/2π).
  - Enforce a steepness bound, Hs/Lp below breaking-limit values, linking Hs and Tp predictions.

  These are regularisers, not features. Evidence for their benefit against instruments is absent.
- **Sim-to-real with JONSWAP/Airy renders** is a larger project (renderer, sky and lighting, a camera model at 3 m height with the horizon in view). The UAV-pose result warns that a visible horizon creates shortcut biases tied to camera height. That matters here because horizon tilt is itself a genuine signal. Low priority.

### Gaps
- There are no published wall-clock numbers for SEA-RAFT at 2688x1520 or 1080p on a specific GPU; only the relative "2.3x faster" claim was found.
- Whether Aurora's wave checkpoint and AIFS wave outputs can be cheaply re-run or downloaded for the 2025 NDBC/CDIP hours (archive extent of AIFS open data) was not verified.
- No study was found that late-fuses timestack or spectral features with a CNN/ViT image embedding for buoy-validated Hs or Tp.
- UniMatch and FlowFormer runtime and accuracy for sea-surface video were not researched (no time).

---

## Q3. Licences, weight availability and compute

### Takeaway
Weights that are both freely usable and easy to get exist for SEA-RAFT (BSD-3), Prithvi-EO-2.0 (Apache-2.0) and OceanSAR-1 (public on Hugging Face, licence not shown on the org page). DINOv3 satellite weights are gated under Meta's custom DINOv3 licence. OceanSAR-2 and WV-Net weights were not confirmed as released. Aurora and AIFS licences were not verified in this session. Every model in scope is small enough for frozen-feature extraction on one GPU. The SSL pretraining runs themselves (12 days on 8 V100s for WV-Net) are not something to repeat.

### Cited Findings
- **OceanSAR-1**: 4 checkpoints at 23.6M (feature extractor, plus TenGeoP, wind and wave heads) are public on Hugging Face under galeio-research (updated 28 Apr 2025). The licence is not stated on the org page, larger ViT variants are "on-demand", and no datasets are public. — [Hugging Face galeio-research](https://huggingface.co/galeio-research)
- **OceanSAR-2**: the arXiv abstract mentions releasing benchmark datasets but not weights. The paper is CC BY-SA 4.0, which covers the article, not the weights. — [arXiv 2601.07392](https://arxiv.org/abs/2601.07392)
- **WV-Net**: weights and code "will be made available" at github.com/hawaii-ai/WVNet. No weight licence is stated, and the paper is CC BY-NC-SA 4.0. The pretraining data (GOALI) is unpublished. Training took about 12 days on 8x V100-32GB. — [arXiv 2406.18765](https://arxiv.org/html/2406.18765v1)
- **Prithvi-EO-2.0**: the official Hugging Face card is tagged apache-2.0 (ibm-nasa-geospatial), with tiny-TL, 100M, 300M and 600M variants, distributed through TerraTorch. Third-party re-implementations have incompatible parameter names. — [Hugging Face Prithvi-EO-2.0-600M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M); [OneScience reproduction note](https://huggingface.co/OneScience-Group/PrithviEO)
- **DINOv3 SAT-493M** (ViT-L/16 distilled and ViT-7B/16): gated on Hugging Face (contact-sharing agreement) under the "DINOv3 License" (custom; terms not retrieved). — [Hugging Face dinov3-vitl16-pretrain-sat493m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m); [dinov3-vit7b16-pretrain-sat493m](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-sat493m)
- **SEA-RAFT**: BSD-3-Clause, with weights on Google Drive and Hugging Face (e.g. `MemorySlices/Tartan-C-T-TSKH-spring540x960-M`). — [GitHub princeton-vl/SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT)
- **RAFT** in MATLAB (opticalFlowRAFT) is available in the Computer Vision Toolbox. — [MathWorks](https://www.mathworks.com/help/vision/ref/opticalflowraft.html)
- **Lange WaveCrestDetection**: PyTorch, with training data and the trained model on Google Drive (tinyurl.com/timestack) and a checkpoints/ folder. The repo has a Licence.md, but its terms were not shown, and checkpoint names in the documentation are inconsistent (epoch50 vs epoch17). — [GitHub AthinaLange/WaveCrestDetection](https://github.com/AthinaLange/WaveCrestDetection)
- **Kamagata 2026 (period from video)**: the paper is an arXiv preprint; code and weight release were not confirmed. — [arXiv 2606.13302](https://arxiv.org/html/2606.13302v1)
- **AIFS Marine**: a prototype; the blog names the open-source Anemoi framework but does not state weight availability or licence. — [ECMWF AIFS Marine blog](https://www.ecmwf.int/en/about/media-centre/aifs-blog/2026/aifs-marine)
- **Aurora**: the public documentation includes a wave-prediction example, implying a wave checkpoint is usable. The licence was not verified. — [Aurora docs: Predictions for Ocean Waves](https://microsoft.github.io/aurora/example_wave.html)
- **Compute reference points**:
  - OceanSAR-2 claims its pretraining cost about 1/1000 of DINOv3's. — [arXiv 2601.07392 HTML](https://arxiv.org/html/2601.07392v1)
  - Buscombe's gauge runs in under 100 ms per image on a CPU. — [USGS](https://www.usgs.gov/publications/optical-wave-gauging-using-deep-neural-networks)
  - The Lange U-Net trains at batch 8 on 256x256 tiles. — [GitHub](https://github.com/AthinaLange/WaveCrestDetection)

### Inferences
- **Practical shortlist by licence and effort:**
  - DINO-family RGB backbone: already the project's direction, and supported by the OceanSAR-2 result for DINOv3.
  - Classical front ends: horizon (Hough), AWE whitecaps and FFT/timestack spectra. These are licence-free, written in-house or from published algorithms.
  - SEA-RAFT (BSD-3), only if flow is tried on WebCOOS.
  - Lange U-Net, if its licence permits, for crest tracking.
  - ERA5, already in the package, as the physical prior.

  Not worth the licence or effort: Prithvi, DOFA or other EO models (no oblique evidence, multispectral inputs), and OceanSAR or WV-Net (SAR-only features).
- **Compute is not the constraint for the buoy strips.** A frozen ViT on six 480x270 images per hour over multi-year archives is a modest single-GPU job. Given the desktop GPU overheating, it can run on the lab GPU server or in CPU-batched form for small backbones.
- **WebCOOS flow is the only compute-heavy item.** Spectral and timestack features avoid the GPU entirely, which is another reason to prefer them first.

### Gaps
- Licence terms were not verified for: OceanSAR-1 weights, the DINOv3 License (commercial and redistribution terms), Aurora weights and code, AIFS (Single, ENS, Marine) weights, the Lange repo Licence.md, DOFA, Clay, SatMAE, Scale-MAE, SkySense and CROMA (not researched within the tool budget).
- The release status of OceanSAR-2 weights and of WV-Net at github.com/hawaii-ai/WVNet was not checked directly.
- There are no absolute GPU timings for any flow model at WebCOOS resolution.
