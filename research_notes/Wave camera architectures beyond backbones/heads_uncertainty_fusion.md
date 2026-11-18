# Output heads, uncertainty heads and fusion for camera + met-ocean wave estimation

Scope: spectral output heads, distributional/uncertainty heads and conformal prediction, ordinal heads, image-plus-tabular fusion (including tabular foundation models), multi-task and physical consistency, and remote-sensing wave literature that used these heads. Current as of 2026-11-18.

How these notes were built: sources were read through abstracts, search snippets, and three full texts extracted from PDF (Tripathi et al. ICASSP 2024; Wang et al. Remote Sensing 2026; Afham & Chandrasena engrXiv). Items marked "(canonical; not re-fetched)" are well-known primary papers cited from their standard arXiv/DOI pages. I did not re-open them this session, so their headline claims are stated conservatively. Items released June–November 2026 are marked **[2026-H2]**. Most of those I verified only at the abstract or listing level.

---

## Q1. Has anyone predicted a full wave spectrum from optical cameras or video, or from SAR, radar or ship motion? With what head, and how accurate?

### Takeaway
I found no paper that trains a learned head to output E(f) or E(f,θ) from monocular optical imagery or video and validates it against buoy spectra. The optical work regresses bulk parameters directly (Hs, Tp, Tm, direction), sometimes with physics constraints, or reconstructs the surface from stereo and then computes the spectrum. Full-spectrum heads are established for SAR: U-Nets mapping SAR cross-spectra to a frequency–direction grid, trained on WW3 or ERA5 spectra. They also exist for X-band radar (CNN from image spectrum to directional spectrum) and for ship motion (encoder–decoder, adversarial and hybrid Bayesian methods). Nearly all of these are trained and validated against model spectra (WW3/ERA5), not buoys. The strongest recent design is a partitioned one: a learned wind-sea scalar fills a JONSWAP prior, and a U-Net then fuses it with the observed swell spectrum. That maps closely onto "camera sees wind sea, ERA5 sees swell".

### Cited Findings

**Optical camera / video (no learned spectral head found)**
- **[2026-H2]** Kamagata et al. (arXiv 2607.11998, July 2026) estimate five coastal wave parameters from one shore camera. The pipeline is a self-supervised V-JEPA backbone, a SlowFast temporal encoder, Farneback optical flow, and a *regression head with Airy dispersion constraints*. Reported Pearson r: 0.832 for wave direction, 0.680 for zero-upcrossing period, and 0.643 for peak period. The authors name regression attenuation and directional accuracy as limits and propose larger datasets and stereo or multi-camera setups — [arXiv 2607.11998](https://arxiv.org/abs/2607.11998)
- **[2026-H2]** "Physics-Guided Spatiotemporal Learning for Coastal Wave Peak Period Estimation from Video" (arXiv 2606.13302, June 2026) regresses nearshore Tp directly from monocular surf-zone video. It explicitly skips timestacks, celerity inversion and *spectral analysis*. The authors also note that benchmark and annotated coastal-video datasets are scarce — [arXiv 2606.13302](https://arxiv.org/abs/2606.13302)
- Wave height *classification* from monoscopic ocean video uses a CNN+LSTM over frame sequences, with ADCP ground truth. Single-snapshot models did poorly in sea states 4–5 — [Ocean Engineering (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0029801823023867)
- Stereo route: a neural 3-D wave reconstruction from a stereo camera (arXiv 2512.06024) predicts disparity and point clouds. Its spectra closely follow the stereo-vision reference across a wide frequency band, including the power-law high-frequency tail. A related version (PRIME-Net) reports dominant-frequency error below 0.01 Hz, with the camera 11 m above the sea surface. Here the spectrum is *computed* from the reconstructed surface, not predicted by a head — [arXiv 2512.06024](https://arxiv.org/pdf/2512.06024)

**SAR (full-spectrum heads are established)**
- Quach, Glaser, Stopa, Mouche & Sadowski (IEEE TGRS, vol. 59(3), accepted 2020, often cited as 2021). A deep network takes Sentinel-1 wave-mode image cross-spectra plus hand-crafted features and predicts *scalar* Hs. Training used 2015–2017 SAR–altimeter collocations; on 2018 test data RMSE fell from 0.6 m to 0.3 m against altimeters (−50%). The authors trained on altimeters rather than WW3 because WW3 is unreliable in high seas — [Caltech record](https://resolver.caltech.edu/CaltechAUTHORS:20200724-100654154); [NSF PAR copy](https://par.nsf.gov//servlets/purl/10273998)
- Operational: since 2022-06-07, Sentinel-1 Wave Mode OCN products include a "total" Hs from this NN algorithm, trained on altimeter data — [ESA Sentinel news](https://sentinels.copernicus.eu/-/total-significant-wave-height-is-now-provided-for-the-sentinel-1-wave-mode-acquisitions)
- Tripathi, Chapron, Collard, Guitton, Lopez-Radcenco, Mouche & Fablet, "Deep Learning Inversion of Ocean Wave Spectrum from SAR Satellite Observations" (ICASSP 2024, pp. 8711–8715). From the extracted full text:
  - Output: a *normalised* WW3 wave spectrum on a polar grid of 24 wavenumbers (0.005–2.063 rad m⁻¹) × 32 directions (0–345°), shape (1, 24, 32).
  - Data: 142,000 collocations (100,000 train, 10,000 validation, rest test).
  - Heads: a direct U-Net, and a two-stage pair of U-Nets (one denoises the SAR cross-spectrum towards a closed-form simulated cross-spectrum, the second inverts it), each trained with MSE.
  - Results: more than 65% MSE reduction and more than 50% correlation improvement over the ESA baseline. The 4-scale U-Net beat a single-scale one by 30% in MSE. A plain ConvNet looked less suitable in preliminary runs.
  - [Archimer record](https://archimer.ifremer.fr/doc/00885/99670/); [PDF](https://archimer.ifremer.fr/doc/00885/99670/118241.pdf)
- Wang, Chen, Zhu, Chang, Fang, Huang, Yang & Chapron, *Remote Sensing* 18, 1703 (published 25 May 2026). From the extracted full text:
  - Wind-sea step: XGBoost estimates wind-sea height, which parameterises a **JONSWAP prior** spectrum (cosine-2s spreading).
  - Fusion step: a U-Net with skip connections fuses that prior with the official OSW swell spectrum, trained with an **L1 loss** against ERA5 2-D spectra.
  - Bulk results against ERA5: Hs RMSE 0.4026 m, mean period 0.4342 s, mean direction 20.42°, under strong azimuth cut-off.
  - The XGBoost wind-sea-height step has RMSE 0.48 m against ERA5.
  - Against the DL method SAR2WV (Cao et al., TGRS 2024), mean dimensionless spectral error was 0.29 versus 0.33. Hs RMSE was slightly worse (0.403 m versus 0.386 m), which the authors attribute to their harder cut-off subset.
  - The authors stress that validation is only against ERA5 and that buoy validation is still needed.
  - [MDPI PDF](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-18-01703/article_deploy/remotesensing-18-01703.pdf); [DOI](https://doi.org/10.3390/rs18111703)
- Wu & Li (Remote Sensing of Environment, 2024), "Wave-Spec-CNN", retrieve omnidirectional spectra from SAR with a calibration branch and physics constraints. Cao et al. (TGRS 2024), "SAR2WV", is a Pix2pix-style cross-spectrum → wave-spectrum model. Both known from search summaries only, with no numbers verified here — [x-mol listing](https://x-mol.com/paper/1830113904990691328); [summary listing](https://bohrium.dp.tech/paper/arxiv/53e99efbb7602d97027c7d0e)
- Partition heads in SAR: operational OCN products split the spectrum into up to five partitions with a watershed algorithm. Benchaabane et al. (Preprints.org 2025, not peer reviewed) learn a per-partition quality flag from errors against WW3 partitions — [Preprints.org](https://www.preprints.org/manuscript/202507.1929)

**X-band radar**
- A 3-D CNN (VGG-style) on 64-frame radar sequences reached Hs RMSE 0.3576 m against buoys; the training set was skewed to low waves, so the authors used stratified sampling — [PLOS ONE 2023 (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10615293/)
- A CNN takes the radar *image spectrum* and outputs the *directional wave spectrum*, a fully data-driven spectral head. Numbers were not retrieved because MDPI returned 403 — [JMSE 12(9):1609, 2024](https://doi.org/10.3390/jmse12091609)
- A shore-based CNN with self-attention for Hs and period cut Hs RMSD by 0.24 m versus SNR-based methods and by 0.11 m versus an earlier CNN — [JMSE Feb 2024 (DOAJ)](https://doaj.org/article/3f96efd538094f1d98b9d88d084d76ea)
- A 2026 TU Delft thesis (ViT on shipborne X-band) reports that cross-vessel transfer is unresolved and that low wind is the most consistent failure mode — [TU Delft repository](https://repository.tudelft.nl/record/uuid:29d1170b-30b0-4e46-b86e-0ac5c259f891)

**Ship motion ("ship as a wave buoy")**
- Scholcz & Mak (OMAE 2020) use a deep convolutional encoder–decoder on 2 years of in-service frigate motion data, with WAMOS II radar spectra as labels, and predict full directional spectra in real time. No accuracy numbers are given in the abstract; the authors say more data is needed — [MARIN publication page](https://www.marin.nl/en/publications/ship-as-a-wave-buoy-estimating-full-direction-wave-spectra-from-in-service-ship-motion-measurements-using-deep-learning)
- Han et al. (Marine Structures 2022) pair a 2-D-spectrum estimator with a discriminator network that detects high-order inconsistencies, an adversarial "physical plausibility" loss. Evidence is from simulation only — [SINTEF Brage](https://sintef.brage.unit.no/sintef-xmlui/handle/11250/2999929)
- Nielsen et al. (Marine Structures 2023) use a hybrid: a CNN predicts integral parameters, then a Bayesian, transfer-function-based step estimates the directional spectrum *conditioned on* them. It was tested on about 2 years of container-ship data — [DTU Orbit](https://orbit.dtu.dk/en/publications/wave-spectrum-estimation-conditioned-on-machine-learning-based-ou/); [2024 slides](https://www.g2.kaiyodai.ac.jp/marine-ai/activity/img/ML%20estimation%20of%20direc.%20wave%20spectra%20%28Nielsen%20et%20al.%202024%29.pdf)
- A "spectrum-regularized deep learning with joint wave parameter prediction" model (ship motions) uses a shared dual-domain encoder and three cross-attention branches for Hs, direction and Tp, coupled through spectral relations. Ablation numbers were not retrieved (TRID refused the connection) — [TRID record](https://trid.trb.org/View/2782381)

**Spectral losses and moment consistency**
- Afham & Chandrasena (engrXiv preprint, "Physics-Informed Residual Learning for Directional Wave Spectrum Reconstruction"). From the extracted full text:
  - Method: an analytical Fourier-expansion baseline built from S(f) and (a1, b1, a2, b2), with a network trained on the *residual*.
  - Losses: spectral MSE, an energy-conservation loss (integrated energy must match the input S(f)), a directional-moment loss, a shape loss, and clamping for non-negativity.
  - Results on ERA5: physics + residual beat pure ML and residual-without-physics.
  - Limits: analytical MEM and NDBC methods *remained the most accurate overall*. On unseen North Sea data the ML models were sensitive to domain shift, while the analytical methods were robust.
  - [engrXiv PDF](https://engrxiv.org/preprint/download/6980/11389/9697)
- 1-D Wasserstein/EMD losses for spectra are computed from cumulative distributions and are used to penalise energy shifted along the frequency axis (molecular spectra preprint, arXiv 2610.03444 **[2026-H2]**). An earlier ICML-2016 workshop paper compared a 1-D EMD loss with KL and MSE for spectral deep learning — [arXiv 2610.03444](https://arxiv.org/pdf/2610.03444); [ICML 2016 CompBio workshop PDF](https://cvhci.anthropomatik.kit.edu/~mtapaswi/papers/ICML2016_CompBioWorkshop.pdf)
- A squared-EMD loss over ordered bins (Hou, Yu & Samaras 2016) is the standard reference for EMD on ordered outputs (canonical; not re-fetched) — [arXiv 1611.05916](https://arxiv.org/abs/1611.05916)
- Search found no wave-spectrum paper that names an EMD/Wasserstein training loss. A 2025 regional wave-spectra predictor (CNN + xLSTM) used a *frequency-weighted* loss to recover high-frequency energy — [AGRIS record](https://agris.fao.org/search/en/records/68d2839db3b40a20d617fe40)

**Multi-task across Hs / Tp / direction**
- One multi-task ship-response study (DNN on cross-spectra of 6-DOF accelerations → Hs, Tp, relative heading) reports that vector-decomposing the relative heading (predicting sin/cos components) was more accurate than the alternatives — [DTU Orbit](https://orbit.dtu.dk/en/publications/on-the-determination-of-the-relative-wave-direction-based-on-meas/)
- Counter-evidence on adding inputs: in an LSTM Hs study, adding Tp and wind speed as inputs did not improve Hs — [arXiv 2201.00356](https://arxiv.org/pdf/2201.00356)
- Kendall, Gal & Cipolla (CVPR 2018) weight multi-task losses by learned homoscedastic uncertainty (canonical; not re-fetched) — [arXiv 1705.07115](https://arxiv.org/abs/1705.07115)

### Inferences
- **A spectral head on NDBC cameras would be a novel contribution.** No learned E(f) head from optical imagery appeared in the searches. The SAR/radar recipe transfers directly: 1-D swden bins as the output vector, with Hs, Tp and Tm derived from moments. The closest template is Tripathi 2024: normalised spectrum plus a separately predicted energy scale. That split separates *shape* (camera-learnable for wind sea) from *energy*, and Hs = 4√m0 follows from the scale.
- **Partitioned design.** Wang et al. 2026 is the closest analogue to this project. A learned *wind-sea* scalar parameterises a JONSWAP prior, and the network fuses it with an external *swell* spectrum. The camera would supply the wind-sea part (its WWH RMSE of 0.215 m already beats ERA5's 0.341 m), and ERA5 would supply swell. Because NDBC wind-wave and swell heights are each 4√(partition m0) split at a separation frequency, energies add: Hs² = WWH² + SwH². This is a definitional consequence that should be checked against the NDBC processing document before it is used as a hard constraint.
- **Losses.** Mass-conserving losses on spectra are cheap for 1-D bins: EMD via CDF differences, KL on normalised spectra, plus an m0 or Hs term. A moment-consistency term ties the head to the bulk targets. The Afham result is a warning that a learned residual on top of a good analytical or reanalysis prior can *lose* to the prior under domain shift. Any residual-to-ERA5 head must be checked on unseen buoys.
- Direction should be predicted as (sin, cos) or via r1/alpha1 moments, not as a raw angle.

### Gaps
- No accuracy numbers were retrieved for Wave-Spec-CNN, SAR2WV (beyond Wang's comparison table), the JMSE radar spectral CNN, or the ship-motion spectral networks (Scholcz & Mak abstract gives none).
- No source found that tests whether multi-task training (Hs + WWH + swell + Tp) beats single-task *for image inputs*. The evidence above is ship-motion and LSTM only.
- No wave-spectrum paper found using a Wasserstein/EMD loss, Gaussian-mixture spectral head, or explicit wind-sea/swell partition head trained end-to-end from images.
- Nearly all SAR spectral-head work validates against WW3/ERA5, not buoy spectra, so its accuracy against instruments is unknown.

---

## Q2. Which uncertainty methods give calibrated regression intervals when test sites are unseen, and what does conformal prediction guarantee there?

### Takeaway
No uncertainty head is calibrated at unseen sites by construction. Gaussian NLL, quantile, mixture, ensemble and evidential heads all inherit training-site miscalibration. Conformal prediction guarantees marginal coverage only under exchangeability. With unseen buoys the valid version is *group-level*: calibrate on held-out buoys, treat buoys as exchangeable draws from a population, and get marginal coverage for a new observation at a new buoy (hierarchical conformal prediction). That is not coverage at each buoy. Weighted conformal fixes covariate shift only, and Mondrian (per-site) conformal needs calibration data from that site. In practice, the defensible pipeline is: a deep ensemble (or β-NLL / quantile heads), then group-split or leave-buoy-out CQR, then reporting coverage per buoy, CRPS and PIT.

### Cited Findings

**Heads and their known failure modes**
- Heteroscedastic Gaussian NLL can give "very poor but stable" estimates. The cause is the log-likelihood loss itself: points with high predicted variance get down-weighted. Seitzer et al. (ICLR 2022) propose **β-NLL**, which weights each point's loss by its predicted variance raised to β (β = 0 recovers NLL). Across domains it improved RMSE and log-likelihood and was more robust to hyperparameters — [arXiv 2203.09168](https://arxiv.org/abs/2203.09168)
- Deep Evidential Regression is a heuristic, not exact UQ. It has gaps in its mathematical foundation; corrections and redefinitions are proposed in Meinert, Gawlikowski & Lavin, AAAI 2023 — [arXiv 2205.10060](https://arxiv.org/abs/2205.10060); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/26096)
- Follow-up by Jürgens et al.: second-order losses are hard to optimise, and evidential epistemic uncertainty is *relative, not absolute* — [arXiv 2402.09056](https://arxiv.org/abs/2402.09056)
- Original DER: Amini et al., NeurIPS 2020 (canonical; not re-fetched) — [arXiv 1910.02600](https://arxiv.org/abs/1910.02600)
- Deep ensembles (Lakshminarayanan et al., NeurIPS 2017) are a simple, scalable baseline that outperformed MC dropout in their experiments (canonical; not re-fetched) — [arXiv 1612.01474](https://arxiv.org/abs/1612.01474)
- MC dropout as approximate Bayesian inference (Gal & Ghahramani 2016; canonical; not re-fetched) — [arXiv 1506.02142](https://arxiv.org/abs/1506.02142)
- Aleatoric and epistemic split with heteroscedastic loss (Kendall & Gal 2017; canonical) — [arXiv 1703.04977](https://arxiv.org/abs/1703.04977)
- Post-hoc recalibration of regression CDFs (isotonic, on held-out data) — Kuleshov, Fenner & Ermon, ICML 2018 (canonical; not re-fetched) — [arXiv 1807.00263](https://arxiv.org/abs/1807.00263)
- CRPS as a strictly proper score for predictive distributions (Gneiting & Raftery, JASA 2007; canonical) — [DOI 10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437)
- Post-processing precedent from weather: Rasp & Lerch (MWR 2018) trained a CRPS-optimised distributional NN over station forecasts, with **station embeddings**, to post-process NWP ensembles (canonical; not re-fetched) — [arXiv 1805.09091](https://arxiv.org/abs/1805.09091)

**Calibration evidence from tabular foundation models (relevant if TabPFN is the fusion head)**
- **[2026-H2]** TabReliBench (arXiv 2610.13654, 9 Oct 2026; 81 regression datasets, up to 10,000 rows sampled per repeat, TFM context at most 5,000). Findings:
  - Uncalibrated regression TFMs usually reach near-nominal *marginal* 90% coverage with narrow intervals. The authors state that this "does not imply valid conditional coverage".
  - Worst-slab coverage is below marginal coverage for every model. TabPFNv3.5, TabICLv2 and Causilo stay closest to target on both.
  - XGBoost quantile intervals under-cover severely despite being widest.
  - Split-conformal CQR mainly helps BART; worst-slab coverage often gets *worse* after conformal calibration.
  - Covariate-adaptive conformal variants (CalLCP, RLCP, SLCP, PCP, RCP) did not consistently fix worst-slab error.
  - Accuracy does not indicate calibration.
  - TabPFNv3.5 had the best mean CRPS rank.
  - The benchmark does **not** test distribution shift.
  - [arXiv 2610.13654](https://arxiv.org/html/2610.13654v1)

**Conformal guarantees and their limits**
- Split conformal and CQR give finite-sample *marginal* coverage under exchangeability. CQR (Romano, Patterson & Candès 2019) gives adaptive-width intervals (canonical; not re-fetched) — [arXiv 1905.03222](https://arxiv.org/abs/1905.03222). For a survey, see Angelopoulos & Bates — [arXiv 2107.07511](https://arxiv.org/abs/2107.07511)
- Under out-of-distribution target domains, naive split conformal loses marginal coverage because exchangeability is violated. A corrected method is proposed, with simulation evidence only (Zou & Liu, AAAI 2024) — [arXiv 2403.19950](https://arxiv.org/pdf/2403.19950)
- Weighted conformal (Tibshirani, Barber, Candès & Ramdas 2019) restores coverage under *covariate* shift only, given the likelihood ratio dP_test(x)/dP_train(x) (canonical; not re-fetched) — [arXiv 1904.06019](https://arxiv.org/abs/1904.06019)
- Barber et al. (2023), "Conformal prediction beyond exchangeability": the coverage gap is bounded by distribution-shift terms (canonical; not re-fetched) — [arXiv 2202.13415](https://arxiv.org/abs/2202.13415)
- **Hierarchical conformal prediction (HCP)**, Lee et al.:
  - Assumes exchangeability *within* groups and *across* groups.
  - Gives distribution-free marginal coverage for a new observation from a **previously unseen group**.
  - The full-conformal variant is computationally heavy.
  - [arXiv 2306.06342](https://arxiv.org/abs/2306.06342)
- Predecessor: Dunn et al. treat two-layer hierarchical data with CDF pooling and subsampling — [arXiv 1809.07441](https://ar5iv.labs.arxiv.org/html/1809.07441)
- **[2026-H2]** Generalized HCP (Mallick, Tchetgen Tchetgen, Dobriban & Lee; arXiv 2608.15500, Aug 2026) covers the case where a *few* observations from the test group are available. It restores symmetry by assigning the test group a randomly "donated" reference size. Preprint; peer-review status unverified — [arXiv 2608.15500](https://arxiv.org/pdf/2608.15500)
- Spatial conformal (LSCP, Jiang & Xie; ICML 2026 poster): finite-sample marginal coverage under *spatial exchangeability* and asymptotic conditional coverage under stationarity and mixing, at unobserved locations — [arXiv 2412.01098](https://arxiv.org/abs/2412.01098); [ICML 2026](https://icml.cc/virtual/2026/poster/64211)
- Adaptive conformal inference (Gibbs & Candès 2021) gives long-run coverage under arbitrary shift in an online setting (canonical; not re-fetched) — [arXiv 2106.00170](https://arxiv.org/abs/2106.00170)

### Inferences
- **What is valid at unseen buoys.** Calibrate on *held-out buoys* (group split, or leave-buoy-out cross-conformal), not on held-out rows from training buoys. Under the HCP assumption that the 12 test buoys are exchangeable with the 67 training buoys, this gives *marginal* coverage averaged over buoys and observations, not per-buoy coverage.
- Two cautions follow. First, with only ~12–67 groups, the group-level quantile is coarse. Second, if test buoys were chosen non-randomly (region, hull type), the exchangeability premise fails, and the result should be stated as an assumption.
- **Mondrian per site** is impossible for unseen buoys. Mondrian conformal *by sea-state bin or by buoy class* (hull type, region, ERA5-Hs tercile) is feasible and directly targets the "few calm-sea samples" problem: report coverage per bin.
- **Weighted conformal** does not fix site shift here, because P(y|x) changes across buoys (camera mount, hull, local bathymetry), not just P(x).
- **Recommended stack.** A 5-member deep ensemble, which also addresses the large seed-to-seed variance, with β-NLL or quantile heads. Then group-split CQR. Report CRPS, PIT histograms and per-buoy and per-sea-state coverage.
- Avoid evidential regression as the primary UQ, given the Meinert and Jürgens critiques.

### Gaps
- No empirical study was found of conformal coverage specifically at unseen *buoys or wave sensors*.
- No study found comparing β-NLL, MDN, evidential and ensembles for wave-height regression.
- TabReliBench does not test shift, so TFM calibration at unseen sites is unknown.

---

## Q3. Is TabPFN v2, or a newer version, suitable as the fusion head for ~40k rows of camera-embedding features + ERA5 + wind? What do benchmarks say against GBDT?

### Takeaway
TabPFN v2 is not suitable at 40k rows: its documented limit is 10,000 training rows and 500 features. TabPFN-2.5 and 2.6 are documented up to 100k rows and 2,000 features, and TabPFN-3 and 3.5 (May and September 2026) up to 1M rows, so ~40k rows × (≤768 embedding dims + ERA5 + wind) fits. All TabPFN weights since 2.5 are under **non-commercial** licences, while code is Apache-2.0. On TabArena, TabPFN-3 and 3.5 and TabICLv2 now lead tuned GBDTs per their own reports, but there is no shift or unseen-site evidence. TabArena has only 13 regression datasets, and the 2022 Grinsztajn benchmark still favoured trees on medium-size data. Treat a TFM as one stacking-head candidate, compared head-to-head against LightGBM/CatBoost and ridge on leave-buoy-out folds.

### Cited Findings
- Prior Labs documentation of model limits and licences:

  | Model | Training rows | Columns | Weights licence |
  |---|---|---|---|
  | TabPFNv2 (Nature 2025) | 10,000 | 500 | Prior Labs License (Apache 2.0 + attribution) |
  | TabPFN-2.5 (Nov 2025) | 100,000 | 2,000 | TABPFN-2.5 License v1.1, non-commercial |
  | TabPFN-2.6 (Apr 2026) | 100,000 | 2,000 | TABPFN-2.6 License v1.0, non-commercial |
  | TabPFN-3 (May 2026) | 1,000,000 | 2,000 (Thinking: 200,000 rows) | TABPFN-3 License v1.0, non-commercial |
  | **[2026-H2]** TabPFN-3.5 (Sep 2026) | 1,000,000 | 20,000 (recommended ≤ 6,000) | TABPFN-3-5 License v1.0, non-commercial |

  - Code is Apache 2.0 throughout.
  - The docs state: "Any evaluation that influences commercial decisions or internal production planning is considered commercial use."
  - [Prior Labs docs: Models](https://docs.priorlabs.ai/models)
- The TabPFN-2.5 paper (arXiv 2511.08667) gives a headline *recommended* size of 50,000 rows and 2,000 features, benchmarked up to 100k rows. Note the discrepancy with the docs' 100k figure — [arXiv 2511.08667](https://arxiv.org/abs/2511.08667)
- Nature 2025 (Hollmann et al.): TabPFN outperforms previous methods on datasets up to 10,000 samples; "in 2.8 s" it beats an ensemble of the strongest baselines tuned for 4 h in classification. The press release stresses small tables (<10,000 rows) with outliers or missing values — [Nature DOI 10.1038/s41586-024-08328-6](https://doi.org/10.1038/s41586-024-08328-6); [Uni Freiburg release](https://uni-freiburg.de/en/new-ai-model-tabpfn-enables-faster-and-more-accurate-predictions-on-small-tabular-data-sets/)
- TabPFN-3 tech report (arXiv 2605.13986; v1 13 May 2026, v3 3 Nov 2026):
  - Scales to 1M training rows on one H100 via a reduced KV cache and row chunking.
  - Beats 8-hour-tuned GBDT baselines on datasets up to 1M rows and 200 features.
  - Up to 20× faster than TabPFN-2.5.
  - TabPFN-3-Plus (API) is more than 200 Elo above all non-TabPFN models on TabArena.
  - Pretrained only on synthetic data. The abstract does not mention regression or distribution shift.
  - [arXiv 2605.13986](https://arxiv.org/abs/2605.13986)
- **[2026-H2]** TabPFN-3.5 tech report (arXiv 2609.17895; v1 15 Sep 2026, v3 3 Nov 2026) claims a new TabArena state of the art over TabPFN-3 and all baselines. The Fast variant is up to 3× faster than TabPFN-3. The abstract mentions non-i.i.d. data, high-cardinality categoricals and wide tables, but gives no numbers — [arXiv 2609.17895](https://arxiv.org/abs/2609.17895)
- TabArena v0.1 (51 datasets: 38 classification, 13 regression):
  - Tuned and ensembled RealMLP was top (Elo ≈ 1569), ahead of LightGBM (≈ 1532), CatBoost (≈ 1489) and TabPFNv2 (≈ 1414). These Elo values were read from a table snippet without headers, so they are approximate.
  - Tuned and ensembled TabM and RealMLP match or beat GBDTs.
  - [arXiv 2506.16791](https://arxiv.org/abs/2506.16791)
- A third-party summary of TabArena leaderboard v0.1.4 lists TabPFN-3, TabPFN-2.6, RealTabPFN-2.5 and TabICLv2 as the top four, with the best GBDT about 240 Elo lower. I could not fetch the official leaderboard (it redirected to Hugging Face), so this is unverified — [Codesota](https://www.codesota.com/tasks/tabular-ml)
- TabICLv2 (arXiv 2602.11139; ICML 2026) on TabArena's 13 regression datasets:
  - TabICLv2 default had average rank 4.54, tied with RealTabPFN-2.5 (tuned + ensembled).
  - AutoGluon 1.5 extreme (4 h) ranked 3.92; LimiX 8.34; TabPFNv2 8.82.
  - TabPFN models output a predictive *distribution*, while Mitra and TabDPT output MSE point estimates.
  - [arXiv 2602.11139](https://arxiv.org/pdf/2602.11139)
- Other recent tabular foundation models (vendor reports unless noted):
  - **[2026-H2]** LimiX-2M (arXiv 2606.04485): on CTR23 regression, Mitra scored R² 0.624 versus 0.745 for LimiX-16M. The authors are LimiX developers — [arXiv 2606.04485](https://arxiv.org/abs/2606.04485)
  - **[2026-H2]** LimiX-2 (arXiv 2609.17488) had the best R² rank in TabReliBench but gives no regression predictive distribution — [arXiv 2609.17488](https://arxiv.org/pdf/2609.17488); [TabReliBench](https://arxiv.org/html/2610.13654v1)
  - **[2026-H2]** Xiaomi-TabLDM (arXiv 2609.03880) reports average rank 4.03 on 100 regression datasets, behind TabFM (3.72) and ahead of TabPFN-v3 (4.88), TabICL-v2 (6.10) and TabPFN-v2.6 (6.33). Vendor report — [arXiv 2609.03880](https://arxiv.org/pdf/2609.03880)
  - **[2026-H2]** TabDPT-Turbo (arXiv 2608.01400) — [arXiv 2608.01400](https://arxiv.org/pdf/2608.01400)
- Pre-TFM baseline: Grinsztajn, Oyallon & Varoquaux (NeurIPS 2022 D&B) found tree-based models still beat deep nets on medium-sized (~10k-row) tabular data. They attribute this to NN smoothness bias and sensitivity to uninformative features (canonical; not re-fetched) — [arXiv 2207.08815](https://arxiv.org/abs/2207.08815)
- FT-Transformer (Gorishniy et al., NeurIPS 2021) is a strong deep tabular baseline, but GBDT remained competitive (canonical; not re-fetched) — [arXiv 2106.11959](https://arxiv.org/abs/2106.11959)
- Stacking without leakage: scikit-learn's stacking trains the final estimator on *cross-validated* predictions of the base estimators (cross_val_predict), not in-sample fits. GroupKFold keeps a group (here, a buoy) entirely in one fold (canonical docs) — [scikit-learn ensemble/stacking](https://scikit-learn.org/stable/modules/ensemble.html#stacking); [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)

### Inferences
- **Fit.** Around 40k rows with ~800 features (768-d DINOv3 ConvNeXt-T embedding + ERA5 swh/shww/swell + wind + time/metadata) is inside TabPFN-2.5/2.6 limits and well inside TabPFN-3/3.5.
  - The 2.5 paper's recommended 50k is close, so prefer 3.x or reduce the embedding (PCA to 64–128 dims) before handing it to a TFM.
  - The camera *prediction* (OOF Hs/WWH) is a far more informative single feature than 768 raw dims.
  - Academic research use fits the non-commercial licence, but the licence text must be checked before any operational or partner deployment.
- **Main risk is site shift, not size.** In-context learners treat the context rows as i.i.d. draws. At an unseen buoy, every context row comes from other buoys, so the TFM cannot "learn the site". Benchmarks (TabArena, TabReliBench) are i.i.d. splits. Any TFM gain must be shown with leave-buoy-out.
- **Leakage protocol for stacking.**
  1. Camera-model features for the combiner must be *out-of-fold by buoy*: GroupKFold over the 67 training buoys, with the backbone fine-tuned per fold or at least the head refit per fold.
  2. The combiner is trained on those OOF predictions.
  3. The 12 test buoys are touched once.
  - Using in-sample camera predictions for the 67 training buoys would make the combiner over-trust the camera. That is a plausible reason simple fusion failed to help total Hs.
- With only 13 regression datasets in TabArena and no geoscience shift tests, the TFM-vs-GBDT verdict for this problem is open. A small, cheap ablation (ridge / LightGBM / CatBoost / TabPFN-3 on identical OOF features, leave-buoy-out, ≥ 5 seeds) is the deciding experiment.

### Gaps
- Could not fetch the official TabArena leaderboard or the regression-only view.
- No TabPFN-3/3.5 regression-specific numbers were verified.
- No published evaluation was found of TFMs under group or site shift.
- The exact TabPFN-3.5 regression output format (bar distribution or otherwise) was not verified from primary text.

---

## Q4. What fusion designs (residual-to-prior, FiLM, cross-attention, gated) have evidence for combining imagery with model or reanalysis priors in geoscience?

### Takeaway
The most direct geoscience evidence is for **prior-structured fusion**: feed the model or reanalysis prior in and let the network learn the correction. The SAR example is a JONSWAP prior from a learned wind-sea height, fused with the observed swell spectrum by a U-Net, which beat the operational product. The spectrum example is a residual on an analytical baseline, which beat pure ML in-domain but lost robustness under domain shift. For generic image + tabular fusion, benchmarks find late fusion or concatenation hard to beat. DAFT/FiLM-style conditioning helps modestly in medical imaging. I found no geoscience head-to-head of FiLM versus cross-attention versus gated fusion.

### Cited Findings
- SAR prior fusion (Wang et al. 2026): XGBoost wind-sea height → JONSWAP prior → U-Net fusing the prior with the OSW swell spectrum (L1 versus ERA5). Hs RMSE 0.4026 m, Tm 0.4342 s, direction 20.42°. It "completely eliminates the artifacts" of the standard product, per the authors' highlights — [Remote Sensing 18:1703](https://doi.org/10.3390/rs18111703)
- Residual-to-analytical-prior for directional spectra (Afham & Chandrasena) beat pure ML and residual-without-physics. However, analytical MEM and NDBC methods stayed most accurate, and ML was sensitive to domain shift on unseen North Sea data — [engrXiv](https://engrxiv.org/preprint/download/6980/11389/9697)
- ML post-processing of ERA5 Hs at buoy sites: two DNN types (including an LSTM) reduced ERA5 RMSE and raised correlation at two locations (Ocean Engineering 2023; abstract only) — [NOAA repository](https://repository.library.noaa.gov/view/noaa/55796)
- Related forecast-correction work: WaveUformer corrects an AI wave model's 24–240 h Hs forecasts, lowering annual mean RMSE — [Frontiers Mar. Sci. 2026](https://public-pages-files-2025.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2026.1732870/pdf); a deep-learning real-time bias correction for western North Pacific Hs forecasts — [arXiv 2311.15001](https://arxiv.org/pdf/2311.15001)
- Ship motion, hybrid conditioning: CNN-predicted integral parameters condition a Bayesian spectral estimator, which clearly beat the unconstrained estimate on 2 years of container-ship data — [DTU Orbit](https://orbit.dtu.dk/en/publications/wave-spectrum-estimation-conditioned-on-machine-learning-based-ou/)
- FiLM (Perez et al., AAAI 2018) applies feature-wise affine modulation of CNN features from a conditioning vector (canonical; not re-fetched) — [arXiv 1709.07871](https://arxiv.org/abs/1709.07871)
- DAFT (Pölsterl et al., MICCAI 2021), applied to 3-D MRI + tabular, beat concatenation baselines and FiLM overall. FiLM was marginally ahead on diagnosis and DAFT ahead on time-to-dementia (authors' own results) — [arXiv 2107.05990](https://arxiv.org/pdf/2107.05990)
- Multimodal AutoML "bag of tricks" (image/text/tabular): late fusion generally beats early fusion, with an aligned late-fusion variant best except on text+tabular — [arXiv 2412.16243](https://arxiv.org/pdf/2412.16243)
- A fusion comparison across five image–tabular datasets found concatenation and DAFT the most robust — [chatpaper summary of TIME](https://chatpaper.com/paper/145072)
- VT-Bench (arXiv 2605.08146, May 2026) is a unified visual–tabular benchmark with DAFT among its baselines; rankings not retrieved — [arXiv 2605.08146](https://arxiv.org/pdf/2605.08146)
- Station/site embeddings in NN post-processing of NWP (Rasp & Lerch 2018; canonical; not re-fetched) — [arXiv 1805.09091](https://arxiv.org/abs/1805.09091)

### Inferences
- **Residual-to-prior with a gate is the natural first design.** Predict ΔHs (or Δ log-energy per spectral bin) relative to ERA5, from camera embedding + ERA5 + wind. Add a learned gate α(x) ∈ [0, 1] between the camera estimate and ERA5, per partition (wind sea versus swell).
  - This hard-codes the observed pattern: camera wins on WWH, ERA5 wins on swell.
  - It degrades gracefully to ERA5 when the camera is uninformative (night, fog, glare).
  - The Afham result says to test it on unseen buoys and keep a fall-back to the prior.
- **FiLM.** Conditioning the image backbone on ERA5 and wind is cheap and gives the backbone a prior on what to look for. Medical-imaging evidence shows only small gains over concatenation, so it should be an ablation, not the main bet.
- **Cross-attention** between six camera views, with the met-ocean vector as a query token, is plausible for the six-view strips. I found no geoscience evidence for or against it.
- **Do not add a buoy-ID embedding.** It cannot generalise to unseen buoys. Use transferable metadata instead: hull type, camera height if known, latitude, distance to coast or depth.
- The failure of simple fusion on total Hs is consistent with the combiner being trained on non-OOF camera predictions; see the leakage protocol in Q3. It is also consistent with the combiner not separating wind-sea and swell regimes. A partition-aware combiner (WWH from camera+ERA5, swell from ERA5, Hs = √(WWH² + swell²)) is a direct test.

### Gaps
- No geoscience study was found that compares FiLM, cross-attention and gated fusion head-to-head for imagery + reanalysis.
- Full text of the ERA5-buoy LSTM post-processing paper was not retrieved, so its RMSE numbers and site generalisation are unknown.

---

## Q5. Do ordinal heads beat regression-then-bin for ordered classes like sea state or Beaufort?

### Takeaway
There is no wave-specific evidence either way. In general vision benchmarks, CORN beats CORAL, and soft-label cross-entropy (SORD) is a strong, simple ordinal baseline. Because this project has continuous instrument labels (Hs) whose bins are defined by fixed thresholds, regression (or a distributional head) followed by thresholding retains more information and lets the bins be changed later. Ordinal heads mainly matter when labels are *only* ordinal. Compare both on the same macro-F1 and MAE-in-classes metrics used by the target papers.

### Cited Findings
- CORN (Shi, Cao & Raschka; arXiv 2111.08851, Pattern Analysis and Applications) removes CORAL's weight-sharing constraint and trains K−1 conditional binary tasks, with rank-consistent probabilities via the chain rule. The authors report improvements over CORAL across datasets. Code: Raschka-research-group/corn-ordinal-neuralnet — [arXiv 2111.08851](https://arxiv.org/abs/2111.08851)
- A 2026 AIMECON study citing CORN found it had the smallest or comparable average MAE and RMSE among the ordinal methods it tested — [ACL Anthology 2026.aimecon-main.55](https://aclanthology.org/2026.aimecon-main.55.pdf)
- CONDOR (Jenkinson et al.) argues that CORAL has theoretical and practical limits and that CORN-style methods can still produce rank-inconsistent outputs in some cases. It proposes a universally rank-consistent alternative — [arXiv 2110.07470](https://ar5iv.labs.arxiv.org/html/2110.07470)
- CORAL (Cao, Mirjalili & Raschka, Pattern Recognition Letters 2020): shared-weight K−1 binary outputs with rank-consistency guarantee (canonical; not re-fetched) — [arXiv 1901.07884](https://arxiv.org/abs/1901.07884)
- SORD (Díaz & Marathe, CVPR 2019): soft labels from a distance-based softmax over class ranks, trained with ordinary cross-entropy. A simple ordinal method that performed competitively with specialised ordinal losses in the authors' benchmarks (canonical; not re-fetched) — [CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2019/html/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.html)
- Squared-EMD loss over ordered classes (Hou et al. 2016; canonical) — [arXiv 1611.05916](https://arxiv.org/abs/1611.05916)
- Wave-specific classification precedent: CNN+LSTM wave-height *classes* from monoscopic video (ADCP labels); single-snapshot models failed in sea states 4–5 — [Ocean Engineering](https://www.sciencedirect.com/science/article/abs/pii/S0029801823023867)

### Inferences
- With continuous Hs labels, a regression or distributional head followed by binning is the principled default. With a predictive distribution (Q2), class probabilities come for free as P(Hs ∈ bin), which gives a calibrated ordinal output without a separate head.
- If a paper being compared uses a class head, fair comparison means running both:
  1. A CORN or SORD head trained on binned labels.
  2. The regression or distributional head binned post hoc.
  - Report macro-F1 and ordinal MAE on the same unseen buoys.
- SORD over fine Hs bins is close to a histogram regression head (as in TabPFN's bar distribution). It is a cheap way to get a distribution and an ordinal output from one head.

### Gaps
- No study found that compares ordinal heads with regression-then-bin for sea state or Beaufort classes.
- Per-dataset CORN-versus-CORAL numbers were not retrieved.
- The claim that regression-then-bin dominates when continuous labels exist is my inference, not a sourced benchmark result.
