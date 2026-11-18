# New camera/video wave-estimation papers (2024 – Nov 2026) and new foundation encoders (Jun – Nov 2026)

Notes current to 2026-11-18. Evidence levels used below:
- **[FT]**: full text, or the relevant section of it, read.
- **[ABS]**: abstract read from the publisher, arXiv or an index.
- **[SNIP]**: search-engine snippet or summary only.
- **[META]**: title and bibliographic metadata only (Crossref or Semantic Scholar).

Search coverage caveats:
- Elsevier (ScienceDirect) pages returned HTTP 403 to every fetch, and Crossref and Semantic Scholar withhold Elsevier abstracts. Several 2026 *Ocean Engineering* camera papers below are therefore known only by title, authors and volume.
- Semantic Scholar and OpenAlex search were rate-limited (HTTP 429). Discovery of papers relied on Crossref, the arXiv API, web search, and the reference list of Kamagata et al. 2026 (arXiv:2607.11998).

---

## Q1. Which new camera-based wave papers (2024 – Nov 2026) exist beyond the known list, and what architectures, splits and results do they report?

### Takeaway
At least four 2026 *Ocean Engineering* papers estimate wave parameters directly from cameras and were not in the known list:
- a 3D-CNN with sim-to-real transfer (Liu, Qin & Ti)
- a multi-domain model trained on reconstructed-3D data (Li et al.)
- a comparison of coastal-camera spectral and plain image inputs (Liu, Xu … Dong)
- a polarimetric ViT (PAMS-ViT)

Three further items:
- a 2025 *Scientific Data* video dataset for breaker type, built largely from **WebCOOS cameras** (ResNet-18 + LSTM baseline, 96.1% with 5-fold CV)
- a 2026 cross-domain ordinal Beaufort classifier (BGO-CDF)
- a December 2025 stereo neural surface-reconstruction network

The ship-as-a-wave-buoy literature grew quickly in 2025–2026: transformer/attention hybrids, frequency-domain modules and class-imbalance losses, nearly all of it evaluated within one ship or in simulation. In most of the papers, headline metrics come from random or k-fold splits. Where a held-out domain is tested, performance drops sharply.

### Cited Findings

#### A. Camera / video → wave parameters (new, not in the known list)

| Paper (venue, date) | Platform / input | Architecture | Labels and instrument | Split | Headline result | Evidence |
|---|---|---|---|---|---|---|
| Liu H., Qin J., Ti Z.-L., "Quantitative video-based wave parameter estimation using a 3D-CNN and two-stage transfer learning framework", *Ocean Eng.* 352:124561 (Apr 2026) | Video (details unknown) | 3D-CNN with two-stage simulation-to-real transfer for H_s and T_p (per citing paper) | Unknown | Unknown | Kamagata et al. cite it as "R² above 0.80 from several hundred annotated video samples". They also say it "needs substantial labeled data" and is "limited to specific classes rather than regression". That characterisation is internally odd given the R², so treat it as unverified. | [META] [Crossref](https://doi.org/10.1016/j.oceaneng.2026.124561); description from [Kamagata et al. 2607.11998 related work](https://arxiv.org/html/2607.11998v1) |
| Li F., Bu W.-H., Qin R.-F., Chen J., "Multi-domain deep learning architectures for vision-based sea wave parameter estimation utilizing reconstructed 3D datasets", *Ocean Eng.* 364:126780 (Aug 2026) | Vision; training data built from reconstructed 3D wave surfaces | "Multi-domain" DL architectures (details unknown) | Labels derived from 3D reconstruction (implied by title) | Unknown | The abstract opens on "scarcity of large-scale, accurately labeled visual datasets" and concludes that multi-domain learning is practical for vision-based wave measurement. No numbers retrieved. | [SNIP] [search listing](https://www.sciencedirect.com/science/article/abs/pii/S0029801826003513); [Crossref](https://doi.org/10.1016/j.oceaneng.2026.126780) |
| Liu Y.-L., Xu G.-J., Xie H.-R., Chen H.-X., Yang J.-X., Liu B., Dong C.-M., "Nearshore wave height estimation using coastal cameras: comparing physics-informed spectral and conventional image representations", *Ocean Eng.* 366:127543 (Oct 2026) | Coastal camera | Compares a physics-informed spectral input (image FFT/spectrum) with raw-image input | Unknown | Unknown | No abstract retrievable | [META] [Crossref](https://doi.org/10.1016/j.oceaneng.2026.127543) (PII S0029801826033779) |
| Ahammed M.I., Dang F.-Y., "Polarization-aware multi-scale vision transformer for significant wave height estimation" (PAMS-ViT), *Ocean Eng.* (Oct 2026), 127753 | Raw polarimetric images from an **open lab dataset** with "nominal Hs in the centimeter range" | ViT with a Learnable Stokes Fusion module (learned polarization-channel fusion) and Multi-Scale Polarimetric Tokenization (fine, intermediate and coarse tokens) | Lab H_s (instrument not stated) | **12-fold leave-one-wave-train-out CV**, described as leakage-free | Best pooled RMSE "0.134 cm". Ablation: learned channel fusion beats fixed analytic Stokes transforms. | [ABS] [Michigan Tech Digital Commons](https://digitalcommons.mtu.edu/michigantech-p2/2959/) |
| Liu J., Zhou Z., Yan J., Guo A., Benetazzo A., Li H., "Neural reconstruction of 3D ocean wave hydrodynamics from camera sensing", arXiv:2512.06024 (4 Dec 2025) | Stereo cameras, real sea | Attention-augmented pyramid network with global multi-scale attention, plus physics constraints for 3D velocity fields | Stereo-vision dataset (reference reconstruction) | Not stated; claims generalisation under occlusion | "millimetre-level wave elevation" in the central region; dominant-frequency error < 0.01 Hz; 2 million points in 1.35 s | [ABS] [arXiv](https://arxiv.org/abs/2512.06024) |
| Wu J. et al., "Research on wave measurement and simulation experiments of binocular stereo vision based on intelligent feature matching", *Front. Mar. Sci.* 11:1508233 (2024) | Binocular stereo | "Intelligent" (learned) feature matching | Unknown | Unknown | Not retrieved | [META] cited in [2607.11998 refs](https://arxiv.org/html/2607.11998v1) |
| Vieira M., Guedes Soares C., Guimarães P., Bergamasco F., Campos R., "Nearshore space-time ocean wave observation using low-cost video cameras", *Coastal Eng.* 104694 (online Dec 2024, 2025 issue) | Low-cost video cameras (stereo, WASS group) | Probably classical stereo, not DL (unverified) | Unknown | Unknown | Not retrieved | [META] [Crossref](https://doi.org/10.1016/j.coastaleng.2024.104694) |
| "Deep Learning-Based Waterline Detection Applied to Wave Period Measurement in the Nearshore Swash Zone", *Remote Sens.* 18(9):1385 (30 Apr 2026) | UAV imagery of the swash zone | DeepUNet waterline segmentation, then runup and wave period | Unknown | Unknown | DeepUNet pixel segmentation F1 = 75.0% | [SNIP] [doi](https://doi.org/10.3390/rs18091385) |
| "Nearshore Bathymetry and Waves Estimates from UAV Video", Coastal Dynamics 2025 (Springer chapter) | 14-min UAV video, Torrey Pines; water surface from sky glint | Not clearly DL | In-situ sensor | n/a | H_s 1.93 m vs 1.47 m from the sensor (+31%) | [SNIP] [Springer](https://link.springer.com/chapter/10.1007/978-3-032-15477-4_65) |

New facts on papers already in the known list:
- **Kamagata et al. arXiv:2607.11998.**
  - Pearson r is 0.451 (H_s), 0.578 (H_max), 0.643 (T_p), 0.680 (T_z) and 0.832 (direction) — [SNIP of abstract](https://arxiv.org/abs/2607.11998).
  - The held-out test set is "five plunge-breaker videos" from **Oak Island East, Oak Island West and Xishawan C4**, taken from Yin et al.'s *Scientific Data* video set. Test RMSE is 0.5863 m, PCC 0.436 and **R² 0.107**, against RMSE 0.562 m on the full set.
  - The page states "66 annotated training scenes" (894 clip windows) in one place and "six" scenes elsewhere.
  - No label instrument is stated.
  - Source: [FT, results section](https://arxiv.org/html/2607.11998v1).
- **Kamagata et al. arXiv:2606.13302.**
  - The pipeline has three stages: synthetic pretraining, then "silver label" adaptation, then expert fine-tuning. The region of interest comes from temporal pixel variance, and a physics-guided frequency mask is applied.
  - **LtViViT gave the highest accuracy, while TinyWaveNet (recurrent-convolutional) had better temporal stability and "oceanographic skill"** — [ABS](https://arxiv.org/abs/2606.13302).
  - Its reference list cites "On the nearshore significant wave height inversion from video images based on deep learning, *Applied Ocean Research* 154, 104358 (2025)". This title is identical to Xu et al. 2024 *JMSE* 12:2003, so it is probably a mis-citation (see Gaps) — [search result](https://arxiv.org/pdf/2606.13302).
- **Yang et al. 2026 and S0029801826003513 are probably one paper.** Semantic Scholar lists "A deep learning method for spatiotemporal significant wave height estimation with ship attitude compensation" (Xin-De Yang, Hong-Dong Wang, Run-Tao Zhang, Shi-Ping Sun, Ming-Yang Zhang) as *Ocean Eng.* 124517, April 2026, which is the same volume range as 352:124561 — [S2 batch lookup, doi](https://doi.org/10.1016/j.oceaneng.2026.124517); [ScienceDirect listing](https://www.sciencedirect.com/science/article/abs/pii/S0029801826003513). The known list's "Yang et al. 2026 (OE 352)" and "OE 2026 ship image+attitude network (S0029801826003513)" are likely the same paper (volume not directly confirmed).

#### B. Wave breaking, whitecaps, runup and rip currents from images/video (architecture-informative)

**Yin H., Cai F., … Chen X., "A Video Dataset for Nearshore Wave Breaking Type Classification", *Sci. Data* 12:1722 (30 Oct 2025)** [FT via Europe PMC PMC12575733] — [Nature](https://www.nature.com/articles/s41597-025-06005-5); [Europe PMC full text](https://europepmc.org/article/PMC/PMC12575733)
- **Data.** 9,000 clips, each 30 s, from 15 cameras at six sites. Three classes, 3,000 each: spilling, plunging, surging.
- **Cameras.** Camera IDs include **currituck_hampton_inn, jennette_north/south, oakisland_east/west, horace_caldwell and tmmc_prls**. These are the WebCOOS camera names, at NC, TX and CA sites. The remaining cameras are Argus-style systems at Xisha Bay (Quanzhou) and Huangcuo (Xiamen) in China.
- **Raw video.** 5–20 fps (e.g. 2688×1520 at 20 fps; 1440×1080 at 5 fps). Clips are cropped to 224×224. Set_B holds frames at 2 Hz, 60 per clip, 540,000 frames in total.
- **Labels.** Human, from a two-stage annotation; about 1,000 ambiguous clips were dropped. No instrument labels.
- **Baseline.** CRNN: ImageNet ResNet-18 encoder → LSTM → FC; Adam, lr 0.001, dropout 0.3.
- **Split.** 5-fold cross-validation over clips. This is lenient, because sites and cameras are shared across folds. The authors use "temporal staggering and spatial displacement" so that the same wave event is not repeated.
- **Result.** Training accuracy > 98.5%; mean validation accuracy **96.1%**.

Other items:
- **WaBViC: "Objective wave breaker type classification from flume video using 3D CNNs across multiple plane slopes"** (*Applied Ocean Research*, 2026; PII S0141118726003664). A 3D-CNN on flume video, trained on a balanced set of laboratory breaker events — [SNIP](https://www.sciencedirect.com/science/article/pii/S0141118726003664).
- **"Autonomous Classification of Wave Breaker Type in a Large Wave Flume"**, *Coastal Eng.* 204:104902. A YOLO plus random-forest pipeline with 94% accuracy; data on Dryad — [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.3ffbg79w1).
- **ICCE 2024 proceedings (published May 2025)** [META]:
  - "Video-based flood and swash mapping – an application of deep learning to nearshore research" — [doi](https://doi.org/10.9753/icce.v38.currents.23)
  - "A generalized tool for wave-by-wave estimation of nearshore wave breaking dissipation from optical imagery" — [doi](https://doi.org/10.9753/icce.v38.waves.50)
  - "Deep learning to unravel nonlinearity in wave breaking using wave flume video images" — [doi](https://doi.org/10.9753/icce.v38.waves.54)
- **Foam/breaking segmentation over a tidal cycle** with DeepLabv3+ (4 classes: breaking foam, seepage face, unsaturated beach, background), *Remote Sens.* 16(24):4616 (2024) — [SNIP](https://doi.org/10.3390/rs16244616).
- **Runup:**
  - Kang et al., *ESurf* 12:1 (2024): DeepLab v3+ on ResNet-18 water segmentation, more than 500 hand-labelled images — [SNIP](https://esurf.copernicus.org/articles/12/1/2024/).
  - van der Grinten et al., *Coastal Eng.* 2025: color-contrast and ML runup methods on a dissipative beach; the companion TU Delft thesis uses a CNN on color-contrast-preprocessed images, cross-validated on 9 hand-labelled series — [SNIP](https://repository.tudelft.nl/record/uuid:8882db80-2627-4762-89ab-19daaeb163a0).
- **Rip currents:**
  - RipVIS, a video instance-segmentation benchmark, CVPR 2025 — [CVF](https://openaccess.thecvf.com/content/CVPR2025/papers/Dumitriu_RipVIS_Rip_Currents_Video_Instance_Segmentation_Benchmark_for_Beach_Monitoring_CVPR_2025_paper.pdf).
  - Chuang, Chen & Lien, *JMSE* 14(5):496 (Mar 2026): YOLOv4 plus **adaptive time-averaged images** to suppress false positives, with lifeguard verification — [ABS via S2](https://doi.org/10.3390/jmse14050496).
  - *PLOS One* (29 Oct 2026): UAV rip detection with optical flow and "almost local-isometric embedding on the sphere" — [META](https://doi.org/10.1371/journal.pone.0360672).

#### C. Sea-state classification from images (ship/shore cameras)

- **Zhang G.-Y., Wang Z.-H., Zhang Y., Wang Y.-H., Cang J., "Beaufort-geometry ordinal learning for cross-domain sea-state image classification" (BGO-CDF), *Ocean Eng.* 367:128212 (Oct 2026)** [SNIP] — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0029801826040461); [Crossref](https://doi.org/10.1016/j.oceaneng.2026.128212)
  - Learns a Beaufort-ordinal cumulative distribution.
  - Evaluated under **eight domain-separated protocols**.
  - Argues that exact accuracy misleads and that MAE, within-one-class accuracy and quadratic weighted kappa should be reported.
  - No numbers retrieved.
- **UNIRI-SeaState dataset v1.0** (Zenodo, 15 Apr 2026, CC BY 4.0, 6.8 GB): eight Beaufort-linked classes, a fixed test set and multiple training splits. It accompanies Pobar et al., *IEEE OJ-ITS* 7:136–154 (2026), which is in the known list. New fact: **the data are public** — [Zenodo](https://zenodo.org/records/19592919).
- **Vorkapić et al.:**
  - Compares ResNet-101, NASNet, MobileNet_v2 and ViT-B/32. ResNet gave the best overall accuracy, with ViT close behind.
  - **Cross-dataset collapse:** a model trained on another dataset (ResNet-Mu-SSiD) fell from 78% to 31% accuracy on UNIRI-SeaState-LL.
  - Sources: [SNIP](https://www.researchgate.net/publication/382065867_A_Computer_Vision_Approach_to_Estimate_the_Localized_Sea_State); [arXiv 2407.03755](https://arxiv.org/abs/2407.03755).
- **"Ensemble of Vision Transformers for Sea State Classification"** (Authorea preprint, about 2025):
  - Fine-tuned DeiT reached F1 66% and EfficientNet 65% on eight Beaufort classes.
  - The image-only ensemble reached 73%; image plus wind-speed reached 97%.
  - Source: [SNIP](https://www.authorea.com/doi/full/10.22541/au.176340986.61610309/v1).
- **SeaClips** (WACV 2026): a maritime object-detection video dataset with video-level sea-state labels; about a third of its videos are "wavy" — [CVF](https://openaccess.thecvf.com/content/WACV2026/papers/Denk_SeaClips_A_Video_Dataset_for_Maritime_Object_Detection._WACV_2026_paper.pdf).

#### D. Ship / USV as a wave buoy (motion/IMU → sea state with deep learning)

| Paper | Input | Architecture | Labels | Split / domain | Result | Evidence |
|---|---|---|---|---|---|---|
| Majidiyan H., Enshaei H., Howe D., "Direct-transfer evaluation of a transformer-based framework for cross-vessel sea-state estimation through wave buoy analogy", *Ocean Eng.* 126792 (Aug 2026) | Ship motions (incl. roll), speed | Dual-line Transformer; cross-attention between global motion and direction-sensitive roll features; speed-based feature scaling | Sea-state classes (H_s, direction) | Within-vessel validation, plus **direct transfer to unseen vessels** | Within-vessel: H_s accuracy 98.72% (bidirectional variant), direction 91.67%. **Unseen mid-range vessel: 76.2% H_s, 71.4% direction. Out-of-envelope VLCC: 8% and 40%.** | [SNIP of abstract](https://trid.trb.org/View/2725508); [doi](https://doi.org/10.1016/j.oceaneng.2026.126792) |
| Guo X.-Y., Mao R.-Z., Han P.-H., Xu J.-Y., Zhang H.-X., "Spectrum-regularized deep learning with joint wave parameter prediction…", *Ocean Eng.* 128186 (Nov 2026) | Ship motions | DL with a spectrum regularizer; joint H_s, direction, T_p plus an auxiliary sea-state class | Unknown | Unknown | Not retrieved | [META/SNIP](https://doi.org/10.1016/j.oceaneng.2026.128186) |
| Duan W. et al., "A physics-constrained deep learning framework for directional wave spectrum estimation from ship motion data", *Ocean Eng.* 127017 (Aug 2026) | Ship motions | Physics-constrained DL → directional spectrum | Unknown | Unknown | Not retrieved | [META](https://doi.org/10.1016/j.oceaneng.2026.127017) |
| Choi J.-h., *Ocean Eng.* 363 (15 Aug 2026) | Ship-motion time series or spectral features | Nonparametric physics-based spectral optimisation **vs** a CNN multivariate regression | Hindcast | Unknown | Not retrieved | [SNIP](https://trid.trb.org/View/2721069) |
| Gu J. et al., "A Robust Sea State Measurement Method Using Ship Motion Sensors: A Diffusion-Augmented Deep Learning Approach", *IEEE TIM* (2026) | Ship-motion sensors | **LSTransformer** (long short-term transformer) + multi-scale feature extraction; a diffusion model synthesises minority-class signals | Three SSE datasets | Unknown | "consistently outperforming SOTA" (no numbers in abstract) | [ABS](https://doi.org/10.1109/tim.2026.3712906) |
| Xia W., Wang K.-X., Yao Y., Cheng X., Chen S.-Y., "A Temporal-Spectral Mixer…", *IEEE T-ITS* (Jul 2026) | Ship motion | Spectral Frequency Adaptive module + Temporal Multi-Scale Parallel Convolution + imbalanced contrastive clustering loss | Public benchmarks and ship data | Unknown | SOTA claimed | [ABS](https://doi.org/10.1109/tits.2026.3673344) |
| Wang K.-X. et al., "Temporal-channel decoupled learning … under class imbalance", *EAAI* 115767 (Oct 2026) | Ship motion | Temporal/channel-decoupled network | Unknown | Unknown | Not retrieved | [META](https://doi.org/10.1016/j.engappai.2026.115767) |
| Ni Y.-Y. et al., "Comparative Study of ML Models for Instantaneous Wave-Height Estimation Using 3-DOF Ship Motion Responses", *JMSE* 14(13):1158 (Jun 2026) | Heave, roll, pitch of a 1:2 planing craft in regular head seas | SVR, TCN, LSTM, CNN-LSTM, Transformer, LSTM-MHA under one protocol | Wave probe near the centre of gravity (instrument); STAR-CCM+ plus towing tank | Simulation and towing-tank domains | Simulation: **LSTM-MHA best (RMSE 0.001231 m, R² 0.997848)**; TCN near-optimal with fewer parameters and lower latency. Tank: LSTM-MHA and TCN clearly beat SVR. Accuracy in simulation, robustness in the tank and cross-domain transfer "not fully consistent". | [ABS](https://doi.org/10.3390/jmse14131158) |
| Shi F., Liu X., Liu M., Cheng X., *Ship As Wave Buoy: Data-Driven Sea State Estimation Based on Ship Motion Data* (Springer Ser. Naval Arch., 2026, 225 pp.) | Ship motion | DenseNet-style CNNs, prototype classifiers, multi-scale features, adversarial transfer, dynamic graph networks; adversarial-robustness chapter | Public benchmarks and ship datasets | Includes cross-ship and cross-loading transfer | n/a | [SNIP](https://www.schweitzer-online.de/en/ebook/Cheng/Ship-Wave-Buoy-Data-Driven-Sea-State-Estimation-Based-Ship-Motion/9789819567423/A75953903/) |
| Liu K., Wei Y., Lu Z., "Estimating wave height based on feature fusion of ship motion response and X-band radar images", *Front. Remote Sens.* 6 (12 Nov 2025) | **Simulated** heave/pitch (S175 hull) plus simulated X-band radar | **Not DL**: least-squares fusion of significant heave, significant pitch and radar SNR | Simulation input sea state | **Random 1:1 split** of 210 samples | RMSE 0.14–0.26 m (CC ≈ 0.99) vs 1.08 m for SNR only | [FT](https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2025.1693042/full) |
| Ait Habouche M. et al., "Machine Learning-Based Estimation Of Wave Direction For Unmanned Surface Vehicles", arXiv:2412.16205 (Dec 2024) | USV sensor data | LSTM | Unknown | Unknown | "outperforming simpler baselines" | [ABS](https://arxiv.org/abs/2412.16205) |

Reviews:
- Lopac N., Severinski K., Palaić D., Lerga J., "Machine learning for ship-motion-based sea-state estimation", *Ocean Eng.* 126362 (Jul 2026). Its summary says H_s and period are estimated more reliably than direction and spectral shape — [doi](https://doi.org/10.1016/j.oceaneng.2026.126362); [SNIP](https://airi.uniri.hr/research/machine-learning-for-ship-motion-based-sea-state-estimation/).
- Li X., Mao R.-Z., Han P.-H., Zhang H.-X., systematic review, *Ocean Eng.* 126747 (Aug 2026) — [doi](https://doi.org/10.1016/j.oceaneng.2026.126747).

Further ship-motion SSE papers by the Cheng/Chen/Liu group [META] — [Crossref listing](https://doi.org/10.1109/tits.2025.3555792):
- *IEEE TIM* 2025, lightweight frequency MLP
- *IEEE TASE* 2025, wavelet-DCT
- *IEEE T-ITS* 2025, adaptive multi-scale features plus learnable prototypes
- *IEEE TCYB* 2026, time-frequency collaborative learning with missing labels

### Inferences
- **Recurring camera architectures:**
  - **ImageNet CNN + LSTM (CRNN):** Kim 2023 (GoogLeNet+BiLSTM); Yin 2025 (ResNet-18+LSTM).
  - **3D-CNN:** Liu/Qin/Ti 2026; WaBViC flume.
  - **ViT/ViViT:** Kamagata 2606.13302 (LtViViT); PAMS-ViT.
  - **Self-supervised video ViT plus a two-stream temporal model:** Kamagata 2607.11998 (V-JEPA ViT-S + SlowFast + optical flow).
  - Segmentation nets (DeepLabv3+, U-Net variants) dominate runup, foam and swash work.
- **Head-to-head results found:**
  - LtViViT beat TinyWaveNet on accuracy but lost on temporal stability (2606.13302).
  - ResNet-101 was at least as good as ViT-B/32 for Beaufort stills (Vorkapić).
  - DeiT ≈ EfficientNet (66 vs 65 F1) for Beaufort stills.
  - For ship-motion wave height, LSTM-MHA > TCN > SVR, with TCN best for efficiency (Ni 2026).
  - Learned polarimetric fusion beat fixed Stokes transforms (PAMS-ViT).
  - None of these comparisons includes a modern frozen foundation encoder against a supervised CNN on field data with instrument labels.
- In the ship-motion SSE literature, frequency-domain branches, attention or transformer temporal blocks, and class-imbalance losses are now standard. Cross-vessel transfer is the open weakness. This is directly analogous to cross-buoy or cross-camera transfer for the project's horizon-tilt motion signal.
- The Yin et al. dataset is built on WebCOOS cameras the project already ingests (Oak Island, Jennette's Pier, Currituck, Horace Caldwell, TMMC). It could serve as a breaker-type auxiliary or pretraining task on the same viewpoints. Its labels are visual, not instrumental.

### Gaps
- Abstracts and metrics for Liu/Qin/Ti 2026, Li et al. 2026, Liu Y.-L. et al. 2026, BGO-CDF 2026, Guo et al. 2026, Duan et al. 2026, Choi 2026 and WaBViC could not be retrieved: ScienceDirect returned 403 and Elsevier abstracts are withheld from open indexes. Their inputs, splits, label instruments and RMSE remain unknown.
- The "*Applied Ocean Research* 154, 104358 (2025)" citation in 2606.13302 could not be matched to a real paper. It is probably a mis-citation of Xu et al. 2024 *JMSE*.
- No 2024–2026 deep-learning paper using **buoy-mounted cameras** (as in Choi 2020) was found apart from the known list. No drone/USV camera DL paper with buoy-validated H_s was found either.
- Parameter counts and hardware were not available for any of the new camera papers.

---

## Q2. Are there any held-out-site results against instruments, and which architectures produced them?

### Takeaway
Very few, and they are weak. The only camera-based held-out-site regression found in this window is Kamagata et al. (arXiv:2607.11998): V-JEPA ViT-S + SlowFast + optical flow, tested on Oak Island and Xishawan clips. It reached H_s RMSE 0.586 m with R² = 0.107, and its label instrument is not stated. Other strict-split results are cross-vessel (a ship-motion transformer: 76% → 8% accuracy on unseen hulls), cross-dataset sea-state classification (78% → 31%), and lab leave-one-train-out (PAMS-ViT). Every high headline number (96.1%, R² > 0.8, R² 0.998) comes from k-fold, random or simulation splits.

### Cited Findings
- **Kamagata et al. 2607.11998, held-out sites.**
  - Five plunge-breaker videos from geographically separate sites (Oak Island East/West, Xishawan C4), "not part of the training distribution".
  - Test RMSE 0.5863 m, PCC 0.436, R² 0.107.
  - Architecture: V-JEPA ViT-S + dual-stream SlowFast + optical-flow stream.
  - The label instrument is not stated on the page.
  - Source: [arXiv HTML](https://arxiv.org/html/2607.11998v1).
- **Yin et al. 2025:** 5-fold CV over clips pooled across 15 cameras, so sites are not held out. Mean validation accuracy 96.1% with ResNet-18+LSTM; labels are human visual — [Europe PMC full text](https://europepmc.org/article/PMC/PMC12575733).
- **Majidiyan et al. 2026:** transformer within-vessel H_s accuracy 98.72% → **76.2% on an unseen mid-range vessel → 8% on an out-of-envelope VLCC** — [SNIP](https://trid.trb.org/View/2725508).
- **Vorkapić et al.:** cross-dataset sea-state accuracy fell from 78% to 31% (ResNet-Mu-SSiD → UNIRI-SeaState-LL) — [SNIP](https://www.researchgate.net/publication/382065867_A_Computer_Vision_Approach_to_Estimate_the_Localized_Sea_State).
- **BGO-CDF 2026:** eight domain-separated protocols for Beaufort images (numbers not retrieved) — [SNIP](https://www.sciencedirect.com/science/article/abs/pii/S0029801826040461).
- **PAMS-ViT:** 12-fold leave-one-wave-train-out in a lab dataset. Strict within the lab, but at cm-scale H_s — [ABS](https://digitalcommons.mtu.edu/michigantech-p2/2959/).
- **Ni et al. 2026:** model rankings in simulation did not carry over to the towing tank; LSTM-MHA and TCN were the most robust — [ABS](https://doi.org/10.3390/jmse14131158).
- **Liu, Wei, Lu 2025:** random 1:1 split of 210 simulated samples — [FT](https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2025.1693042/full).

### Inferences
- No published camera-based work found here tests across many independent instrumented sites. The project's 79-buoy NDBC set with same-buoy labels, and its WebCOOS/CDIP pairing, appear to be a stricter protocol than anything in the 2024–2026 literature. This is worth stating explicitly when positioning results.
- Kamagata's held-out R² of 0.107 is a low bar, but its protocol (few training scenes, unstated label source) is not comparable with buoy-labelled hourly H_s.

### Gaps
- I could not determine the ground-truth source for the Oak Island and Xishawan test clips in 2607.11998. The Yin dataset itself has no instrument labels.
- I did not find any camera-based held-out-site result reported in metres against buoys outside the known list.

---

## Q3. Which encoders were released June – Nov 2026, and do any beat V-JEPA 2/2.1 on motion or DINOv3 on dense geometry with usable licences?

### Takeaway
No major-lab successor appeared between June and November 2026: no V-JEPA 3 or 2.x after 2.1, no DINOv4, no SigLIP 3, no VideoPrism 2, no InternVideo3, no Cosmos-Embed2, no C-RADIOv5. The official V-JEPA and DINOv3 repos list nothing newer than V-JEPA 2.1 (16 Mar 2026), plus DINOv3 adaptation code (FINO, 12 Jun 2026) and the CHMv2 head.

The window did produce research encoders that claim wins in narrow settings:
- **JEM** (Oct 2026): patch-level SSL to 7B; beats DINOv3 on panoptic segmentation with 12× less data.
- **VANE** (Oct 2026): next-frame embedding prediction; +7.1 points over V-JEPA 2.1 on EPIC-KITCHENS anticipation.
- **TT-VidT** (Sep 2026): DINOv3-initialised ViT-B; leads SSv2 and Diving-48 *fine-tuning* at matched ~180M scale.
- **λ-JEPA** (Sep 2026): beats V-JEPA 2 on SSv2/K400 in its own small-scale setting.

None has verified public weights plus a frozen-probe win over V-JEPA 2-g (77.3% SSv2, 90.2% Diving-48) or over DINOv3-7B on depth. V-JEPA 2/2.1 (MIT) and DINOv3 (custom licence) therefore remain the reference encoders.

### Cited Findings

**Verified against official repos/cards**
- **V-JEPA 2 / 2.1** (Meta) — [GitHub facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2):
  - README news: only 2026-03-16 (V-JEPA 2.1) and 2025-06-25 (V-JEPA 2); no later version.
  - 2.1 checkpoints at 384 px: ViT-B 80M, ViT-L 300M, ViT-g 1B, ViT-G 2B. V-JEPA 2 checkpoints: L 300M, H 600M, g 1B (256), g 1B (384); plus V-JEPA 2-AC.
  - Licence: MIT (three Apache-2.0 files).
  - Frozen attentive probe: SSv2 **77.3%** (g/384; previous best 69.7%), Diving-48 **90.2%** (previous best 86.4%); ViT-L: 73.7% and 89.0%.
- **DINOv3** (Meta) — [GitHub facebookresearch/dinov3](https://github.com/facebookresearch/dinov3):
  - News: 2026-06-12, metadata-guided training code on a `FINO` branch ("Who Needs Labels? Adapting Vision Foundation Models With the Metadata You Already Have", Gardès et al. 2026), with recipes for FMoW and HPA-WholeHR.
  - 2026-03-10: CHMv2 canopy-height head on a DINOv3 ViT-L.
  - 2025-11-20: ConvNeXt distillation code.
  - No new backbone. Licence: the custom "DINOv3 License".
- **PE-AV, Perception Encoder Audiovisual** (Meta, open-sourced 18 Dec 2025): extends Perception Encoder to joint audio-video, audio-text and video-text embeddings. Six public checkpoints from a 16-frame small model to all-frame large variants; trained on more than 100M videos; CVPR 2026 paper. Sources: [Meta on X](https://x.com/AIatMeta/status/2001698702961053750?lang=en); [MarkTechPost](https://www.marktechpost.com/2025/12/22/meta-ai-open-sourced-perception-encoder-audiovisual-pe-av-the-audiovisual-encoder-powering-sam-audio-and-large-scale-multimodal-retrieval/); [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Vyas_Pushing_the_Frontier_of_Audiovisual_Perception_with_Large-Scale_Multimodal_Correspondence_CVPR_2026_paper.pdf).
- **EUPE, Efficient Universal Perception Encoder** (Meta, arXiv:2603.22387, Mar 2026): distils PEcore-G, PElang-G and DINOv3-H+ into 6M–89M students [SNIP] — [arXiv](https://arxiv.org/pdf/2603.22387).
- **Pixio** (Meta, CVPR 2026): a dense-prediction encoder trained by pixel reconstruction; GitHub repo exists; Hugging Face weights not confirmed [SNIP] — [GitHub](https://github.com/facebookresearch/pixio).
- **C-RADIOv4** (NVIDIA): released 2026-01-27, already covered. A C-RADIOv4-1D-H card commit carries a 07/01/2026 date and an ICML 2026 tag, so this is a variant, not a new generation. No C-RADIOv5 found. Sources: [NVlabs/RADIO](https://github.com/nvlabs/radio); [HF commit](https://huggingface.co/nvidia/C-RADIOv4-1D-H/commit/b242b034a5aa19ae46d33a123a39adf6645a3ded).
- **Cosmos-Embed1** (NVIDIA): EVA-ViT-G + Q-Former + BERT text encoder; 224/336/448p; 8 frames; 256-d or 768-d embeddings. A 448p anomaly-detection fine-tune was posted around March 2026. No Cosmos-Embed2 found. Sources: [HF card](https://huggingface.co/nvidia/Cosmos-Embed1-448p); [NVIDIA TAO docs](https://docs.nvidia.com/tao/tao-toolkit/latest/text/embedding/cosmos_embed1.html); [VSS docs](https://docs.nvidia.com/vss/3.2.0/models/cosmos-embed1.html).
- **Google:** no SigLIP successor, VideoPrism v2 or TIPS update found. Gemma 4 was released 2 Apr 2026 under Apache 2.0, but its vision encoder is not identified in the sources found — [Wikipedia](https://en.wikipedia.org/wiki/Gemma_(language_model)).
- **InternVideo3 and VideoPrism 2:** no evidence that either exists — [search](https://research.google/blog/videoprism-a-foundational-visual-encoder-for-video-understanding/).

**Research encoders June – Nov 2026 (arXiv, abstracts read via the arXiv API) [ABS]**

| Model (arXiv, date) | What it is | Claimed results | Weights | Source |
|---|---|---|---|---|
| **JEM**, "Scalable Patch-Level Self-Supervised Learning" (2610.10013, 7 Oct 2026; Seitzer, Trivigno, Vobecký, Yi, Oquab, Vo) | Principled patch-level student-teacher SSL with information- and structure-preservation losses; ViT-L 300M, g 1B, 7B (about 6.7B), trained on "A140M" for 500k steps | Beats the DINOv2 algorithm on segmentation at all scales; **at 7B exceeds DINOv3 on panoptic segmentation with 12× less data and no refinement stages** | Not found | [arXiv](https://arxiv.org/abs/2610.10013); [EmergentMind summary](https://www.emergentmind.com/papers/2610.10013) |
| **VANE**, "Learning from Next-Frame Prediction" (2610.30724, 22 Oct 2026) | Causal encoder predicting the next frame's representation in a frozen semantic-tokenizer space, plus an auxiliary generative decoder; unified image and video | SOTA on most anticipation, recognition and dense benchmarks "among unified encoders"; **+7.1 points mean-class recall@5 over V-JEPA 2.1 on EPIC-KITCHENS-100 anticipation** | Not stated | [arXiv](https://arxiv.org/abs/2610.30724) |
| **TT-VidT** (2609.33419, 27 Sep 2026) | DINOv3-initialised ViT-B/16 per-frame path plus a compact Temporal Transfer Layer trained by "Diff Compression"; 24-way matched study at 170–190M on 1.7M clips for 8 epochs | Leads **Jester, SSv2, ARID and Diving48 fine-tuning** by 54–121% over the next row in its matched setting; 55% fewer encoder FLOPs than VideoMAE or V-JEPA 2 | Not stated | [arXiv](https://arxiv.org/abs/2609.33419) |
| **λ-JEPA** (2609.35288, 28 Sep 2026) | SACReg spectral anti-collapse regulariser applied to the backbone | "improves over LeVJEPA and V-JEPA 2 on SSv2 and K400". Snippet: ViT-B SSv2 gains of 13.5 points over LeVJEPA at 240 epochs and 7.9 at 1,085 epochs; CLS probe 45.7. This is small-scale, far below 77.3. | Code on GitHub | [arXiv](https://arxiv.org/abs/2609.35288); [PDF snippet](https://arxiv.org/pdf/2609.35288) |
| **VideoMSN** (2609.40347, 30 Sep 2026) | Masked Siamese training on "super-image" frame grids, starting from DINOv3 or DeiT-v3 | SOTA on K400, UCF101 and HMDB51 with up to 32× fewer pretraining epochs | Project page | [arXiv](https://arxiv.org/abs/2609.40347) |
| **EV-JEPA** (2610.14324, 9 Oct 2026; authors include M. Assran, T. Moutakanni) | Width-elastic V-JEPA that can be split across devices | Within 1 point of separately trained baselines | Not stated | [arXiv](https://arxiv.org/abs/2610.14324) |
| **Fovea-JEPA** (2610.34033, 25 Oct 2026) | Finds V-JEPA's "geometric blind spots" for **camera-motion** recognition and fuses VGGT geometry features | +9.67 / +8.67 / +24.38 points on CameraMotionDataset / CameraBench / ShotBench | "upon acceptance" | [arXiv](https://arxiv.org/abs/2610.34033) |
| **SDM**, Structured Dynamics Model (2607.21576, 23 Jul 2026; Knobel, Zisserman, Asano) | Separates **camera motion from object motion** on frozen image-ViT features via future-feature prediction; new ProbeMotion suite | Beats CLS and mean-pooled baselines; "compares favorably" to VGGT | Project page | [arXiv](https://arxiv.org/abs/2607.21576) |
| **ALPS** (2610.30120, 21 Oct 2026) | ViT-B "visual state encoder" with language, reconstruction and correspondence objectives | Matches V-JEPA 2-L (3.5× larger) on LIBERO state-probe error | Not stated | [arXiv](https://arxiv.org/abs/2610.30120) |
| **Omni-Encoder** (2611.05406, Oct 2026) | One shared SigLIP-2 ViT for image, video, audio and audio-visual input | Competitive on 16 benchmarks | "will be released" | [arXiv](https://arxiv.org/abs/2611.05406) |
| **CoVisco** (2609.39924, Sep 2026; Baidu ERNIE) | Codec-native, token-compressing image/video encoder; 565M image-text pairs plus 6.4M videos | Near or above OneVision-Encoder with 400 tokens per 64 frames | GitHub | [arXiv](https://arxiv.org/abs/2609.39924) |

Supporting evaluations from the window:
- **Latent-prediction robustness study** (2605.15618, May 2026). Compares V-JEPA 2.1, V-JEPA 2, VideoPrism and VideoMAEv2 at matched capacity on SSv2 and EGTEA. The latent-prediction models "degrade more gracefully under pixel corruption", "uniquely encode the arrow of time", and alone give near-complete planning success — [arXiv](https://arxiv.org/abs/2605.15618).
- **V-JEPA-LM** (2611.17831, 13 Nov 2026). A linear probe reads **speed from frozen V-JEPA features at 87%**. The dynamics sit in a subspace shared across videos for predictive encoders, but not for frame-wise text-aligned encoders such as SigLIP — [arXiv](https://arxiv.org/abs/2611.17831).
- **Controlled image-vs-video SSL study** (2608.13183, Aug 2026). DINOv2-style pretraining is strongest under limited resources. Adding VideoMAE objectives helps classification and segmentation but **degrades tracking and camera-pose estimation** — [arXiv](https://arxiv.org/abs/2608.13183).
- **ViTok** (2610.02903). An AM-RADIO-style student distilled from SigLIP2 + DINOv3-L merely matches the DINOv3-L teacher on ADE20K (48.5 mIoU) — [arXiv](https://arxiv.org/abs/2610.02903).

### Inferences
- For a motion-centred task (wave periodicity, horizon tilt), the V-JEPA 2/2.1 family is still the best-documented frozen video encoder with a permissive MIT licence. Nothing released June–Nov 2026 has public weights *and* a frozen-probe SSv2 or Diving-48 result above 77.3 / 90.2.
- JEM and VANE are the two to watch:
  - JEM is a dense-geometry contender to DINOv3. The author overlap with the DINO papers (Oquab, Vo) suggests Meta FAIR, but affiliation was not verified.
  - VANE is a motion and anticipation contender to V-JEPA 2.1.
  - Check both for weight releases before planning any runs.
- TT-VidT and VideoMSN show a cheap recipe: a DINOv3-initialised image ViT plus a light temporal module or video SSL. Similar ideas (SDM, Fovea-JEPA) explicitly separate camera motion from scene motion, which maps onto ship or buoy camera ego-motion versus wave motion. These are architecture ideas rather than drop-in checkpoints.
- Fovea-JEPA's "geometric blind spots" and 2608.13183's tracking and camera-pose trade-off are caution flags. V-JEPA-style features may under-represent camera geometry, so the horizon-tilt motion signal may be better kept as an explicit input than inferred from V-JEPA features.

### Gaps
- Weight availability, licences and absolute frozen-probe numbers (SSv2, Diving-48, NYUv2 depth, ADE20K) for JEM, VANE, TT-VidT and λ-JEPA were not verified. Only abstracts were read, and λ-JEPA's numbers come from a snippet.
- No official confirmation either way was found for Qwen3-VL or InternVL 3.5 vision-encoder updates in the window, Apple encoder releases, Genie-family weights, or Cosmos Predict updates.
- Licence terms were not checked for PE-AV, C-RADIOv4-1D-H or Cosmos-Embed1.

---

## Q4. Are video diffusion/world-model features shown to encode physical dynamics better than V-JEPA-style encoders?

### Takeaway
No, the evidence is mixed. The most systematic frozen-probe comparison (Punzo et al., June–Oct 2026) finds V-JEPA best on MVP (95.0% pair consistency), VideoMAE best on IntPhys 2 (73.9%), and the diffusion model LTX-Video consistently behind both. One paper ("The Invisible Hand of Physics", June–Oct 2026) reports diffusion-transformer probes at about 81.27% on IntPhys/InfLevel, beating V-JEPA and VideoMAE. Robustness, speed-decoding and relational-physics studies all favour latent-prediction (V-JEPA) features. No study probes fluid, free-surface or wave dynamics. The closest evidence is SciVid weather forecasting, where 4DS-e and VideoMAEv2-g were the best frozen backbones and DINOv2 lagged far behind.

### Cited Findings
- **Punzo et al., "How Do Video Foundation Models Encode Intuitive Physics?"** (arXiv:2606.09646; v1 8 Jun 2026, v3 30 Oct 2026; NeurIPS 2026 World Models workshop) — [arXiv](https://arxiv.org/abs/2606.09646)
  - Frozen linear, MLP and temporal-attentive probes on IntPhys 2 and MVP.
  - **V-JEPA reaches 95.0% pair consistency on MVP; VideoMAE reaches 73.9% on IntPhys 2.** LTX-Video's attentive probes "consistently underperform".
  - Physics information is most accessible at intermediate-to-late layers.
  - The diffusion signal peaks at intermediate denoising depth, and frame shuffling hurts.
- **"The Invisible Hand of Physics: When Video Diffusion Models Know More Than They Show"** (arXiv:2606.05328; v1 3 Jun 2026, v2 9 Oct 2026) — [arXiv](https://arxiv.org/abs/2606.05328)
  - Physical plausibility is linearly decodable from diffusion-transformer states, at **about 81.27% average on IntPhys and InfLevel**, outperforming "V-JEPA and VideoMAE".
  - The abstract names neither the diffusion models nor the baseline numbers, and it does not cover quantitative variables such as velocity.
- **"Seeking Physics in Diffusion Noise"** (arXiv:2603.14294, Mar 2026) notes concurrent work showing plausibility is linearly decodable from intermediate diffusion features — [arXiv](https://arxiv.org/pdf/2603.14294).
- **"Interpreting Physics in Video Encoders using Multi-Object Interaction Videos"** (arXiv:2610.31983, 22 Oct 2026) — [arXiv](https://arxiv.org/abs/2610.31983)
  - Both V-JEPA 2 and VideoMAE separate elastic from inelastic collisions.
  - A kinetic-energy readout survives an environment change for V-JEPA 2 but not for VideoMAE.
  - Neither holds a friction coefficient across a change of geometry.
- **Localized VoE evaluation** (arXiv:2611.18979, 13 Nov 2026): global and localized violation-of-expectation judgments for V-JEPA models can diverge in both directions, so aggregate VoE scores can mislead — [arXiv](https://arxiv.org/abs/2611.18979).
- **Latent-prediction robustness study** (2605.15618): V-JEPA 2/2.1 beat VideoPrism and VideoMAEv2 on corruption robustness and the arrow of time, and alone support near-complete planning success — [arXiv](https://arxiv.org/abs/2605.15618).
- **V-JEPA-LM** (2611.17831): a frozen V-JEPA linear probe decodes speed at 87%, while SigLIP-style frame encoders lack a shared dynamics subspace — [arXiv](https://arxiv.org/abs/2611.17831).
- **Phantom** (arXiv:2604.08503) uses V-JEPA 2's embedding space as the physical-dynamics signal to guide a video diffusion generator. The generator borrows physics from V-JEPA rather than the reverse — [arXiv](https://arxiv.org/html/2604.08503v1).
- **GenDSR** (arXiv:2610.41314, Oct 2026) injects video-generation-model features into a VLM and reaches 67.2% on DSR-Bench (new SOTA). Generator features help spatial-dynamics reasoning when fused — [arXiv](https://arxiv.org/abs/2610.41314).
- **Fluid-Gen-Zero** (arXiv:2610.10984, Oct 2026) hands fluid dynamics to a physics simulator to guide CogVideoX- and Wan-based generators. This cuts fluid flow EPE by 67.9–84.0% and object trajectory error by 26.7–81.5% — [arXiv](https://arxiv.org/abs/2610.10984).
- **SciVid** (ICCV 2025; arXiv:2507.03578) — [arXiv HTML](https://arxiv.org/html/2507.03578v1)
  - On WeatherBench 2, **4DS-e and VideoMAEv2-g were the best frozen backbones** (Z500 wRMSE about 594–601); DINOv2 "lagged far behind".
  - V-JEPA-H frozen reached 92.4 mAP on CalMS21; 4DS-L was best on Digital Typhoon (test RMSE 5.23).
  - Frozen backbones were SOTA on 3 of 5 tasks.
  - LoRA rank 32 came close to full fine-tuning while training about 2.3% of parameters.

### Inferences
- For a wave-dynamics probe, current evidence favours latent-prediction or masked video encoders (V-JEPA 2/2.1, VideoMAEv2, 4DS) over diffusion-generator features as frozen representations. Diffusion features may hold plausibility signal, but it is less accessible and depends on layer and noise level. Fluid-Gen-Zero implies generators do not natively get fluid motion right.
- SciVid's weather result is the closest analogue to continuous geophysical fields. There, video-native masked or 4D encoders beat an image encoder by a wide margin. This supports keeping a video encoder in the project's comparison, alongside DINOv3 stills.

### Gaps
- No study found probes ocean-wave, free-surface or turbulence dynamics from any foundation encoder or video generator.
- No Physics-IQ results comparing frozen encoders with diffusion features were found in the window.
- The diffusion models and V-JEPA versions used in 2606.05328 were not identified from the abstract. Its 81.27% is not directly comparable to Punzo et al.'s 95.0% and 73.9%, because the benchmarks and metrics differ.
