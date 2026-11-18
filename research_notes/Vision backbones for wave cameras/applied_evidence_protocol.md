# Vision backbones for wave cameras: applied evidence and a fair, cheap first comparison

Compiled 2026-11-18. How well each source was checked: **full text** means the paper itself was read, in this
session or in the project's own verified paper notes (`docs/literature/papers/*.md`). **Abstract**,
**README** or **snippet** means only that level of source was available, so treat those numbers as
provisional. Project notes are cited by path next to the publisher URL.

## 1. Sea-state and wave estimation from images and video: which backbones were used, and what was learned about architecture versus data

### Takeaway
In every wave or sea-state camera study found, swapping between modern ImageNet-pretrained networks of
reasonable size changed results by only 0.1 to 2 points: VGG16 84.05 % vs ResNet 83.89 % (Choi), ResNet-101
97.4 % vs ViT-B/32 95.5 % (Vorkapić), five CNNs at 0.873 to 0.882 (Stringari). Other choices moved results by
7 to 60 points: whether the network was pretrained, whether it saw several frames, which part of the image
it was given, and above all how the data were split (88 % falls to 28 % on another ship's camera). No
published wave study compares backbones at unseen sites with instrument labels. None compares a modern
self-supervised foundation model with an ImageNet CNN under one protocol.

### Cited Findings
**Choi et al. 2020, *Ocean Eng.* 201 (NDBC buoy cameras; full text, project notes)**
- The study compared VGG, Inception v3, ResNet and DenseNet, each with ImageNet transfer and a global-feature head (GMP, GAP or a learned conv extractor). Each sample was one camera view: the lower half of the image, resized to 224². Data came from 20 buoys between 2018-03-06 and 2018-05-10. The task was binary: calm (H_s < 1.52 m) vs rough (> 3.05 m) — [Choi et al. 2020](https://doi.org/10.1016/j.oceaneng.2020.107129); `docs/literature/papers/choi2020.md`
- Over 10 replicates, VGG16 with transfer learning and the global-feature head scored 84.05 % ± 0.14 (balanced 83.99 %). ResNet scored 83.89 %. Without ImageNet weights accuracy was about 77 %. Training and testing on a single camera position gave 80.6 to 84.2 % — [Choi et al. 2020](https://doi.org/10.1016/j.oceaneng.2020.107129); `choi2020.md`
- The split was random (35,696 train / 8,924 test), with the same buoys and days in both sets. Labels are WVHT reported about 30 min before the image. Images with mean RGB below 30 or above 190 were removed — same sources
- Ship regression: frame differences went into a pretrained VGG, then a bidirectional ConvLSTM. On 344 test clips the error was RMSE 0.133 m and MAE 0.089 m. The split is not described — same sources

**Kim et al. 2023, *Ocean Eng.* (fixed camera at Jeju, AWAC labels; full text, project notes)**
- The study compared a small snapshot CNN (4 blocks, 16-32-64-64) with GoogLeNet features fed to a BiLSTM (2,000 hidden units). The target was five classes of *average* wave height — [Kim et al. 2023](https://doi.org/10.1016/j.oceaneng.2023.116002); `docs/literature/papers/kim2023.md`
- Accuracy rose from 0.783 (snapshot) to 0.904 (180-s clip, CNN-BiLSTM) to 0.932 (clip with screen-splitting augmentation) — same
- The data were 30 days with one hour of video per day. Snapshots and clips were drawn at random, so frames from the same hour, with the same label, appear in both training and test — same

**Vorkapić, Pobar & Ivašić-Kos 2024, *Ocean Eng.* 309:118318 (ship bridge camera; accepted manuscript read in full this session)**
- One camera on the bridge of one ocean-going vessel recorded 44 sessions: 174,439 frames, about 48.5 min of video in total, covering Beaufort 1 to 8. Class 8 comes from 3 sessions lasting 1 min 47 s in total (Table 1). Labels are expert (officer) Beaufort estimates — [arXiv:2407.03755v2](https://arxiv.org/abs/2407.03755)
- About 750 training images per class, plus about 300 validation and 300 test images per class, were sampled at intervals from the same footage. The text does not state that whole sessions were held out. A pilot run found accuracy stopped improving beyond about 300 training images per class — [arXiv:2407.03755v2](https://arxiv.org/abs/2407.03755)
- **Recipe:** ImageNet weights, trained in two stages.
  - Stage 1 trained only the head for 30 epochs (Adam, 1e-4).
  - Stage 2 unfroze part of each network: ResNet-101 24.8 M of 42.7 M parameters, ViT-B/32 21.3 M of 87.5 M, MobileNetV2 0.7 M of 2.7 M, NASNet-Mobile 1.6 M of 4.3 M. It used RMSProp for 200 to 1,030 epochs, "chosen according to the point where there was no further improvement in the training accuracy", with learning-rate reduction on a validation-accuracy plateau.
  - One model was trained per architecture and dataset, so there are no seed replicates — same
- Accuracy with random-position crops: ResNet-101 0.881, ViT-B/32 0.868, NASNet 0.781, MobileNetV2 0.502. With a fixed lower-left crop: 0.974, 0.955, 0.902 and 0.844 (Tables 4 and 5) — same
- **Across datasets:**
  - A ResNet trained on MU-SSiD scored 0.78 on MU-SSiD and 0.31 on UNIRI-LL.
  - A ResNet trained on UNIRI scored 0.88 on UNIRI and 0.28 on MU-SSiD (Tables 6 and 7).
  - The paper is internally inconsistent here: the in-domain 0.88 matches Table 4 (random crops), not the 0.974 of Table 5 (fixed crop) — same

**Pobar, Paulin, Ivašić-Kos & Vorkapić 2026, *IEEE OJ-ITS* 7:136–154 (doi 10.1109/ojits.2025.3650057; abstract only)**
- Bridge camera, Beaufort 1 to 8, a larger and more varied test set. The paper reports 96 % within one class — [TRID record](https://trid.trb.org/View/2685736); [DOAJ record](https://doaj.org/article/c22ccffde2d443a3958da43e580e42df)
- The project's screening note adds that real and synthetic images were mixed, predictions were voted over consecutive frames, and mean F1 was 62 % — `docs/literature/landscape_review.md` (screened only)

**Kamagata et al. 2026, arXiv:2607.11998 (shore video; full text, project notes)**
- Frame embeddings (384-d) from a pretrained "V-JEPA ViT-S", given in the paper as `vit_small_patch16_224`, are cropped by optical-flow saliency and passed to a 2-layer SlowFast transformer head with a physics-informed loss. Training used 6 scenes; testing used 5 held-out videos — [arXiv:2607.11998](https://arxiv.org/abs/2607.11998); `docs/literature/papers/kamagata2026.md`
- Held-out H_s: RMSE 0.586 m, R² 0.107, Pearson 0.436, regression slope 0.12. Training used a single seed (42), with early stopping at epoch 31 of 50. The source of the labels is not stated — same
- A companion ViViT study estimated peak period with RMSE 0.80 s on 4 held-out scenes, using expert and pseudo labels — [arXiv:2606.13302](https://arxiv.org/abs/2606.13302); `docs/literature/benchmark_targets.md`

**Xu et al. 2024, *JMSE* 12:2003 (Qingdao shore camera)**
- An improved ResNet (ResNet-SW) classifies wave type. Inversion-Net combines the image with meteorological and oceanographic factors, with separate wind-sea and swell models. Final MAE 0.04 m, MAPE 8.52 % — [Xu et al. 2024](https://www.mdpi.com/2077-1312/12/11/2003) (abstract; the publisher page returned 403 this session)
- The project's earlier read of the paper:
  - one camera at 1 frame/min, labelled by a nearby Waverider MKIII, hourly values interpolated to each minute;
  - a **random 6:2:2 split of frames**;
  - image-only accuracy 57.7 %; wind-sea vs swell classification 94 % — `docs/literature/benchmark_targets.md`

**Stringari et al. 2021, *Sci. Rep.* 11:3604 (active wave breaking)**
- Five backbones were implemented: VGG16, ResNet50V2, InceptionResNetV2, MobileNetV2 and EfficientNetB5. A search snippet of the paper says that architectures other than VGG16 gave "identical results" — [Stringari et al. 2021](https://www.nature.com/articles/s41598-021-83188-y)
- **Repository test results (1,200 samples):**

  | Backbone | Accuracy | AUC |
  |---|---|---|
  | VGG16 | 0.876 | 0.855 |
  | ResNet50V2 | 0.881 | 0.843 |
  | InceptionResNetV2 | 0.882 | 0.839 |
  | EfficientNet | 0.873 | 0.827 |
  | MobileNet | 0.875 | 0.768 |

  MobileNet's recall was only 0.171. All models were trained **from scratch** ("weights ... reset"), on 20k crops with a random 80/20 train/validation split — [deepwaves README](https://github.com/caiostringari/deepwaves)
- **Conflict:** a search-engine summary gave ResNetV2 accuracy 0.97 and EfficientNet 0.772. The README does not support those values, so the README is used here — [deepwaves README](https://github.com/caiostringari/deepwaves)

**Buscombe et al. 2020, *Coast. Eng.* 155:103593 (optical wave gauging)**
- The code trains every combination of MobileNetV1, DenseNet201, InceptionV3 and InceptionResNetV2 with 4 batch sizes. DenseNet201 replaces the MobileNetV2 used in the paper, which fails under TensorFlow 2. Pretrained MobileNetV1 models for H and T are provided. The README reports no score for any individual backbone — [OWG repository](https://github.com/dbuscombe-usgs/OpticalWaveGauging_DNN)
- Reported errors: RMSE 0.14 m for infrared at Duck and 0.11 m for oblique RGB at Santa Cruz. Each test is at a single site, and the split is not verified — [Buscombe et al. 2020](https://www.sciencedirect.com/science/article/abs/pii/S0378383919301243); `docs/literature/landscape_review.md`

### Inferences
- The backbone matters less than the data in this literature. Among networks of 25 M parameters or more, backbone differences are 0.1 to 2 points. Pretraining is worth about 7 points (Choi), temporal context about 12 (Kim), crop placement about 9 (Vorkapić R vs LL) and the camera domain about 60 (Vorkapić across datasets). The project's own seed-to-seed spread, if a few points, is already as large as published backbone effects. A backbone comparison therefore needs seed replicates and confidence intervals, or it will measure noise.
- Small "mobile" networks are the exception. They lost 7 to 38 points in Vorkapić (fine-tuned with only 0.7 M trainable parameters) and had recall 0.17 in Stringari. Do not include them as candidates for minority (rough-sea) classes.
- Every number above, except Kamagata's, comes from random frame or clip splits at one site or ship, so none of it says which backbone generalises across sites. The one held-out-site video result (Kamagata, R² 0.11, slope 0.12) is weak and has labels that cannot be traced to an instrument.
- Kamagata's "V-JEPA ViT-S" needs checking before it is treated as a foundation-model data point. `vit_small_patch16_224` is the timm name of an ImageNet-supervised ViT-S, and the original V-JEPA release, as I remember it, provided ViT-L/16 and ViT-H/16. This comes from my training knowledge and was not re-checked this session.
- Vorkapić chose epochs by training accuracy, and its validation set came from the same footage as the test set. Its numbers are optimistic about generalisation, though not a case of selecting on the test set.

### Gaps
- Pobar 2026 full text was not accessible (DOAJ 403). The backbone list in the request (DeiT III, Swin, XciT, CoAtNet), the per-backbone numbers, and whether test voyages were separate could not be confirmed.
- Xu 2024 could not be re-read this session (MDPI 403). Its details rely on the project's earlier read.
- Buscombe 2020's per-backbone results were not found: the README has none, and the paper is paywalled.
- No wave or sea-state paper compares frozen DINOv2, CLIP or SigLIP features with a fine-tuned ImageNet CNN. That experiment would be new.

## 2. Analogous geophysical camera tasks (river stage, flood, snow, visibility, smoke, rip currents): which backbones won, frozen or fine-tuned, and generalisation to unseen cameras

### Takeaway
The applied environmental camera literature mostly uses ImageNet-pretrained ResNet-50-class CNNs, fine-tuned. The strongest regression evidence at unseen cameras (Vandaele 2023: 86 training cameras, 4 unseen test cameras) uses a fine-tuned ImageNet ResNet-50 with labels standardised per site. Where foundation models appear (Hu 2026, river level), they help by extracting a physically meaningful intermediate, a water mask and an index computed from it, not by being a better end-to-end regressor. Authors repeatedly report that camera or site effects outweigh network effects. **I found no applied geophysical camera study that compares frozen DINOv2 or CLIP probes with fine-tuned CNNs at unseen sites and reports numbers.**

### Cited Findings
**River stage: Vandaele, Dance & Ojha 2021, *HESS* 25:4435 (full text)**
- **Setup:**
  - Networks: DeepLabv3 with a ResNet-50 backbone (pretrained on COCO-stuff) and UperNet with ResNet-50 (pretrained on ADE20k).
  - Fine-tuned on LAGO (300 images) or WATERDB (12,684 images), either all layers at once (WHOLE) or head then whole network (2STEPS).
  - Applied to 4 river cameras on the Severn and Avon — [Vandaele et al. 2021](https://hess.copernicus.org/articles/25/4435/2021/)
- Water segmentation reached balanced accuracy 0.91 to 0.97 at the four cameras (DeepLab-LAGO). The derived water index correlated with gauges at Pearson r > 0.94 (abstract claim). At one site (TEWK) some configurations gave correlations near zero or negative — same
- The authors found "no clear dominance of a particular CNN, fine-tuning dataset or methodology". Camera location, occlusion by boats and lock operations affected results more than the choice of network — same

**River stage at unseen cameras: Vandaele, Dance & Ojha 2023, *Environmental Data Science* 2:e11 (article read this session)**
- ResNet-50 initialised from ImageNet and fine-tuned for 30 epochs, with an L1 loss for regression and BCE for flood classification. Data: 327,215 images from 95 cameras; the abstract's "32,715" is a typo, since the splits sum to 293,214 + 18,894 + 15,107 — [Vandaele et al. 2023](https://www.cambridge.org/core/journals/environmental-data-science/article/calibrated-riverlevel-estimation-from-river-cameras-using-convolutional-neural-networks/912B6F25D567B59C38C6D76185829F76)
- **Split by camera:** 86 training, 5 validation and **4 test cameras unseen in training**, whose gauges serve no training camera. Labels are z-scores per gauge, and the flood class is above each gauge's 95th percentile — same
- At the unseen cameras, the estimates are about as accurate as using a distant gauge within 50 km. The MAE and balanced-accuracy values are given only in figures — same

**River stage with a foundation model: Hu, Wang, Fan & Zhang 2026, *Water Resources Research* (abstract and secondary digests only)**
- A one-shot foundation segmentation model extracts water masks, which are turned into a Static Observer Flooding Index (SOFI). A lightweight regressor then maps SOFI to water level. The study compares this with a VGG16 end-to-end regression baseline and reports better generalisation when few samples per site are available — [Hu et al. 2026](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025WR042686) (full text 403)
- A secondary digest claims the method roughly halves high-water-level error compared with end-to-end regressors. This was not verified against the paper — [TerraMosaic digest 2026-05-12](https://www.terramosaic.org/news/terramosaic-daily-digest-2026-05-12/)

**Visibility and snow depth from webcams (snippets only)**
- A model trained with multi-task labels for snow depth and visibility on about 20,000 images from 200 AMOS webcams worked across many camera settings and lighting conditions. The authors note that earlier work used one or a few cameras — [EGU2020-19802](https://meetingorganizer.copernicus.org/EGU2020/EGU2020-19802.html)
- One visibility study argues that a model trained on a single camera does not generalise to new locations or fields of view. Comparing each query image with calibrated reference images from the same camera did generalise to new NYSM/ASOS sites, provided a few references per site were available — [Visibility estimation from camera images](https://shareok.org/items/bfebee6b-5002-40d7-b93d-4d6892e5292d)
- For MeteoSwiss webcams, a CNN produced a per-pixel visibility map, and adding depth maps helped it tell snow from cloud — [Rożniak 2020, ETH MSc thesis](https://ethz.ch/content/dam/ethz/special-interest/baug/igp/photogrammetry-remote-sensing-dam/documents/pdf/Student_Theses/2020/MA_Rozniak.pdf)
- CNN regression of snow depth from highway monitoring cameras was validated at two sites — [JSCE paper](https://www.jstage.jst.go.jp/article/jsceiii/2/1/2_26/_pdf/-char/en)

**Wildfire smoke (snippets only)**
- SmokeyNet combines a CNN on image tiles plus the previous frame, an LSTM, and a ViT across tiles. It was trained on FIgLib, about 25,000 labelled images from fixed cameras in Southern California, and its authors say it rivals human performance. No results at held-out cameras were found — [FIgLib & SmokeyNet](https://arxiv.org/abs/2112.08598)
- A multimodal follow-up adds weather-sensor data, splits by fire (131 / 63 / 61 fires) and reports 13.6 % faster detection — [Multimodal smoke detection](https://ar5iv.labs.arxiv.org/html/2212.14143); [SDSC news](https://sdsc.edu/News%20Items/PR20230809_smokeynet.html)

**Rip currents (snippets only)**
- de Silva et al. 2021 used Faster R-CNN with temporal aggregation, reported above human accuracy — [arXiv:2102.02902](https://arxiv.org/abs/2102.02902)
- The 2026 RipAID benchmark uses YOLOv11n-OBB. A reviewer notes that the dataset over-samples low-energy Mediterranean beaches. No protocol that trains on some beaches and tests on unseen ones was found — [egusphere-2026-1138](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-1138/egusphere-2026-1138.pdf)

**Frozen foundation features for continuous physical quantities**
- When frozen encoders were probed for continuous physical measurement on FreiHAND, DINOv2 scored test R² 0.523 against 0.551 for CLIP. The ranking depends on the task — [arXiv:2603.06459](https://arxiv.org/pdf/2603.06459) (snippet)
- **VTAB-1K, ViT-B, 1,000 training samples per task** (accuracy %, linear probe / full fine-tuning / LoRA) — [Mai et al., arXiv:2409.16434](https://arxiv.org/abs/2409.16434), Table 7:

  | Task | Backbone | Linear | Full FT | LoRA |
  |---|---|---|---|---|
  | EuroSAT (satellite) | DINOv2 | 89.2 | 66.9 | 94.2 |
  | EuroSAT (satellite) | IN-21k | 90.0 | 88.1 | 94.9 |
  | Resisc45 (satellite) | DINOv2 | 78.8 | 25.9 | 84.2 |
  | Resisc45 (satellite) | IN-21k | 74.9 | 81.6 | 85.9 |
  | KITTI distance | DINOv2 | 60.3 | 47.1 | 83.8 |
  | KITTI distance | IN-21k | 64.6 | 77.9 | 79.9 |
  | Clevr counting | DINOv2 | 47.5 | 27.3 | 89.8 |
  | Clevr counting | IN-21k | 37.5 | 56.2 | 82.9 |

- An image-regression study applying LoRA to DINOv3 for pasture biomass exists (*Sensors* 2026) but was not read — [doi:10.3390/s26165285](https://doi.org/10.3390/s26165285)

### Inferences
- Vandaele 2023 is the closest design analogue to our buoy test: many sites, regression, the test set split by site, and an ImageNet ResNet-50 fine-tuned end to end. It reached gauge-like usefulness without a foundation model, which suggests that the number of distinct sites matters more than the backbone.
- Cross-site failures in these fields trace to properties of each camera: height, field of view, framing and occlusion (Vorkapić, the visibility work, Vandaele 2021). The remedies that worked change what the model is given, not the backbone: per-site standardised labels, references from the same camera, a physically meaningful intermediate such as a water mask. For buoys, the analogue would be a buoy-specific input, such as a calm-sea reference strip or camera height and metadata. That is worth testing alongside any backbone swap.
- On VTAB-1K's "structured" tasks (distance, counting), which are most like estimating a physical quantity, **a linear probe on frozen DINOv2 is weak (47 to 60 %) while LoRA recovers 84 to 90 %**. If wave height depends on fine texture and scale cues, a linear probe alone may undersell foundation models. Include an attentive or MLP probe and an efficient fine-tuning arm before concluding that a foundation model does not help.

### Gaps
- None of the environmental camera papers found reports foundation models against CNNs at unseen sites with numbers. Hu 2026 is closest, but its full text and numbers were not accessible.
- Cloud cover (all-sky imagers), PhenoCam and crop monitoring, flood depth from photos, and surf-zone segmentation were not searched in this pass because of the tool budget.
- The visibility, snow, smoke and rip-current entries rest on abstracts and snippets. Their split designs and error metrics were not verified.

## 3. Protocol: comparing backbones fairly on a small, domain-specific dataset (probes vs fine-tuning, seeds, splits held out by site, bootstrap, compute, pitfalls)

### Takeaway
A fair and cheap first comparison has six parts:

1. One sealed test set split by site (the 12 buoys), plus a separate set of validation buoys taken from the training buoys for every selection decision.
2. Frozen-feature probes for several backbones, each with a linear and an attentive or MLP probe, all trained on features cached once.
3. Fine-tuning only the top 2 or 3 backbones, with LP-FT or layer-wise LR decay or LoRA, and with each backbone given its own learning-rate sweep.
4. At least 5 seeds per fine-tuned arm, varying initialisation, data order and augmentation together.
5. Paired differences with a bootstrap over buoys (clusters).
6. No epochs or hyperparameters chosen on the test buoys.

The probe sweep costs roughly two orders of magnitude less than a fine-tuning sweep.

### Cited Findings
**Transfer and probing methodology**
- Kornblith, Shlens & Le (CVPR 2019) tested 16 networks on 12 datasets. ImageNet accuracy predicted transfer accuracy at r = 0.99 with fixed features and r = 0.96 with fine-tuning. Regularisers that raise ImageNet accuracy produced penultimate features that transfer much worse when frozen. On two small fine-grained datasets, ImageNet pretraining gave minimal benefit — [arXiv:1805.08974](https://arxiv.org/abs/1805.08974)
- Kumar et al. (ICLR 2022 oral) tested 10 distribution-shift datasets, including FMoW, a satellite set with geographic shift. Compared with linear probing, fine-tuning was about 2 % better in distribution and about 7 % worse out of distribution. LP-FT (train a linear probe, then fine-tune from it) was about 1 % better in distribution and about 10 % better out of distribution than full fine-tuning — [arXiv:2202.10054](https://arxiv.org/abs/2202.10054)
- V-JEPA (Bardes et al. 2024) made attentive probing its default frozen protocol. Re-evaluated with an attentive probe, DINOv2 ViT-g/14 on Kinetics-400 rose from 78.4 % (linear, as reported by Oquab et al.) to 83.4 % — [arXiv:2404.08471](https://arxiv.org/abs/2404.08471)
- V-JEPA 2 trains a 4-layer attentive probe on frozen encoders and applies the same protocol to every baseline, DINOv2 included. The authors note that the baselines' pretraining data differ — [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)
- Mai et al. (2024, v5 March 2025) found that, once hyperparameters are tuned carefully, different parameter-efficient methods reach similar accuracy but make different mistakes. For low-shot training they recommend a **non-zero drop-path rate**, because "low-shot training is especially prone to over-fitting" — [arXiv:2409.16434](https://arxiv.org/abs/2409.16434)

**Variance, seeds and statistics**
- Bouthillier et al. (MLSys 2021) found that data sampling, parameter initialisation and hyperparameter choice "impact markedly the results" across 5 tasks. Randomising more of these sources together gave an estimator close to the ideal one at 51 times lower compute cost — [arXiv:2103.03098](https://arxiv.org/abs/2103.03098)
- Agarwal et al. 2021 recommend interval estimates (stratified bootstrap confidence intervals) and robust aggregates such as the interquartile mean when only a few runs are available. This paper was listed in search results; the details come from my prior knowledge, not a fetch this session — [arXiv:2108.13264](https://arxiv.org/abs/2108.13264)
- Picard 2021 studies how much computer-vision results vary with the random seed. Listed in search results, not read — [arXiv:2109.08203](https://arxiv.org/abs/2109.08203)

**Splits that respect site and time**
- Roberts et al. 2017 (*Ecography* 40:913–929) recommend block cross-validation (by space, hierarchical group, phylogeny or predictor space) whenever the data are dependent, *even when no correlation is visible in the model residuals* — [Roberts et al. 2017](https://nsojournals.onlinelibrary.wiley.com/doi/full/10.1111/ecog.02881)

**Pitfalls seen in the wave literature (Section 1 sources)**
- **Random frame or clip splits within one site:**
  - Kim 2023: frames from the same hour in training and test.
  - Xu 2024: frames 1 min apart in a random 6:2:2 split.
  - Choi 2020: the same buoys and days in both sets.
  - Vorkapić 2024: training and test frames from the same 44 sessions.
  - Across datasets, Vorkapić's accuracy falls from 0.88 to 0.28 — sources in Section 1
- **Regression dilution:** Kamagata's held-out slope is 0.12, meaning predictions collapse toward the mean — [arXiv:2607.11998](https://arxiv.org/abs/2607.11998); `kamagata2026.md`
- **Label noise and mismatch:**
  - Choi labels each image with a wave record from about 30 min earlier.
  - Kim's target is H_ave, not H_s (H_ave / H_s ≈ 0.63).
  - Vorkapić and Pobar use observer Beaufort estimates.
  - Kamagata's labels cannot be traced to an instrument — project notes `choi2020.md`, `kim2023.md`, `kamagata2026.md`; [arXiv:2407.03755](https://arxiv.org/abs/2407.03755)
- **Epoch selection:** Vorkapić stopped on training accuracy, and its validation set came from the same footage as the test set. Kamagata trained one seed with early stopping — same

### Inferences
**Compute.** This is my estimate; the FLOP figures are standard numbers from the ConvNeXt paper ([arXiv:2201.03545](https://arxiv.org/abs/2201.03545)), not re-fetched this session.
- One forward pass at 224² costs about 4.5 GFLOPs for ConvNeXt-T, about 17.6 for ViT-B/16 and about 62 for ViT-L/16.
- Caching features for about 41k strips × 6 views (≈ 246k images) is a single forward pass per backbone. For ViT-B that is about 4 PFLOPs, minutes on one A100 or H200 and well under an hour on a smaller GPU. Probes on cached features then train in seconds to minutes and need no GPU.
- Fine-tuning costs about 3 times a forward pass per image per epoch. So 30 epochs × 5 seeds cost about 450 times the feature cache, per backbone.
- Feature extraction is therefore the only GPU-heavy step of the probe stage, and it can run once on the HPC. That fits the current rule against GPU training on the desktop.

**Proposed first comparison (buoy-camera track):**
1. **Splits.** Keep the 12 test buoys sealed until the end. From the 67 training buoys, set aside about 10 to 12 validation buoys, spread across regions. Every choice is made on these: probe type, layer, learning rate, epochs, LoRA rank. Because probes are cheap, also run grouped cross-validation by buoy over the 67 (for example 5 folds), so that rankings rest on more than 10 sites. Report a time-blocked check as well, such as the most recent months held out, because buoys tested in the same months as training share weather regimes.
2. **Backbones for the first pass.**
   - ConvNeXt-T IN-1k, the current model, run frozen and fine-tuned under the new protocol so the baseline is measured the same way.
   - ConvNeXt-T IN-22k: the same architecture with more pretraining data.
   - DINOv2 ViT-S/14 and ViT-B/14 (the register variants).
   - CLIP or SigLIP ViT-B/16.
   - Optionally DINOv3 ViT-B, if its licence allows.
   - Keep V-JEPA 2 for the video track.
   - Leave out mobile-class networks (Section 1).
3. **Probes on cached features.**
   - (i) Logistic regression or ridge on the pooled token per view (CLS concatenated with the mean patch token), joined across the 6 views.
   - (ii) A 2-layer MLP on the same input.
   - (iii) An attentive probe: learned queries cross-attending over all 6 × N patch tokens, which pools the views naturally.
   - Sweep the learning rate and weight decay for each backbone on the validation buoys.
   - The attentive-vs-linear gap (+5 points for DINOv2 in V-JEPA) is larger than most published backbone gaps, so a linear-only comparison could pick the wrong backbone.
4. **Fine-tuning for the top 2 or 3.**
   - LP-FT, starting the head from the best probe.
   - Layer-wise LR decay for the ViTs. The 0.65 to 0.75 values common in MAE and BEiT recipes come from my prior knowledge and are unverified.
   - A non-zero drop-path rate.
   - LoRA (r = 8 to 16) on the attention projections as a separate arm.
   - The same augmentation and epoch budget for every arm, early stopping on the validation buoys, and a learning-rate sweep for each backbone, since full fine-tuning of DINOv2 collapses under a shared grid (Section 4).
5. **Seeds.**
   - The current ConvNeXt seeds already differ noticeably.
   - Use ≥ 5 seeds per fine-tuned arm, randomising initialisation, data order and augmentation together (Bouthillier).
   - Probes are close to deterministic, so 3 seeds suffice.
6. **Statistics.**
   - Report balanced accuracy on the wind-sea classes, MAE and RMSE in metres, within-one-class accuracy, and the regression slope of predicted against observed H_s (to show dilution).
   - Compare arms by paired differences with a **cluster bootstrap over test buoys**: resample buoys with replacement, compute the metric for both arms on the same resample, and average over seeds. Also show a table per buoy.
   - With 12 clusters, intervals will be wide. Declare in advance a smallest difference worth acting on, for example ≥ 0.02 balanced accuracy with a 95 % interval excluding 0.
7. **Pitfall checklist.**
   - Never split by strip or frame.
   - Give each backbone its own preprocessing: normalisation statistics, input resolution, and the patch-14 vs patch-16 token grid.
   - Never use test-buoy curves for early stopping.
   - Report the regression slope.
   - Keep the label time offset fixed across arms.
   - Expect neighbouring buoys at the same hour to be correlated, so the 12 test buoys are not 12 fully independent samples.

### Gaps
- DINOv2's own linear-evaluation protocol (as I remember it: CLS token concatenated with average-pooled patch tokens, a grid over the learning rate and over the last 1 or 4 layers) was not verified against the paper this session.
- No source gave measured compute costs for probes against fine-tuning. The figures above are estimates.
- I did not find a source that quantifies how many seeds are needed for small vision regression with a site split. The recommendation of 5 follows Bouthillier's range and the project's observed seed spread.

## 4. Efficient fine-tuning (LoRA, adapters, partial unfreezing) of ViT foundation models on small data: does it beat full fine-tuning or frozen probes under domain shift?

### Takeaway
On low-shot benchmarks (1,000 samples per task), carefully tuned LoRA and adapter methods beat both linear probing and full fine-tuning on all 19 VTAB-1K tasks. The gap is largest for DINOv2 backbones and for structured, quantity-like tasks, where naive full fine-tuning of DINOv2 collapsed. Under distribution shift, these methods preserve a pretrained model's robustness better than full fine-tuning does. However, full fine-tuning combined with weight-space ensembling (WiSE) did better still on both target and shifted data, and LP-FT gives about +10 % out of distribution over plain fine-tuning. All of this evidence comes from classification benchmarks. I found no study of LoRA vs full fine-tuning vs probes for geophysical camera regression at unseen sites.

### Cited Findings
- Mai et al. (OSU and Google; arXiv:2409.16434 v5, 25 March 2025) compared linear probing, full fine-tuning and 14 parameter-efficient methods (including LoRA, VPT, Adapter, AdaptFormer, Convpass, BitFit, SSF) on ViTs — [arXiv:2409.16434](https://arxiv.org/abs/2409.16434)
  - With tuned hyperparameters, these methods "consistently outperform linear probing and full FT on all 19 image classification tasks (with 1,000 training samples) in VTAB-1K".
  - They also match or beat full fine-tuning when much more data are available, while training far fewer parameters.
  - With a CLIP ViT-B/16 fine-tuned on ImageNet (100 samples per class), they preserve the average accuracy on ImageNet-V2, -S, -R and -A better than full fine-tuning does. But full fine-tuning with WiSE "can achieve even higher accuracy in both downstream and distribution shift data".
- **DINOv2 vs IN-21k backbones** (same source, Table 7). Linear probing generally improves with DINOv2 features, while "full fine-tuning performance significantly worsens with DINOv2", which the authors attribute to overfitting. Examples (DINOv2, linear / full FT / LoRA, %):

  | Task | Linear | Full FT | LoRA |
  |---|---|---|---|
  | Resisc45 | 78.8 | 25.9 | 84.2 |
  | KITTI distance | 60.3 | 47.1 | 83.8 |
  | Clevr counting | 47.5 | 27.3 | 89.8 |
  | Pets | 93.4 | 68.7 | 94.1 |
  | DTD | 74.9 | 45.2 | 78.4 |

  The Houlsby Adapter failed on DINOv2 (for example 7.1 % on Pets), so not every efficient method transfers to every backbone.
- When the downstream domain is far from the pretraining data, methods with more parameters work better, and full fine-tuning "risks washing out" useful pretrained knowledge on domains close to the pretraining data — [arXiv:2409.16434](https://arxiv.org/abs/2409.16434)
- Kumar et al. 2022 (LP-FT): fine-tuning distorts pretrained features, which costs about 7 % out of distribution against linear probing. LP-FT gives about +10 % out of distribution and +1 % in distribution over full fine-tuning across 10 shift datasets — [arXiv:2202.10054](https://arxiv.org/abs/2202.10054)
- Small-data fine-grained classification with no shift test (DINOv2-Base, 394 font classes): linear probe 36.6 %, LoRA r = 8 98.7 %, full fine-tuning 93.4 %. The gap between training and validation loss widened under full fine-tuning but stayed narrow under LoRA — [arXiv:2602.13889](https://arxiv.org/html/2602.13889)
- A student project, not peer-reviewed (DINOv2-S on Oxford Pets): LoRA r = 16 scored 94.41 % against 93.89 % for full fine-tuning, training 1.44 % of the parameters — [GitHub](https://github.com/SalajBansal05/parameter-efficient-finetuning-lora)
- Work on LLaMA2-7B, a language model rather than a vision model, reports that LoRA and full fine-tuning reach different solutions ("intruder dimensions") — [arXiv:2410.21228](https://arxiv.org/html/2410.21228v3)
- Vorkapić 2024 is the only wave study found that uses partial unfreezing: the top layers of ImageNet CNNs and a ViT, after a stage with a frozen backbone (an LP-FT-like schedule). It did not compare against full fine-tuning or a frozen probe — [arXiv:2407.03755](https://arxiv.org/abs/2407.03755)

### Inferences
- The DINOv2 full fine-tuning collapse in Mai et al. probably reflects a shared hyperparameter grid that suits DINOv2 poorly. DINOv2 is known to need small learning rates and layer decay; this is my prior knowledge, not verified this session. For our comparison, either give full fine-tuning of each ViT its own learning-rate and layer-decay sweep, or report it with that caveat. Otherwise a result such as "the foundation model loses to ConvNeXt" may be an artefact of the recipe.
- Our data are "low-shot" in the sense that matters for unseen buoys. There are about 246k view images, but only 67 training sites, so the diversity of camera domains is small. That is the regime where LP-FT and efficient fine-tuning help most in the benchmarks above, and the closest analogue is Kumar's FMoW geographic shift.
- **Suggested adaptation arms:**
  - LP-FT of ConvNeXt-T, a cheap upgrade of the current recipe;
  - LoRA (r = 8 to 16) on DINOv2 ViT-B/14, starting the head from the attentive probe;
  - optionally, full fine-tuning of the ViT with layer decay.
- WiSE (averaging the weights of the LP-initialised model and the fine-tuned model) costs nothing after training and can be checked on the validation buoys.

### Gaps
- No geophysical or environmental study found compares LoRA or adapters with full fine-tuning and frozen probes at unseen sites.
- No regression-specific evidence on efficient fine-tuning was found: all benchmark evidence above is classification. The DINOv3 pasture-biomass LoRA paper (*Sensors* 2026, [doi:10.3390/s26165285](https://doi.org/10.3390/s26165285)) may be the closest and was not read.
- None of the sources read report seed-to-seed variance of efficient fine-tuning against full fine-tuning.
