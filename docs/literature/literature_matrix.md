# Literature Matrix

Verification levels (V1 full text / V2 abstract / V3 project notes) are
explained in [source_verification.md](source_verification.md). Per-paper notes:
[`papers/`](papers/).

## Core reviewed papers

| | Choi 2020 | Kim 2023 | Kamagata 2026 | Yang 2026 |
|---|---|---|---|---|
| **Venue** | Ocean Eng. 201 | Ocean Eng. 288 | arXiv 2607.11998 (IEEE conf.) | Ocean Eng. 352 |
| **Verification** | V2 / V3 | V2 / V3 | **V1** | V2 / V3 |
| **Real or simulated** | real | real | real (video) | **simulated** (digital twin) |
| **Platform** | NDBC buoy cameras; container ship | fixed camera on WEC, Jeju (SW Korea) | shore cameras | virtual ship |
| **Input** | single image (cls); 16-frame sequences (reg) | grayscale frames / sequences up to 300 s | 16-frame clips, 10 fps, 224² | image sequence + IMU attitude |
| **Ground truth** | NDBC buoy (cls); shipboard **wave radar** (reg) | seafloor **ADCP (Nortek AWAC)** | **not stated** (see caveat) | **virtual buoys** (zero-crossing + spectral) |
| **Label timing** | ~30 min image–wave offset (NDBC) | synchronised (details V3) | clip-level annotations | synchronous by construction |
| **Output** | low/high class; continuous H_s | 5 classes of mean wave height (0.5 m bins) | H_s, H_max, T_p, T_z, θ | H_s |
| **Range** | cls excludes 1.52–3.05 m | 0.75–3.25 m | H_s 0.85–2.80 m | H_s ≲ 2.5 m, 47 sea states |
| **Model** | VGG/Inception/ResNet/DenseNet; VGG + BiConvLSTM | 4-block CNN; GoogLeNet + BiLSTM(2000) | V-JEPA ViT-S + SlowFast transformer + optical-flow crops + physics loss | ResNet-18 + attitude MLP + multi-head self-attention |
| **Headline result** | 84 % accuracy; MSE 0.0177 | accuracy 0.783 → 0.904 (180 s) → 0.932 (+aug.) | **held-out** H_s R² 0.107, PCC 0.436; T_p R² 0.203 | RMSE 0.032 m, R² 0.994 (simulation) |
| **Data scale** | 44,620 pairs; 3,019 clips | ~1 h/day, several days | **6 scenes**, 894 windows | ~4,700 pairs |
| **Data public?** | NDBC source public; pairs not released | no (statement inconsistent) | no statement; test clips from CC BY-NC-ND dataset | no |
| **Key limitation** | timestamp offset; brightness filtering; one ship | single site; daytime; filtered weather; class imbalance | label provenance; tiny, narrow-range data; "full-set" metrics include training data | simulation-only; domain gap |
| **Key lesson for this project** | raw imagery carries sea-state signal; temporal models enable regression | temporal context beats snapshots (0.78 → 0.90) | architecture has outrun data; low R² with high PCC indicates regression dilution | platform motion must be modelled; attention fusion > concatenation (in simulation) |

## Cross-cutting observations

1. **Reference instruments differ in every paper**: buoy, radar, ADCP,
   unspecified, virtual buoy. H_s estimators differ too (H₁/₃ vs H_m0, a ~5 %
   systematic difference). Numbers across papers are not directly comparable.
2. **No paper reports a cross-station test with instrument labels.**
   Kamagata's held-out clips come from other sites, but their labels are not
   instrument-traceable.
3. **Difficult conditions are excluded** (night, glare, fog, rain, ships)
   rather than annotated. Performance where monitoring matters most is
   unknown.
4. **Temporal context helps in every study that tests it** (Kim: 180 s >
   snapshot; Yang: sequence > single image; Choi: sequence enables regression).
5. **None release code and data together**, so none can be independently
   reproduced.

## Adjacent literature

The full survey (modalities, models, industry, and open data) is in
[landscape_review.md](landscape_review.md). The table below lists anchors only.

| Topic | Key references | Relevance |
|---|---|---|
| Argus coastal imaging | Holman & Stanley 2007 | calibrated camera products (timex, variance, timestacks), the basis of CoastCam |
| Stereo wave reconstruction | Benetazzo 2006; Bergamasco et al. 2017 (WASS) | geometric space–time surface ground truth from video |
| Wave-breaking video datasets | Yin et al. 2025 (VWBT) | public multi-site coastal video; breaking-type labels only |
| Video representation learning | Bardes et al. 2024 (V-JEPA); Feichtenhofer et al. 2019 (SlowFast) | self-supervised pretraining on unlabelled ocean video |
| X-band radar wave inversion | WaMoS II; Kwon et al. 2023 (3-D CNN, public data) | the radar reference used by Choi et al.'s regression dataset; radar analogue of the camera task |
| Deep learning on shore imagery | Buscombe et al. 2020 (OWG); arXiv:2606.13302 (T_p, V1) | instrument-labelled single-site (OWG) vs expert/pseudo labels (2606.13302) |
| SAR deep learning | Quach et al. 2021 (operational S-1 WV Hs since 2022) | template: independent-sensor labels at scale + temporal hold-out |
| Whitecaps and horizon geometry | Brumer et al. 2017; Schwendeman & Thomson 2015a,b | physical features for single-frame buoy cameras (Q-M4, Q-M5) |
| Stereo datasets | Guimarães et al. 2020 (CC BY 4.0) | geometric ground truth for spectral validation |
