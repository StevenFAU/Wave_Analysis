# Pretrained image backbones as feature extractors for camera-based wave-height estimation (status: November 2026)

Scope: supervised CNNs/ViTs (ConvNeXt V1/V2, EfficientNetV2, ResNet, ViT/DeiT/Swin), self-supervised and vision-language image models (DINOv2 ± registers, DINOv3, MAE, iBOT, I-JEPA, SigLIP/SigLIP 2, CLIP/OpenCLIP, EVA), and 2025-2026 successors (Web-SSL, Perception Encoder, RADIO family). Target use: NDBC buoy-camera views (480x270 RGB, six per strip, ~41k strips from 79 buoys, labels Hs / wind-wave / swell height in metres, test at unseen buoys; current baseline ImageNet ConvNeXt-Tiny fine-tuned per view, balanced accuracy ~0.80 on wind-sea classes), plus shore frames at 2688x1520. Compute: one shared H200 per run.

Conventions: "verified" means the number was read from the primary source (paper HTML, official repo or model card) in this session. Items taken from search-result snippets only (page not fetched, or the fetch was refused with a 403) are marked "(snippet only)". Arithmetic and judgments of mine are under "Inferences", never under "Cited Findings".

---

## Q1. Evidence that each backbone transfers to texture-heavy, fine-grained, dense or regression tasks and to OOD/domain-shifted data

### Takeaway
For dense and geometric regression with frozen features (depth RMSE, a close analogue of "sea-surface roughness / crest geometry" regression), self-supervised DINO-family models are clearly ahead. DINOv3 leads at every size, then come RADIO and PE-spatial (both distilled partly from DINO), then DINOv2. Language-supervised models (SigLIP 2, PE-core, CLIP) are much weaker on dense or metric probes but equal or better on global fine-grained and texture classification (e.g. OpenCLIP beats DINOv2 on DTD textures). MAE is poor frozen. The most decision-relevant result for this project is that DINOv3's distilled ConvNeXt-Tiny (same architecture and size as the current baseline) beats supervised ImageNet-22k ConvNeXt-T by large margins on dense, depth and robustness benchmarks. It also does not degrade at higher input resolution, as the supervised model does.

### Cited Findings

**DINOv3 (Siméoni et al., arXiv 2508.10104, submitted 13 Aug 2025): frozen dense linear probes (Table 3)**
- ADE20k mIoU / NYUv2 depth RMSE (lower is better) / KITTI RMSE: DINOv3 7B/16 55.9 / 0.309 / 2.346; AM-RADIOv2.5 g/14 53.0 / 0.340 / 2.918; PE-spatial G/14 49.3 / 0.362 / 3.082; DINOv2 g/14 49.5 / 0.372 / 2.624; Franca g/14 46.3 / 0.445 / 3.140; Web-DINO 7B/14 42.7 / 0.466 / 3.158; SigLIP 2 g/16 42.7 / 0.494 / 3.273; PE-core G/14 38.9 / 0.590 / 4.119 — [DINOv3 paper HTML](https://arxiv.org/html/2508.10104v1)
- 3D correspondence recall, NAVI (geometric) / SPair (semantic): DINOv3 64.4 / 58.7; DINOv2 60.1 / 56.1; AM-RADIOv2.5 59.4 / 56.8; SigLIP 2 49.4 / 42.6; PE-core 39.9 / 23.1 — [DINOv3 paper HTML](https://arxiv.org/html/2508.10104v1)
- Frozen DINOv3 with a depth head gets NYU ARel 4.3, against 4.4 for fine-tuned Depth-Anything-v2 ViT-g. Frozen DINOv3 segmentation reaches 63.0 mIoU on ADE20k, matching fine-tuned ONE-PEACE. The authors state that "with DINOv3, finetuning is not necessary to obtain strong performance" (on their benchmarks) — [DINOv3 paper HTML](https://arxiv.org/html/2508.10104v1)

**DINOv3: global linear probes and OOD robustness (Table 7)**
- ImageNet val / ImageNet-R / Sketch / ImageNet-A / ImageNet-C (mCE, lower is better) / ObjectNet: DINOv3 7B 88.4 / 91.1 / 71.3 / 86.9 / **19.6** / 79.0; DINOv2 g/14 87.3 / 81.1 / 65.4 / 81.7 / 24.1 / 66.4; SigLIP 2 g/16 89.1 / 92.2 / 71.8 / 84.6 / 30.0 / 78.6; PE-core G/14 89.3 / 92.2 / 71.9 / 89.0 / 22.7 / 80.2; AIMv2 3B 87.9 / 82.3 / 67.1 / 74.5 / 29.5 / 69.0; EVA-CLIP 18B 87.9 / 85.2 / 64.0 / 81.6 / 33.0 / 71.9; Web-DINO 7B 85.9 / 75.6 / 64.0 / 71.6 / 31.2 / 69.7 — [DINOv3 paper HTML](https://arxiv.org/html/2508.10104v1)
- Fine-grained linear probes, Fine-S (mean of 12 datasets) / iNat21: DINOv3 93.0 / 89.8; DINOv2 92.6 / 86.1; SigLIP 2 93.7 / 82.7; PE-core 94.5 / 87.0; EVA-CLIP 92.9 / 83.5 — [DINOv3 paper HTML](https://arxiv.org/html/2508.10104v1)
- OCR- and logo-heavy classification (GTSRB / Logo-2K+): PE-core G 94.8 / 93.2; DINOv3 7B 87.5 / 86.0; DINOv2 g 78.2 / 52.9. Language supervision still wins where reading text or symbols matters — [DINOv3 paper HTML](https://arxiv.org/html/2508.10104v1)

**DINOv3 distilled family vs matched-size competitors (Table 14: IN-ReaL / ADE20k / NYU RMSE)**
- ViT-S: DINOv2 87.3 / 45.5 / 0.446 vs DINOv3 87.0 / 47.0 / 0.403; ViT-S+ DINOv3 88.0 / 48.8 / 0.399 — [DINOv3 paper HTML](https://arxiv.org/html/2508.10104v1)
- ViT-B: PE-core 87.5 / 37.4 / 0.641; SigLIP 2 89.3 / 41.6 / 0.512; DINOv2 89.0 / 48.4 / 0.416; DINOv3 89.3 / 51.8 / 0.373 — [DINOv3 paper HTML](https://arxiv.org/html/2508.10104v1)
- ViT-L: PE-core 90.1 / 39.7 / 0.650; SigLIP 2 90.1 / 43.6 / 0.484; DINOv2 89.7 / 48.8 / 0.394; DINOv3 90.2 / 54.9 / 0.352; ViT-H+ DINOv3 90.3 / 54.8 / 0.352 (H+ gives no gain over L on these) — [DINOv3 paper HTML](https://arxiv.org/html/2508.10104v1)

**DINOv3-distilled ConvNeXt vs supervised ImageNet-22k ConvNeXt (Table 15; "supervised → DINOv3", eval resolution 256/512 where two values are given)**
- ConvNeXt-T: IN-ReaL 87.3/83.0 → 86.6/87.7; ImageNet-R 45.0/33.0 → 73.7/74.1; ObjectNet 44.5/27.1 → 52.6/58.7; ADE20k 24.8 → 42.7; NYU RMSE 0.666 → 0.448 — [DINOv3 paper HTML](https://arxiv.org/html/2508.10104v1)
- ConvNeXt-B: IN-R 57.3/46.2 → 77.2/78.2; ADE20k 26.5 → 46.3; NYU 0.596 → 0.420. ConvNeXt-L: ADE20k 33.3 → 47.8; NYU 0.567 → 0.403 — [DINOv3 paper HTML](https://arxiv.org/html/2508.10104v1)
- The paper states that the supervised models "significantly degrade" at 512 resolution while the DINOv3 ConvNeXts improve. In the extracted table the S-size IN-R and ObjectNet values are identical to the T-size values; this may be an extraction or typesetting artefact — [DINOv3 paper HTML](https://arxiv.org/html/2508.10104v1)

**DINOv2 (Oquab et al., arXiv 2304.07193, TMLR 2024): frozen linear probes**
- DTD (texture) linear top-1: OpenCLIP ViT-G/14 **86.0**; DINOv2 g/14 84.5, L/14 84.0, B/14 83.3, S/14 80.6; iBOT L/16 80.7; DINO B/8 79.6; MAE H/14 75.4. On texture classification, the language-supervised OpenCLIP-G is slightly ahead of DINOv2 — [DINOv2 paper HTML](https://arxiv.org/html/2304.07193)
- 12-dataset fine-grained average: OpenCLIP-G 91.9; DINOv2 g 92.1 / L 91.2 / B 90.1 / S 87.7; iBOT-L 86.6; DINO B/8 85.1; MAE-H 78.0 — [DINOv2 paper HTML](https://arxiv.org/html/2304.07193)
- Depth RMSE, NYUd linear-1 / DPT and KITTI linear-1 / DPT: DINOv2 g 0.344 / 0.279 and 2.62 / 2.11; DINOv2 S 0.449 / 0.356 and 3.10 / 2.34; iBOT-L 0.417 / 0.358; MAE-H 0.517 / 0.415; OpenCLIP-G 0.541 / 0.414 and 3.57 / 2.56. The smallest DINOv2 (S/14) is better than OpenCLIP-G and MAE-H for depth — [DINOv2 paper HTML](https://arxiv.org/html/2304.07193)
- iNat2021 linear: DINOv2 g 85.7, B 81.1, S 74.2; OpenCLIP-G 76.0; MAE-H 32.3 — [DINOv2 paper HTML](https://arxiv.org/html/2304.07193)

**SigLIP 2 (Tschannen et al., arXiv 2502.14786, 20 Feb 2025): frozen dense probes (Table 2)**
- So/14 at 224, PASCAL / ADE20k mIoU / NYUv2 depth RMSE / NYUv2 normals error: CLIP L/14 74.5 / 39.0 / 0.553 / 24.3; SigLIP So/14 72.0 / 37.6 / 0.576 / 25.9; SigLIP 2 So/14 77.1 / 41.8 / 0.493 / 24.9. At 384: SigLIP 73.8 / 40.8 / 0.563 / 24.1; SigLIP 2 78.1 / 45.4 / 0.466 / 23.0. The gains come from adding self-distillation and masked prediction (SILC/TIPS) in the last 20% of training — [SigLIP 2 paper](https://arxiv.org/html/2502.14786)

**DINOv2 with registers (Darcet et al., arXiv 2309.16588, ICLR 2024)**
- Without registers, large ViTs produce high-norm artefact tokens (about 10x norm, about 2% of tokens) in low-information background patches. DINOv2 0 → 4 registers: ImageNet linear 84.3 → 84.8; ADE20k 46.6 → 47.9; NYUd RMSE 0.378 → 0.366; LOST object discovery VOC07 35.3 → 55.4. In DINOv2 the artefacts appear only in ViT-L and larger; OpenCLIP and DeiT-III show them at base and large sizes — [Registers paper](https://arxiv.org/html/2309.16588)

**Older and general comparison studies**
- Battle of the Backbones (Goldblum et al., NeurIPS 2023; 1,500+ runs): "convolutional neural networks pretrained in a supervised fashion on large training sets still perform best on most tasks" among the models they considered. SSL is competitive in like-for-like comparisons. This is a 2023 study, superseded by DINOv3-era evidence for dense tasks — [arXiv 2310.19909](https://arxiv.org/abs/2310.19909)
- Dense attentive probing over 18 backbones (Lüddecke & Ecker, TMLR 2025): DINOv2 outperformed all other tested backbones in all three dense categories (instance awareness, local semantics, spatial understanding), including mask- and language-supervised models — [TMLR 2025 abstract](https://mlanthology.org/tmlr/2025/luddecke2025tmlr-characterizing)
- Probe3D (El Banani et al., CVPR 2024) probes frozen features for depth, surface normals and multiview correspondence and reports "several limitations" of current models; per-backbone numbers were not on the pages fetched — [arXiv 2404.08636](https://arxiv.org/abs/2404.08636), [code](https://github.com/mbanani/probe3d)

**Remote-sensing transfer (DINOv3 web vs satellite)**
- GEO-Bench classification mean: DINOv3 Web 7B (RGB) 81.6; DINOv3 Sat 7B 81.1; Sat ViT-L 79.6; DOFA 79.9; best Prithvi-v2 79.6. Segmentation mean: Web 7B 75.9; Sat 7B 75.0; Sat ViT-L 74.5; best Prithvi-v2 72.8 — [DINOv3 paper HTML](https://arxiv.org/html/2508.10104v1)
- Canopy-height regression: the Sat 7B model reduces MAE from 2.4 to 2.2 (SatLidar val) and from 3.4 to 3.2 (Open-Canopy). The paper says the satellite model "excels in metric tasks like depth estimation", while the web model leads on semantic geospatial tasks — [DINOv3 paper HTML](https://arxiv.org/html/2508.10104v1)

### Inferences
- Wave height from a buoy camera is a metric, geometry-plus-texture regression (crest height relative to the horizon, whitecap fraction, surface roughness). The closest public proxies are NYU and KITTI depth RMSE and dense probes, where the order is DINOv3 > RADIO ≈ PE-spatial > DINOv2 >> SigLIP 2 > PE-core/CLIP > MAE. Texture classification (DTD) shows smaller gaps and a slight CLIP advantage, so a CLIP or SigLIP 2 model is a reasonable second family to test as a complement, not as the primary.
- The DINOv3 ConvNeXt-T result is the most actionable item. It is a like-for-like swap for the current ImageNet ConvNeXt-Tiny (29M params, same timm-style architecture), with +17.9 ADE20k mIoU, NYU RMSE 0.666 → 0.448 and ImageNet-R 45.0 → 73.7. The supervised model degrades at 512 px, and our views (480x270, ~360 px equivalent) already exceed the 224/256 training resolution of supervised checkpoints.
- For ground-level ocean cameras, the web-trained (LVD-1689M) DINOv3 is the right default. The satellite (SAT-493M) variant is trained on nadir imagery, and even on remote-sensing benchmarks the web 7B model equals or beats it on semantic tasks.
- Different papers' numbers for the same model are not directly comparable. DINOv2-g ImageNet-C is 28.2 in the DINOv2 paper and 24.1 in the DINOv3 paper; NYU linear is 0.344 vs 0.372. Evaluation protocols (resolution, head) differ, so compare only within a table.

### Gaps
- No paper found that reports DINOv3 or DINOv2 on a texture/material benchmark (DTD, FMD, MINC) next to supervised ConvNeXt at matched size; DTD numbers were not in the DINOv3 text extracted.
- Per-backbone Probe3D surface-normal numbers, and Battle-of-the-Backbones per-model numbers (including whether DINOv2 was in it), were not retrieved.
- No verified counting benchmark comparisons were found (e.g. FSC-147 with frozen DINOv2/v3 vs CNN).
- EVA-02, iBOT and I-JEPA dense and regression numbers beyond the DINOv2 tables were not retrieved; the I-JEPA repo page lists no linear-probe numbers in text.

---

## Q2. Frozen-feature probing (linear / MLP / attentive) vs full or partial fine-tuning on small datasets (10k-50k images), and when frozen DINOv2/v3 matches fine-tuned ImageNet CNNs

### Takeaway
The evidence divides along two lines: (a) whether the test set is in-distribution or OOD, and (b) whether the signal is global and coarse or small and focal. Frozen DINOv2/v3 with a linear or attentive head can match a fine-tuned ConvNeXt on natural-image, domain-specific classification. Fine-tuning still wins when the signal is small or local (small lesions, fonts, defects) or the modality shifts. On OOD tests (our unseen-buoy split), linear probing or LP-then-FT beats naive full fine-tuning. For 10k-50k labelled images the most defensible protocol is: frozen DINOv3 features with an attentive or patch-level probe as a baseline, then LP-FT (or fine-tune with a low backbone LR) of a DINOv3 initialisation, preferably the ConvNeXt-T/B or ViT-S/B distilled models.

### Cited Findings
- Fine-tuning vs linear probing across 10 distribution-shift datasets: fine-tuning averaged about 2% higher accuracy in-distribution but about 7% lower accuracy OOD. LP-FT (linear probe first, then full fine-tune) gave about 1% higher ID and about 10% higher OOD accuracy than full fine-tuning — [Kumar et al., ICLR 2022](https://arxiv.org/abs/2202.10054)
- DINOv2 ViT-g/14 on ImageNet-1k: linear 86.5 → fine-tuned 88.5 at 224 (+2.0), and 86.7 → 88.9 at 448 (+2.2). This is the gain from fine-tuning on a large in-distribution dataset — [DINOv2 paper HTML](https://arxiv.org/html/2304.07193)
- Underwater classification (AQUA20; Rost et al., arXiv 2604.00313, Mar-Aug 2026): logistic regression on frozen DINOv3 ViT-B/16 embeddings reached 91.5% macro-F1 (bootstrap 95% CI 89.0-93.7) with all labels, above a published fully supervised ConvNeXt at 88.9%. It reached 88.5% with 144 images per class and 81.8% with 13 per class — [arXiv 2604.00313](https://arxiv.org/abs/2604.00313)
- Chest radiographs (Tayebi Arasteh et al., Commun Med 2026; 816,183 images, 7 datasets; full fine-tuning): at 224 px DINOv3 did not consistently beat DINOv2; at 512 px DINOv3 was the strongest initialisation, especially with ConvNeXt-B. ConvNeXt-B beat ViT-B/16 under both full and parameter-efficient adaptation. Gains were largest for small focal and boundary-dependent findings. Frozen DINOv3-7B features were inferior to task-adapted ViT-B/16 and ConvNeXt-B (the frozen-vs-adapted part is from the search snippet; the abstract confirms the rest) — [arXiv 2510.07191](https://arxiv.org/abs/2510.07191)
- Industrial inspection (Gharbage et al., CVPR 2026 workshop): with a frozen backbone, DINOv3-distilled ConvNeXt shows "no clear advantage" over supervised ImageNet ConvNeXt. With full fine-tuning on RGB tasks it is a better initialisation (faster convergence, better final performance). On X-ray (modality shift), supervised ImageNet was better in both regimes. The abstract gives no numbers — [arXiv 2605.23472](https://arxiv.org/abs/2605.23472)
- Font classification: frozen DINOv2-B linear probe 36.6% vs 93.4% with full fine-tuning. Frozen features alone were insufficient for fine-grained, non-natural-image discrimination (snippet only) — [arXiv 2602.13889](https://arxiv.org/html/2602.13889v2)
- Chest X-ray small lesions: frozen foundation-model embeddings "discard small-lesion signal"; a fine-tuned ResNet-50 was the reference (snippet only) — [arXiv 2606.11606](https://arxiv.org/pdf/2606.11606)
- Attentive probing: Efficient Probing (Psomas et al., arXiv 2506.10178, Jun 2025, v2 Oct 2025) beats linear probing and prior attentive probes on seven benchmarks. It generalises beyond MIM to other pretraining paradigms and gains most in low-shot settings. The motivation is that linear probing on CLS or pooled features underrates models whose information is spread across patch tokens — [arXiv 2506.10178](https://arxiv.org/abs/2506.10178)
- Dense attentive probing (cross-attention with learnable distance masks) works for any backbone and resolution, and ranks DINOv2 first across 18 backbones — [TMLR 2025](https://mlanthology.org/tmlr/2025/luddecke2025tmlr-characterizing)
- Perturbation study: fine-tuning improved robustness to common perturbations while keeping clean zero-shot accuracy nearly unchanged — [Liu et al., arXiv 2604.14973](https://arxiv.org/html/2604.14973)

### Inferences
- Our setting (about 41k strips, 79 buoys, test at unseen buoys) is an OOD problem in the Kumar et al. sense. Full fine-tuning of the baseline may be overfitting buoy-specific appearance (housing, mast and horizon placement, colour cast). Kumar et al. predict a frozen probe or LP-FT should narrow the seen-vs-unseen gap even when ID accuracy is lower. This is worth testing explicitly by reporting seen-buoy and unseen-buoy scores side by side.
- Whitecaps and short wind-sea texture are "small, focal" signals, like small lesions and defects, where frozen features underperform in the chest-X-ray and industrial studies. A CLS-token linear probe is therefore likely the weakest frozen protocol for us. An attentive probe over patch tokens, or multi-layer patch features, is the fair frozen comparison.
- Practical ranking to try, cheapest first. (1) Frozen DINOv3 ViT-B/16 or ViT-L/16 patch tokens with an attentive probe, run per view and pooled across the 6 views. (2) The same setup with DINOv2-reg ViT-B/L as a control. (3) LP-FT of DINOv3 ConvNeXt-T, a direct replacement for the current baseline. (4) LP-FT of DINOv3 ViT-S/B with layer-wise LR decay.

### Gaps
- No study found at our exact scale (10k-50k images) that compares frozen DINOv2/v3 probes with fine-tuned ImageNet ConvNeXt on a regression target under site or camera hold-out. The closest are classification studies (AQUA20, chest X-ray, industrial).
- Exact Efficient Probing parameter, FLOP and accuracy numbers were not extracted.
- MLP-probe vs linear-probe vs attentive-probe comparisons for DINOv3 specifically were not found.

---

## Q3. Patch size, input resolution (224 / 448 / 518 / higher), and dense patch features vs CLS for small-scale texture

### Takeaway
Resolution matters most where the signal is small. DINOv3 was explicitly trained to keep dense features stable at high resolution (to 4k+), and evidence from other domains shows the DINOv3 gain appears at 512 px rather than 224. Patch-token features carry the local information; CLS or linear probes under-use it, and registers clean up the artefacts in large ViTs. With patch 16, a 480x270 view gives only 30x17 tokens, so each token covers 16x16 px. Upsampling the views, or tiling the 2688x1520 shore frames, is worth testing for whitecap-scale detail.

### Cited Findings
- Patch sizes: DINOv2 uses 14 (S/B/L/g) and ends pretraining with a 518x518 high-resolution phase — [DINOv2 paper HTML](https://arxiv.org/html/2304.07193). All DINOv3 ViTs use 16 — [DINOv3 repo](https://github.com/facebookresearch/dinov3). Web-DINO uses 14 at 224/378/518 — [Web-SSL repo](https://github.com/facebookresearch/webssl). C-RADIOv4 uses 16 and accepts up to 2048 px in 16-px steps — [C-RADIOv4-H card](https://huggingface.co/nvidia/C-RADIOv4-H). PE-core B/16 at 224, L/14 at 336, G/14 at 448; PE-spatial T/S/B/16 at 512 — [PE repo](https://github.com/facebookresearch/perception_models)
- DINOv3 high-resolution behaviour: "stable feature maps at resolutions above 4k". ViT-S+ features are stable from 896x512 to 3584x2048; ViT-L starts drifting only at 7168x4096; ViT-H+ is stable over the whole tested range. The depth head was trained at 1024x768. VGGT was moved from 518 to 592 px to match patch 16 — [DINOv3 paper HTML](https://arxiv.org/html/2508.10104v1)
- DINOv3 repo examples use 256 for classification transforms, 1024 for depth and 896 for segmentation — [DINOv3 repo](https://github.com/facebookresearch/dinov3)
- Supervised ImageNet-22k ConvNeXt-T IN-ReaL drops from 87.3 (256) to 83.0 (512), while DINOv3 ConvNeXt-T rises from 86.6 to 87.7. ObjectNet: supervised 44.5 → 27.1 vs DINOv3 52.6 → 58.7 — [DINOv3 paper HTML](https://arxiv.org/html/2508.10104v1)
- Chest X-ray: DINOv3's advantage appeared at 512 px, mainly for small focal findings. 1024 px "rarely improved performance" at markedly higher compute — [arXiv 2510.07191](https://arxiv.org/abs/2510.07191)
- SigLIP 2 So/14 from 224 to 384: ADE20k 41.8 → 45.4; NYUv2 RMSE 0.493 → 0.466 — [SigLIP 2 paper](https://arxiv.org/html/2502.14786)
- Registers: DINOv2 artefact tokens sit in low-information background (sky or flat sea is the analogue here); 4 registers improve dense metrics (ADE +1.3, NYU 0.378 → 0.366). Artefacts appear only in DINOv2 L and larger — [Registers paper](https://arxiv.org/html/2309.16588)
- timm DINOv3 ViT-L/16 at 256 px outputs 261 tokens, i.e. 256 patches plus 5 prefix tokens (the card does not say whether this is CLS + 4 registers). It uses RoPE; the card suggests casting RoPE periods to bf16 then fp32 to match the original outputs — [timm DINOv3 ViT-L card](https://huggingface.co/timm/vit_large_patch16_dinov3.lvd1689m)
- RADIO returns a "summary" (CLS-like) tensor and spatial features. The spatial features are meant for dense tasks, and the card says dividing input H and W by 16 gives the feature-map grid — [C-RADIOv4-H card](https://huggingface.co/nvidia/C-RADIOv4-H). RADIO accepts non-square inputs and gets higher zero-shot scores when only the shorter side is fixed — [RADIO repo](https://github.com/NVlabs/RADIO)
- Linear probing underrates models whose information is spread over patch tokens; attentive probes recover it — [arXiv 2506.10178](https://arxiv.org/abs/2506.10178)

### Inferences
- Token arithmetic (my calculation). A 480x270 view at patch 16 resized to 480x272 gives 30x17 = 510 tokens. At patch 14, 476x266 gives 34x19 = 646 tokens. ConvNeXt's stride-32 final map is about 15x8. Shore frames at 2688x1520 and patch 16 give 168x95 = 15,960 tokens, which is too many for routine training. Use 1344x760 (84x47, about 4k tokens) or overlapping tiles.
- Distant whitecaps in a 480x270 buoy view may span only a few pixels, below one 16-px patch. Two cheap tests: (i) upsample views 1.5-2x (e.g. 960x544 gives 60x34 = 2,040 tokens), which DINOv3 tolerates by design; (ii) use multi-layer patch features (intermediate blocks) rather than only the last layer. Both should be ablated, not assumed.
- Do not centre-crop to square. The horizon line and far-field texture lie along the long axis. All the RoPE-based ViTs (DINOv3) and RADIO take non-square inputs natively. DINOv2 needs position-embedding interpolation, which timm handles via dynamic image size.
- For DINOv2 ViT-L/g, use the `reg4` variants. For ViT-S/B, registers matter less, but the reg4 weights cost almost nothing extra (timm throughput 820 vs 855 img/s for ViT-S).

### Gaps
- No study found measuring the effect of patch size or resolution on texture-scale regression (e.g. whitecap fraction) specifically.
- Whether CLS + mean-pooled patch tokens (a common DINOv2 linear-eval recipe) outperforms CLS alone for regression was not verified in this session.
- DINOv3's web-model high-resolution training schedule (exact resolutions) was not extracted.

---

## Q4. Robustness to illumination change, glare, low light (twilight) and compression artefacts

### Takeaway
DINOv3 has the best published ImageNet-C robustness (mCE 19.6, against 22.7 for PE-core, 24.1 for DINOv2 and 30.0 for SigLIP 2). DINOv2 embeddings drift less than CLIP's under JPEG, brightness and contrast perturbations, and large ViTs are more robust than ResNets. No benchmark targets twilight, sun glint or low-light sea imagery, so robustness on our twilight views must be measured directly (e.g. score by solar elevation bin).

### Cited Findings
- ImageNet-C linear-probe mCE (lower is better): DINOv3 7B 19.6; PE-core G 22.7; DINOv2 g 24.1; AIMv2 3B 29.5; SigLIP 2 g 30.0; Web-DINO 7B 31.2; EVA-CLIP 18B 33.0 — [DINOv3 paper HTML](https://arxiv.org/html/2508.10104v1)
- DINOv2 paper, ImageNet-C (lower is better): DINOv2 g 28.2, L 31.5, B 42.7, S 54.4; iBOT-L 43.9; OpenCLIP-G 45.3; DINO B/8 56.6; MAE-H 61.4. Robustness scales strongly with DINOv2 size — [DINOv2 paper HTML](https://arxiv.org/html/2304.07193)
- DINOv3-distilled ConvNeXt-T vs supervised IN-22k ConvNeXt-T: ImageNet-R 73.7 vs 45.0 and ObjectNet 52.6 vs 44.5 at 256 — [DINOv3 paper HTML](https://arxiv.org/html/2508.10104v1)
- Liu, Jiang, Hong & Gong (arXiv 2604.14973, 16 Apr 2026), embedding drift under nine perturbations. DINOv2 ViT-L/14 drifts less than CLIP ViT-L/14 under JPEG (DivergenceRadius 0.038 vs 0.084), brightness, contrast, fog, Gaussian noise and glass blur, but more under defocus blur (0.146 vs 0.108). DINOv2-g linear-probe ImageNet accuracy drops: JPEG −2.0, brightness −0.9, contrast −0.2, noise −0.9 points (clean 86.6). ViTs are more robust than ResNets; larger DINOv2 is more robust while larger CLIP is less robust. Overall, foundation models "generally lack robustness to common perturbations" — [arXiv 2604.14973](https://arxiv.org/html/2604.14973)
- DINOv2 and CLIP were benchmarked on ACDC-Night, fog, rain and snow for input monitoring in driving. DINOv2 and DINO variants ran at 35-59 FPS in that setup; specific numbers were not verified (snippet only) — [arXiv 2501.08083](https://arxiv.org/html/2501.08083v3)
- DINOv2's use for low-level lighting restoration was described as "relatively unexplored" in a 2026 ambient-light normalisation paper built on DINOv2 features (snippet only) — [DINOLight, arXiv 2603.12579](https://arxiv.org/pdf/2603.12579)

### Inferences
- ImageNet-C includes brightness, contrast, JPEG and noise corruptions, which are the closest standard proxies for twilight, NDBC JPEG strips and sensor noise. On that proxy DINOv3 ≥ PE-core > DINOv2 > SigLIP 2 / EVA-CLIP > supervised CNNs.
- Twilight, sun glint and buoy-tilted horizons are not covered by any benchmark found. Recommended checks: stratify unseen-buoy error by solar elevation (day / civil twilight / nautical twilight) and by horizon tilt, and compare frozen DINOv3 against the fine-tuned ConvNeXt baseline per stratum.
- Augmentation (brightness/gamma, JPEG quality 30-90, rotation ±10-15° for tilt) during probe training or fine-tuning is likely more important than backbone choice for twilight robustness. This is a judgment, not tested.

### Gaps
- No published evaluation of any of these backbones on night, twilight or low-light outdoor imagery with regression targets.
- No glare or specular-highlight robustness benchmark found for foundation models.
- No DINOv3-specific JPEG or brightness drift numbers (Liu et al. cover only CLIP and DINOv2).

---

## Q5. Licences, weight availability (torch.hub / HF / timm), parameter counts, throughput and memory on one H100/H200; release dates and superseded models

### Takeaway
For FAU research use every model below is usable. The licence differences matter only for any later commercial or ITAR-adjacent work. DINOv2 (Apache-2.0), SigLIP 2 (Apache-2.0 weights), PE (Apache-2.0) and C-RADIOv3/v4 (NVIDIA Open Model License, commercial allowed) are the most permissive. DINOv3 has a custom licence that permits commercial use but bans military/ITAR uses, requires attribution in published research, and gates downloads. RADIOv2.5, Web-SSL, MAE and ConvNeXt V2 weights are non-commercial. On a single H200, frozen-feature extraction for all ~246k views is minutes for ViT-S/B/L. Even ViT-7B extraction is roughly an hour, though fine-tuning 7B is not practical on one shared GPU.

### Cited Findings

**Catalogue (dates, sizes, licences, names)**
- **DINOv3** (Meta; paper 13 Aug 2025; HF Hub 14 Aug 2025; Transformers ≥ 4.56.0 from 29 Aug 2025; timm ≥ 1.0.20 from 17 Sep 2025; ConvNeXt distillation code 20 Nov 2025; CHMv2 canopy model 10 Mar 2026; FINO metadata-guided training branch 12 Jun 2026). Web (LVD-1689M) backbones: ViT-S/16 21M, ViT-S+/16 29M, ViT-B/16 86M, ViT-L/16 300M, ViT-H+/16 840M, ViT-7B/16 6,716M, and ConvNeXt T 29M / S 50M / B 89M / L 198M. Satellite (SAT-493M): ViT-L/16 and ViT-7B/16. torch.hub names like `dinov3_vitb16`, `dinov3_convnext_tiny`; HF names like `facebook/dinov3-vitb16-pretrain-lvd1689m`, `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`, `facebook/dinov3-vitl16-pretrain-sat493m`. Text-aligned dino.txt on ViT-L. Downloads are gated (request access, then emailed URLs) — [DINOv3 repo](https://github.com/facebookresearch/dinov3)
- **DINOv3 License**: a "non-exclusive, worldwide, non-transferable and royalty-free limited license" to use, modify and distribute, not limited to non-commercial use. It prohibits ITAR-subject activities and military/warfare, nuclear, espionage and weapons uses. Published research results must acknowledge use. Derivatives remain under the agreement. The licence has a patent-litigation termination clause, and Meta may modify the terms — [DINOv3 LICENSE.md](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
- timm DINOv3 example: `vit_large_patch16_dinov3.lvd1689m`, 303.1M params, 82.4 GMACs at 256x256, licence "DINOv3 license" — [HF timm card](https://huggingface.co/timm/vit_large_patch16_dinov3.lvd1689m)
- **DINOv2** (Apr 2023; registers paper Sep 2023): Apache-2.0. timm `vit_base_patch14_reg4_dinov2.lvd142m` has 86.6M params and 117.5 GMACs at 518 — [HF timm card](https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m). Param counts from the timm benchmark: S/14 22.06M, B/14 86.58M, L/14 304.37M — [timm benchmark CSV](https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/results/benchmark-infer-amp-nchw-pt240-cu124-rtx4090.csv). ViT-g/14 is 1.1B — [DINOv2 paper](https://arxiv.org/html/2304.07193)
- **SigLIP 2** (Google; 20 Feb 2025): B 86M, L 303M, So400m 400M, g 1B; patch 14 or 16; 224-512 px; NaFlex variants for native aspect ratio — [SigLIP 2 paper](https://arxiv.org/html/2502.14786). HF weights tagged Apache-2.0 (the arXiv page shows CC BY 4.0, which applies to the paper) — [HF siglip2-so400m-patch16-naflex](https://huggingface.co/google/siglip2-so400m-patch16-naflex)
- **Perception Encoder** (Meta; 17 Apr 2025; in timm from 28 May 2025; small and spatial checkpoints 11 Jul 2025): PE-core T/S/B/L/G, PE-lang L/G, PE-spatial G plus distilled T/S/B/L. Apache-2.0 (the separate PLM models are under the FAIR Research License) — [PE repo](https://github.com/facebookresearch/perception_models)
- **RADIO family** (NVIDIA): RADIOv2.5 B/L (Jul 2024), H (Oct 2024), g (Dec 2024), under the non-commercial NSCL. C-RADIO (Dec 2024), C-RADIOv3 B/L/H/g (3 Jun 2025) and C-RADIOv4 SO400M/H (27 Jan 2026; teachers SigLIP2-g, DINOv3-7B, SAM3) are under the NVIDIA Open Model License (commercial) — [RADIO repo](https://github.com/NVlabs/RADIO). C-RADIOv4-H: about 653M params, patch 16, up to 2048 px. ADE20k 55.20 vs C-RADIOv3-H 52.75; kNN ImageNet 86.68 — [C-RADIOv4-H card](https://huggingface.co/nvidia/C-RADIOv4-H)
- **Web-SSL** (Meta; Fan et al., arXiv 2504.01017, Apr 2025): Web-DINO 0.3B-7B (patch 14; 224/378/518), Web-MAE 0.3B-3B. Mostly CC-BY-NC — [Web-SSL repo](https://github.com/facebookresearch/webssl). Weaker than DINOv2-g on dense probes (ADE 42.7 vs 49.5) despite 7B params — [DINOv3 paper](https://arxiv.org/html/2508.10104v1)
- **ConvNeXt V2** (Jan 2023): Atto 3.7M to Huge 660M; T 28.6M (83.0 IN-1k FT; 83.9 at 224 and 85.1 at 384 with IN-22k); B 89M (86.8 / 87.7 with IN-22k). Code MIT, weights CC-BY-NC; repo archived — [ConvNeXt-V2 repo](https://github.com/facebookresearch/ConvNeXt-V2)
- **MAE** (2021-22): CC-BY-NC 4.0; ViT-B/L/H fine-tuned IN-1k 83.6 / 85.9 / 86.9 (H448: 87.8); repo archived 6 Aug 2025 — [MAE repo](https://github.com/facebookresearch/mae). Poor frozen: 12-dataset linear average 78.0 (ViT-H) — [DINOv2 paper](https://arxiv.org/html/2304.07193)
- **I-JEPA** (CVPR 2023): ViT-H/14 224 and ViT-H/16 448 (IN-1k), ViT-H/14 and ViT-g/16 (IN-22k); repo archived 1 Aug 2024; licence in the LICENSE file (not named on the page) — [I-JEPA repo](https://github.com/facebookresearch/ijepa)

**Throughput (timm inference benchmark, file `benchmark-infer-amp-nchw-pt240-cu124-rtx4090.csv`, i.e. AMP on an RTX 4090 per the filename; images/s at the listed resolution and batch)** — [timm benchmark CSV](https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/results/benchmark-infer-amp-nchw-pt240-cu124-rtx4090.csv)
- convnext_tiny 4,797 @224 (4.47 GMACs), 1,644 @384; convnext_base 2,095 @224
- convnextv2_tiny 1,982 @224 (same GMACs as V1, about 2.4x slower)
- efficientnetv2_s 2,605 @288; resnet50 4,218 @224; swin_tiny 2,659 @224; swin_base 1,207 @224; deit3_base 2,583 @224
- vit_small_patch14_dinov2 856 @518 (46.8 GMACs); vit_base_patch14_dinov2 330 @518 (151.7 GMACs); vit_large_patch14_dinov2 108 @518 (507.2 GMACs); reg4 variants within 1-4% of these
- vit_base_patch16_siglip_224 2,439; vit_so400m_patch14_siglip_224 464 (427.7M params); vit_large_patch14_clip_224 658; eva02_base_patch14_224 1,315

### Inferences
- Throughput at our view size (my scaling; ViT cost is roughly linear in token count at these lengths). DINOv2/v3 ViT-B at about 510-650 tokens should run about 2.1-2.7x faster than at 518² (1,369 tokens), i.e. about 700-900 views/s on a 4090. ViT-L: about 230-290 views/s. ConvNeXt-T at 480x270 (2.6x the pixels of 224²): about 1,800 views/s. For ~246k views (41k strips x 6), one frozen pass takes roughly 5 min (ViT-B), 15-20 min (ViT-L) or 2-3 min (ConvNeXt-T) on a 4090; an H200 should be no slower. These are estimates, not measured.
- ViT-7B frozen extraction (my estimate): about 6.7e9 params x 510 tokens ≈ 3.4 TMACs ≈ 6.8 TFLOPs per view. At an assumed ~400 effective bf16 TFLOPS that is about 60 views/s, or about 70 min for all views. Weights take about 13.4 GB in bf16, so this fits easily in 141 GB. Full fine-tuning of 7B with AdamW needs about 16 bytes/param ≈ 107 GB before activations, so it is not practical on one shared H200.
- ViT-L full fine-tune (my estimate): about 304M params x 16 bytes ≈ 5 GB for weights, gradients and Adam states. Activations at about 510 tokens with batch 32-128 in bf16 should fit in 141 GB, with gradient checkpointing as a fallback. ViT-S/B and ConvNeXt-T/B are trivial on an H200.
- Caching frozen patch tokens costs memory. ViT-B (768-d) x 510 tokens x 246k views in fp16 ≈ 193 GB, so store pooled or attentive-probe-ready reduced tokens (e.g. 2x2-pooled) or fp8. A CLS-only cache is about 0.4 GB.
- Superseded or stale: DINOv2 → DINOv3 for dense and robustness (DINOv2 remains the most permissive and best supported); SigLIP → SigLIP 2; RADIOv2.5 → C-RADIOv3 → C-RADIOv4 (also the licence moved to commercial-OK); MAE, I-JEPA and ConvNeXt V2 repos are archived; Web-SSL is outclassed by DINOv3 on dense probes.

### Gaps
- No official H100/H200 throughput or memory benchmarks were found for DINOv3 ViTs or ConvNeXts at non-square ~480x270 inputs; all figures above are scaled from RTX 4090 timm data.
- Exact timm model names for DINOv3 ViT-S/B/H+ and the DINOv3 ConvNeXts were not verified; only `vit_large_patch16_dinov3.lvd1689m` was confirmed.
- PE and EVA-02 parameter counts and licences beyond the PE repo badge were not verified; the I-JEPA licence name was not visible on the repo page.

---

## Q6. Use of these backbones for sea-state / wave / ocean-surface imagery, environmental webcams and geophysical camera regression

### Takeaway
No peer-reviewed study was found that uses DINOv2, DINOv3, SigLIP or other modern foundation-model features for wave height from optical camera images. Published optical wave-gauging and sea-state work uses fine-tuned ImageNet CNNs or ViTs and reports site-specific training needs. The closest analogues are webcam regression with frozen DINOv2 (air quality), foundation-model segmentation for cross-site water level, and DINOv3 for canopy-height regression. This is a genuine gap that the project could fill.

### Cited Findings
- Buscombe, Carini, Harrison, Chickadel & Warrick 2019/2020, "Optical wave gauging using deep neural networks", Coastal Engineering 155:103593. It uses a pretrained CNN plus dense layers for wave height and period from surf-zone imagery (Santa Cruz CA, Duck NC). Inception-ResNetV2 gave the lowest RMS error and MobileNetV2 the worst; inference is under 100 ms per image on CPU. The technique must be trained separately for each site — [USGS record](https://pubs.usgs.gov/publication/70208193); [open copy](https://par.nsf.gov/servlets/purl/10201658)
- Beaufort-scale sea-state classification with fine-tuned DeiT (F1 66%) and EfficientNet (F1 65%). A multimodal ensemble adding anemometer data reached F1 97% (Authorea preprint; snippet only, fetch refused with a 403) — [Authorea preprint](https://www.authorea.com/doi/full/10.22541/au.176340986.61610309/v1)
- Ship-bridge camera Beaufort study comparing ResNet-101, NASNet, MobileNetV2 and ViT-B/32; ViT-B/32 was close to the best (snippet only) — [ResearchGate: A Computer Vision Approach to Estimate the Localized Sea State](https://www.researchgate.net/publication/382065867_A_Computer_Vision_Approach_to_Estimate_the_Localized_Sea_State)
- X-band radar: a ViT on vessel radar images met its operational error target, but use across vessels needed vessel-specific adaptation or multi-vessel training data (TU Delft thesis 2026; snippet only) — [TU Delft repository](https://repository.tudelft.nl/record/uuid:29d1170b-30b0-4e46-b86e-0ac5c259f891). On synthetic X-band data, a ResNet feature extractor gave better OOD accuracy than InceptionV3 or ViT (snippet only) — [Ocean Engineering 2025](https://www.sciencedirect.com/science/article/abs/pii/S0029801825017330). A spectral point transformer beat SWHFormer on sea clutter — [arXiv 2601.04581](https://arxiv.org/abs/2601.04581)
- SwellSight, an open-source GitHub project, uses DINOv2 plus Depth-Anything-V2 and FLUX.1 synthetic data on beach cams. It claims 92% direction accuracy and gives no height accuracy; this is an unverified, non-peer-reviewed listing — [GitHub dinov2 topic listing](https://github.com/topics/dinov2?o=desc&s=updated)
- Webcam air quality: DINOv2 features plus a regressor for AQI, PM2.5 and PM10 on about 12,240 webcam images from India and Nepal (Springer chapter; snippet only) — [Springer](https://link.springer.com/chapter/10.1007/978-981-96-2359-4_4)
- River water level (Hu et al., Water Resources Research 2026): training-free PerSAM segmentation gives a water-pixel ratio, mapped to level by a light regression (polynomial, log or SVR), to "mitigate site-specific data dependency". Full text was not accessible (403), so cross-site numbers were not verified — [WRR 2026](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025WR042686)
- Geophysical regression with DINOv3: CHMv2 global canopy-height maps built on the SAT-493M ViT-L (2026) — [CHMv2 arXiv 2603.06382](https://arxiv.org/pdf/2603.06382); [DINOv3 repo](https://github.com/facebookresearch/dinov3). The DINOv3 Sat 7B model also cut canopy-height MAE (e.g. 2.42 → 2.02 on the São Paulo set) — [DINOv3 paper HTML](https://arxiv.org/html/2508.10104v1)
- Underwater (marine) imagery classification: frozen DINOv3 ViT-B/16 with logistic regression matched or beat a fully supervised ConvNeXt — [arXiv 2604.00313](https://arxiv.org/abs/2604.00313)

### Inferences
- Across optical wave gauging (Buscombe), radar ViTs (TU Delft) and Beaufort classification, the recurring failure mode is site or platform specificity. This matches our unseen-buoy test. It supports trying frozen or LP-FT foundation features, whose OOD advantage is the best-documented property (Kumar et al.; DINOv3 ImageNet-R/C/ObjectNet gains).
- A paper comparing frozen DINOv3 probes, DINOv3-initialised fine-tuning and ImageNet ConvNeXt fine-tuning on NDBC buoy cameras at unseen buoys would appear to be the first of its kind, given what this search found.
- The WRR water-level approach (foundation segmentation, then a physical ratio, then light regression) suggests a hybrid for the shore cameras: whitecap or foam segmentation from DINOv3 dense features, then a physically motivated regression. This is an idea, not evidence.

### Gaps
- No peer-reviewed optical-camera wave-height study using DINOv2, DINOv3, SigLIP, CLIP, PE or RADIO features was found (searched November 2026). The Choi/Kim/Yang and Kamagata 2026 papers already in the project's notes were not re-examined here.
- Full texts for the sea-state ViT ensemble (Authorea) and Hu et al. 2026 (WRR) were blocked (403), so their cross-site results are unverified.
- No snow, visibility or flood webcam study with modern foundation-model features and unseen-camera evaluation was found.
