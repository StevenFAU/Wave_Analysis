# Domain-adaptive self-supervised pretraining, and what the compute allows

Compiled 2026-11-18. Primary sources were fetched unless marked "(search snippet)". Items released June to November 2026 carry **[recent, verified: ...]** and say where they were checked. **Measured** means the source reports the number; **estimate** means I derived it, and the derivation is given. FLOP conventions: torchvision's "GFLOPS" column matches 1 × params × tokens for ViT-B/16 (86.6 M × 197 ≈ 17.1 G plus attention ≈ 17.56), so it counts **multiply-accumulates (MACs)**. All my per-sample estimates below are in MACs; 1 MAC = 2 FLOPs.

---

## Q1. What does published evidence say about gains from continued SSL pretraining on in-domain unlabelled images or video over off-the-shelf DINOv2/v3 or V-JEPA 2, especially out of distribution, and at what compute?

### Takeaway
Continuing SSL from a strong generic checkpoint pays off **when the domain is far from web photos**: microscopy F1 +17.8, chest X-ray AUROC +5.2, satellite linear probe +8.5 points. It pays **little or nothing, and sometimes hurts, when the generic model already covers the domain**: wildlife −1.4 to −3.1, lung ultrasound null. Out-of-distribution gains are smaller and not guaranteed: FMoW worst-group accuracy +3.4, while one external chest X-ray test fell from 60.2 to 57.7 (forgetting). Two things about cost:
- **Data.** About 100k images was the threshold below which continued pretraining did not beat the open weights in one controlled study.
- **Compute.** Published runs cost roughly 10 to 1,300 GPU-hours for ViT-B/L. That is modest next to the original pretraining (DINOv2 ViT-g: 22,000 A100-hours).

### Cited Findings

**FINO** (Meta/DINOv3 team) is the best controlled evidence: DINOv3 ViT-L continued on domain data.
- **[recent, verified: arXiv abs page, v1 3 Jun 2026, v2 9 Nov 2026]** FINO starts from the open DINOv3 ViT-L/16 checkpoint (lvd1689m; 306 M parameters, distilled from the 7B). It continues DINO + iBOT + SIGReg self-supervision on unlabelled domain images and adds weak "metadata guidance" (e.g. plate, country, camera trap). No task labels are used. — [arXiv:2606.05107](https://arxiv.org/abs/2606.05107); [HTML](https://arxiv.org/html/2606.05107)
- Unlabelled set sizes — [FINO HTML](https://arxiv.org/html/2606.05107):

  | Domain | Images |
  |---|---|
  | HPA microscopy | 94,270 (1,239 plates) |
  | FMoW satellite | 76,863 (146 countries) |
  | iWildCam | 129,809 (243 traps) |
  | MIMIC-CXR | 368,960 |

- Frozen-probe results (Table 1). Three of the metrics are OOD: iWildCam OOD F1, FMoW OOD worst-group accuracy, and HPA OOD F1. — [FINO HTML](https://arxiv.org/html/2606.05107)

  | Method | HPA F1 | iWildCam OOD F1 | FMoW OOD WGA | MIMIC AUROC |
  |---|---|---|---|---|
  | Frozen DINOv3 | 42.2 | 51.4 | 45.0 | 75.9 |
  | Continued DINO-only SSL | 60.0 | 50.0 | 48.4 | 81.1 |
  | Supervised fine-tuning | 52.6 | 43.4 | 40.5 | 80.7 |
  | FINO | 61.2 | 52.9 ± 0.8 | 53.3 ± 0.8 | 81.7 ± 0.1 |

- Supervised fine-tuning **hurt OOD** relative to the frozen model: iWildCam 43.4 vs 51.4, FMoW 40.5 vs 45.0. The authors say fine-tuning "can erase pretrained generality". — [FINO HTML](https://arxiv.org/html/2606.05107)
- Training recipe — [FINO HTML](https://arxiv.org/html/2606.05107):
  - Two stages: new heads and patch embedding train with the transformer blocks frozen, then the full backbone is unfrozen with a 1k-iteration linear warm-up.
  - Removing the frozen warm-up at a fixed budget dropped FMoW WGA from 53.3 to 47.7.
  - The authors recommend at least 50 epochs, keeping the best-validation checkpoint; iWildCam was capped at about 12 epochs.
- Compute (**measured**, all on H100 or H200) — [FINO HTML](https://arxiv.org/html/2606.05107):
  - Final runs: FMoW 160.8, iWildCam 10.7, MIMIC-CXR 920 (at 512²), HPA 3,855.5 (at 768²) GPU-hours.
  - Development including ablations: 34,673 GPU-hours.
  - Speed: 0.76 s per iteration at batch 64 on one H100, vs 0.74 s for pure SSL.

**ExPLoRA** (ICML 2025): LoRA plus unfreezing blocks, continuing DINOv2 or MAE pretraining.
- Venue and method: ICML 2025, PMLR v267. Starts from DINOv2 or MAE ViT-L, fully unfreezes 1–2 blocks (standard DINOv2 config: the last block), puts LoRA r = 64 on Q and V of all other blocks, unfreezes norms, and continues the source model's own SSL objective. 18.7 M trainable parameters during pretraining (about 6 % of the ViT); downstream LoRA r = 8 (0.8 M). — [PMLR](https://proceedings.mlr.press/v267/khanna25a.html); [arXiv:2406.10973 HTML](https://arxiv.org/html/2406.10973)
- Results vs DINOv2 — [arXiv:2406.10973](https://arxiv.org/html/2406.10973):

  | Benchmark | ExPLoRA | DINOv2 / prior |
  |---|---|---|
  | fMoW-RGB linear probe, last 4 blocks | 77.48 % | 69.00 % (DINOv2) |
  | fMoW-RGB with LoRA fine-tuning | 79.28 % | 77.80 % (best fully pretrained, ScaleMAE) |
  | RESISC-45 linear probe | 97.32 % | 96.34 % |
  | SpaceNet-v1 mIoU | 76.34 | 76.21 (negligible) |

- WILDS (OOD benchmarks) — [arXiv:2406.10973](https://arxiv.org/html/2406.10973):
  - Camelyon17: linear probe 94.41 vs 93.27; LoRA 94.21 vs 92.97.
  - GlobalWheat: 82.7 / 54.5 AP vs 82.3 / 52.1.
  - **iWildCam linear probe got worse: 62.95 vs 66.04.** The authors attribute this to a small domain gap, since ImageNet contains many animals, and to possible overfitting.
- Compute (**measured**) — [arXiv:2406.10973 appendix](https://arxiv.org/html/2406.10973):
  - Hardware: a single RTX 6000 Ada, or 4 × RTX A4000.
  - Pretraining cost: fMoW-RGB 96, fMoW-Sentinel 320, fMoW-Temporal 100 GPU-hours.
  - DINOv2 schedule: 50k / 100k / 150k / 200k iterations took 24 / 48 / 72 / 96 GPU-hours, at effective batch 1,024 (32 per step × 32 accumulation), lr 2e-3, AdamW without weight decay, 224² global and 98² local crops.
  - Baselines: 100 GPU-hours vs 960 for full SatMAE/ScaleMAE pretraining. DINOv2 trained from scratch on fMoW-RGB for 1,200 GPU-hours reached only 54.29 % linear probe.
  - Unlabelled set sizes: fMoW-RGB 363.6k, fMoW-Sentinel 712.9k, Camelyon17 302.4k, iWildCam 129.8k images.
- Stated limits — [arXiv:2406.10973](https://arxiv.org/html/2406.10973):
  - Gains are "more modest" on datasets under 50k samples (overfitting).
  - Beyond 100k iterations, extra pretraining mainly speeds early fine-tuning, not the final ceiling.
  - The optimal iteration count depends on batch, lr, rank and unfrozen blocks.
  - High-rank LoRA on all matrices (including MLP) hurt.
  - No forgetting analysis.

**RAD-DINO and MedImageInsight**: a controlled data-scaling study of continued pretraining on chest X-rays.
- Setup: continued pretraining of RAD-DINO (DINOv2-style ViT-B) and MedImageInsight from open weights, on nested subsets of 30k, 50k, 100k, 1M and 3.5M chest X-rays from one institution. — [arXiv:2509.12818](https://arxiv.org/html/2509.12818v1)
- Compute (**measured**): 4 nodes × 8 H100, batch 1,280. Wall-clock 0.33 / 0.58 / 1.17 / 11.67 / 40.83 h for the five subsets, i.e. about **11 / 19 / 37 / 373 / 1,307 H100-hours**. — [arXiv:2509.12818](https://arxiv.org/html/2509.12818v1)
- When gains appear: RAD-DINO beat its open weights only from 100k images. MedImageInsight beat its own open weights on some tasks from 30k–50k. — [arXiv:2509.12818](https://arxiv.org/html/2509.12818v1)
- In-domain results at 3.5M — [arXiv:2509.12818](https://arxiv.org/html/2509.12818v1):
  - Tubes AUPRC: 90.6 → 92.1.
  - Lines-and-tubes segmentation Hausdorff: 21.4 → 13.1 (lower is better).
- **External OOD set (VinDR): RAD-DINO showed a "reversed scaling trend" and "signs of catastrophic forgetting", falling from 60.2 (open weights) to 57.7 at 3.5M.** — [arXiv:2509.12818](https://arxiv.org/html/2509.12818v1)

**EchoJEPA**: V-JEPA 2 continued on echocardiography video.
- Initialisation and data: EchoJEPA-L loads V-JEPA 2 ViT-L encoder weights directly; the predictor and mask tokens are randomly initialised. It is pretrained on 525,328 MIMIC-IV-Echo clips. EchoJEPA-G (1.1 B) uses 18.1 M proprietary echo videos; its initialisation is not stated. — [arXiv:2602.02603](https://arxiv.org/html/2602.02603)
- Compute for EchoJEPA-L (**measured** setup; wall-clock not reported) — [arXiv:2602.02603](https://arxiv.org/html/2602.02603):
  - Hardware and batch: 8 × H100, global batch 1,024, 16 frames at 8 fps, 224².
  - Schedule: 12k-update warm-up and lr 1.75e-4. The text says 60k + 24k cooldown updates; Table 8 says 240 + 80 epochs. Either way about 86 M clips (1.4 B frames) were seen.
- Results — [arXiv:2602.02603](https://arxiv.org/html/2602.02603):
  - LVEF MAE, Toronto: EchoJEPA-L 5.97 vs EchoMAE-L 8.15 (same data and compute), against EchoJEPA-G 4.26 and EchoPrime 5.33.
  - Pediatric zero-shot LVEF MAE: G 4.32, L 6.31.
  - Under acoustic perturbations G degraded +2.3 % vs EchoPrime +16.8 %.
  - **No frozen off-the-shelf V-JEPA 2 baseline is reported**, so the gain from continued pretraining over V-JEPA 2 itself is not isolated.
- Domain changes to the V-JEPA 2 recipe — [arXiv:2602.02603](https://arxiv.org/html/2602.02603):
  - fps raised from 4 to 24 (G).
  - Aspect-ratio augmentation narrowed from (0.75, 1.35) to (0.9, 1.1).
  - Crop scale narrowed from (0.3, 1.0) to (0.5, 1.0).
- **[recent, verified: arXiv abs page, 2 Oct 2026]** Counter-evidence: on lung-ultrasound TB screening, the pre-specified EchoJEPA-L vs generic V-JEPA2-L contrast was −0.16 percentage points (p = 0.926). Seventeen encoders spanned only 2.50 pp against a measurement resolution of 2.71 pp. The authors conclude that "the limiting factor is the labelled cohort rather than the encoder". Caveat: video encoders received replicated still images. — [arXiv:2610.03290](https://arxiv.org/abs/2610.03290)

**V-JEPA 2**: what its own pretraining shows about data.
- Data: VideoMix22M (about 22 M samples; SSv2 168k, Kinetics 733k, HowTo100M 1.1M, YT-Temporal-1B 19M curated, ImageNet 1M). Scaling from 2M to 22M videos gave +1.0 point on the six-task average (ViT-L). Curating YT-1B gave +1.4 over uncurated. — [arXiv:2506.09985](https://arxiv.org/html/2506.09985)
- Schedule: 252k iterations at global batch 3,072, 16 frames at 256²; cooldown 12k steps at 64 frames and 256–512 crops. — [arXiv:2506.09985](https://arxiv.org/html/2506.09985)
- Full-resolution ViT-g training at 64×384² is put at "roughly 60 GPU-years"; progressive resolution saves "8.4×". — [arXiv:2506.09985](https://arxiv.org/html/2506.09985)

**Other evidence**
- VideoMAE (NeurIPS 2022): VideoMAE trains from as few as 3.5k videos, and "data quality is more important than data quantity" when pretraining and target domains differ. Pretraining on the target video set beat transferring across video datasets. (search snippet) — [arXiv:2203.12602](https://arxiv.org/abs/2203.12602)
- Hierarchical PreTraining (WACV 2022): starting SSL from an already pretrained model instead of random weights converged up to 80× faster across 16 datasets and was more robust to augmentation and data amount. (search snippet; 80× is an upper bound) — [arXiv:2103.12718](https://arxiv.org/abs/2103.12718)
- GFM (ICCV 2023): continual pretraining of a geospatial model took 93.3 V100 GPU-hours vs 768 for SatMAE. (search snippet of the supplement) — [ICCV supplement](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Mendieta_Towards_Geospatial_Foundation_ICCV_2023_supplemental.pdf)
- Radio astronomy (2026): two-stage continued pretraining beat its MAE start everywhere but beat the strongest DINOv2 baseline only on some datasets under linear probing. (search snippet only) — [arXiv:2603.29660](https://arxiv.org/html/2603.29660v1)
- Original pretraining scale for reference (search snippet): DINOv2 ViT-g took 22,000 A100-hours; ImageNet fine-tuning 1,000 GPU-hours. The model card gives distillation at 4,500 (S), 5,300 (B) and 8,000 (L) A100-hours. — [DINOv2 paper](https://arxiv.org/pdf/2304.07193)

### Inferences
- Across FINO, ExPLoRA and RAD-DINO the pattern is consistent: the further the domain is from web photos, the larger the gain. Microscopy, chest X-ray and multispectral gain a lot; wildlife photos gain nothing or lose.
  - Shore-camera ocean scenes are ordinary RGB outdoor photos, which LVD-1689M and YT-1B almost certainly cover.
  - NDBC buoy-cam strips (low-resolution, horizon-tilted, often hazy or dark) are probably further out.
  - Expect a **small** gain for shore stills, a possibly larger one for buoy strips, and no guarantee for either.
  - This is an inference: no study of ocean-surface imagery was found.
- The main OOD risk in the evidence is **forgetting under long continued runs** (RAD-DINO on VinDR). FINO's two-stage warm-up and ExPLoRA's mostly-frozen LoRA are the two published mitigations.
  - For this project, OOD means **unseen buoys or cameras**. The held-out-station split already in use is the right test, and every continued-pretraining arm should be scored on it against the frozen-DINOv3 arm.
- In FINO, continued DINO-only SSL without labels **beat supervised fine-tuning OOD** on three of four domains. That supports keeping the backbone close to frozen (LP or LP-FT, as the project does) after any domain pretraining.
- The data threshold (about 100k images for RAD-DINO; ExPLoRA "modest" under 50k) is comfortably met: about 246k buoy views and hundreds of thousands of shore stills.
  - For video, 10k–100k clips is below EchoJEPA-L's 525k but in the range where VideoMAE reports in-domain pretraining working.
- The lung-ultrasound null result restates what the project's own notes already find: with 67 training buoys, the labelled set, not the encoder, may limit accuracy. Continued pretraining should be judged against seed-to-seed spread and the buoy-bootstrap intervals.
- **Metadata guidance (FINO)** maps naturally onto label-free metadata the project already has: station ID, camera view index, time of day and season, maybe platform. Using wave labels as "metadata" would make it supervised.

### Gaps
- No published continued-SSL study on ocean, sea-surface or coastal camera imagery or video was found, so the size of the gain for this domain is unknown.
- No paper isolates the gain of continued V-JEPA 2 pretraining over frozen off-the-shelf V-JEPA 2 on video. EchoJEPA omits that baseline; the lung-ultrasound study fed still images.
- No ExPLoRA-style (LoRA SSL) study for V-JEPA / video JEPA was found.
- No controlled continued-DINOv2/v3 results for plant or underwater imagery were found. Searches returned fine-tuning-stage adaptation (DiveSeg, AAAI 2026) and Agri-FM+ (CVPRW 2025), with no before/after numbers verified.
- The original RAD-DINO paper's training compute was not fetched.
- GFM's GeoPile size was not verified.

---

## Q2. Is ExPLoRA-style (LoRA + unfreezing a block) extended pretraining feasible on 1–2 partially free H200s for DINOv3 ViT-B/L and V-JEPA 2 ViT-L, and in how many GPU-hours? What do the official codebases support?

### Takeaway
Yes, for images. ExPLoRA's ViT-L runs fit on a single 48 GB workstation GPU or 4 × 16 GB A4000s in 96–320 GPU-hours. FINO's full-unfreeze DINOv3 ViT-L runs cost 11–920 H100/H200-hours per domain. A one-GPU ExPLoRA-style continuation of DINOv3 ViT-B/L on ~250k–1M images is an estimated **30–150 dedicated H200-hours** (×2–3 on a busy GPU).

For V-JEPA 2 ViT-L, the official code runs locally on named devices and EchoJEPA shows continuation from the released encoder works. A 10k–100k-clip continuation of ~10–20 M clip-samples is an estimated **30–80 dedicated H200-hours**; EchoJEPA-L's scale (86 M clip-samples) would be ~250–400.

Neither official repo documents a continued-pretraining or LoRA mode. DINOv3's launcher is SLURM/submitit only, so on a Docker-only host the launch needs adapting.

### Cited Findings

**DINOv3 official code**
- Backbones and parameter counts — [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3):
  - ViT-S/16 21 M, ViT-S+ 29 M, ViT-B/16 86 M, ViT-L/16 300 M, ViT-H+/16 840 M, ViT-7B/16 6,716 M.
  - ConvNeXt T/S/B/L 29 / 50 / 89 / 198 M.
  - All on LVD-1689M; plus SAT-493M ViT-L and 7B.
  - Licence: the DINOv3 License.
- Training launcher: `python -m dinov3.run.submit dinov3/train/train.py --nodes ... --config-file ...` (submitit/SLURM). The README gives no local-training instructions and no general continued-pretraining instructions. — [dinov3 README](https://github.com/facebookresearch/dinov3)
- Reference ViT-L recipe: `vitl_im1k_lin834.yaml` on 4 × 8 H100-80GB (32 GPUs) takes "approximately 14 hours" (≈ 448 H100-hours) and reaches about 82.0 % k-NN and 83.5 % linear on ImageNet-1k. — [dinov3 README](https://github.com/facebookresearch/dinov3)
- That config — [config](https://raw.githubusercontent.com/facebookresearch/dinov3/main/dinov3/configs/train/vitl_im1k_lin834.yaml):
  - Optimisation: batch_size_per_gpu 64, 100 epochs × OFFICIAL_EPOCH_LENGTH 1,250, lr 1e-3, layer-wise decay 0.9, drop-path 0.3.
  - Crops: 2 global at 224 and 8 local at 96.
  - Precision: bf16 params with fp32 reduce; FP8 off.
  - Checkpoint fields present: `student.pretrained_weights`, `student.resume_from_teacher_chkpt`, `distillation.checkpoint_path`.
- ViT-7B: three stages (pretrain, Gram anchoring, high-res adaptation), each on 32 nodes / 256 GPUs. — [dinov3 README](https://github.com/facebookresearch/dinov3)
- Distillation (repo dates): ConvNeXt distillation code and configs released 2025-11-20; a `multi_distillation_test.yaml` runs on 1 node. — [dinov3 README](https://github.com/facebookresearch/dinov3)
- **[recent, verified: repo README news 2026-06-12, and the arXiv paper above]** FINO metadata-guided domain-adaptation code and recipes (FMoW, HPA-WholeHR) are on the `FINO` branch of the DINOv3 repo. — [dinov3 README](https://github.com/facebookresearch/dinov3)

**V-JEPA 2 official code**
- Model sizes — [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2):
  - V-JEPA 2 (2025-06-25): ViT-L/16 300 M, ViT-H/16 600 M, ViT-g/16 1 B (256 and 384).
  - **V-JEPA 2.1 (repo banner 2026-03-16):** ViT-B/16 80 M, L 300 M, g 1 B, G 2 B, all at 384.
- Pretraining — [vjepa2 README](https://github.com/facebookresearch/vjepa2):
  - Config `configs/train/vitl16/pretrain-256px-16f.yaml`.
  - Local: `python -m app.main --fname <cfg> --devices cuda:0`.
  - Distributed: `python -m app.main_distributed` (SLURM flags `--account`/`--qos`).
  - No documented resume or fine-from-checkpoint flag; staged runs (cooldown, action post-training) use separate configs.
- The ViT-L config — [config](https://raw.githubusercontent.com/facebookresearch/vjepa2/main/configs/train/vitl16/pretrain-256px-16f.yaml):
  - Hardware: nodes 16 × tasks_per_node 8 (128 GPUs), batch_size 24 per GPU (global 3,072).
  - Clips: 16 frames (dataset_fpcs), crop 256, tubelet 2, fps 4.
  - Model: `use_activation_checkpointing: true`, bf16, SDPA; predictor depth 12, embed 384.
  - Optimisation: lr 5.25e-4, EMA 0.99925; `load_checkpoint: true`, `read_checkpoint: null`.
  - Data mix: K710 0.335, SSv2 0.100, HowTo 0.565.
- Paper schedule: global batch 3,072, warm-up 12k, 228k constant, 12k cooldown at 64 frames; abbreviated recipe 90k steps. The GPU count and GPU-hours are not given in the text read. — [arXiv:2506.09985](https://arxiv.org/html/2506.09985)
- Attentive probe: 4 transformer blocks, 16 heads, with a final cross-attention block and a learnable query, on the frozen encoder. Eval configs `configs/eval/vitl/{ssv2,diving48,ek100}.yaml`; README: "Change # nodes and local batch size as needed". — [vjepa2 README](https://github.com/facebookresearch/vjepa2); [arXiv:2506.09985](https://arxiv.org/html/2506.09985)
- Original V-JEPA (2024) zoo: ViT-L/16 and ViT-H/16 at 224 (90k iterations, batch 3,072, VideoMix2M) and ViT-H/16-384 (batch 2,400). **No ViT-S or ViT-B checkpoint.** — [facebookresearch/jepa](https://github.com/facebookresearch/jepa)

**Continued V-JEPA precedent (EchoJEPA-L)**
- Released V-JEPA 2 ViT-L encoder plus fresh predictor and mask tokens; 8 × H100; global batch 1,024; ~72–84k updates; lr 1.75e-4; EMA 0.99925. Hyperparameter changes were confined to fps and crop/aspect augmentation. — [arXiv:2602.02603](https://arxiv.org/html/2602.02603)

**Distilling a big frozen teacher into a small student**
- Proteus (ICLR 2025): DINOv2-g/14 distilled into ViT-L on ImageNet-1K only (no original data) matched DINOv2-L/14 (trained on 142 M images) across 19 benchmarks, at "ImageNet-level costs". (search snippet) — [arXiv:2407.10366](https://arxiv.org/html/2407.10366v2)
- DINOv3's own small models (S, S+, B, L, H+, ConvNeXt) are distilled from the 7B, and the ConvNeXt distillation code is public. — [dinov3 README](https://github.com/facebookresearch/dinov3)

**Measured on the project's shared lab H200 server (shared)**
- Feature extraction for all four arms of the WebCOOS shore-video experiment takes about **1.5 s per clip**, decoding included; the full sample is about 1.5 h of GPU time. Jennette clips are about 290 MB each (2592×1944). — local note `~/Projects/Wave_Analysis/docs/experiments/webcoos_video.md` (Progress, 2026-11-18)

### Inferences
**Image GPU-hour estimate (DINOv3 ViT-B/L, ExPLoRA-style)**
- Anchors:
  - ExPLoRA DINOv2 ViT-L: 96 GPU-hours for 200k iterations on an RTX 6000 Ada-class GPU.
  - FINO ViT-L, full unfreeze: 161 GPU-hours (FMoW, 77k images) to 920 (MIMIC, 369k at 512²) on H100/H200.
  - DINOv3 ImageNet-1k ViT-L recipe: ≈ 448 H100-hours.
- An H200 has roughly 2–3× the dense BF16 of a workstation card, and LoRA saves weight-gradient and optimizer work.
- **Estimate: 30–150 dedicated H200-hours** for an ExPLoRA-style DINOv3 ViT-L continuation on ~250k–1M images at 224². ViT-B costs roughly 1/3 (61.6 vs 17.6 GMACs per 224 image). Full unfreeze, FINO-style, is ~1.5–3× that.
- On a GPU already 70–99 % busy, budget 2–3× wall-clock. This is not measured; time-slicing with other processes is assumed.

**Throughput anchor from the DINOv3 ViT-L recipe (estimate)**
- 125,000 iterations × 2,048 = 256 M multi-crop samples in 448 H100-hours ≈ **159 samples/s per H100**.
- Per-sample work: about 12 ViT-L-224 forward-equivalents (student forward + backward on 2 global + 8 local crops, teacher forward on 2 globals) × 61.6 GMAC ≈ 0.75 TMAC ≈ 1.5 TFLOP.
- That implies ≈ 240 TFLOP/s per H100, about 24 % of dense BF16 (989).
- FINO's measured 64 / 0.76 s ≈ **84 images/s per H100** at higher resolution (336–768) is consistent.

**Video GPU-hour estimate (V-JEPA 2 ViT-L continuation)**
- Per-clip compute at 16×256² (2,048 tokens):
  - Encoder forward ≈ 0.304 B × 2,048 + 2 × 24 × 2,048² × 1,024 ≈ 0.83 TMAC (1.65 TFLOP).
  - A training step covers the EMA target forward on all tokens, the context encoder forward + backward on the ~10–30 % unmasked tokens, the predictor (~22 M-class) on all tokens, and +33 % for activation recomputation. That is ≈ 3–4 TFLOP per clip-sample.
- At ~300 TFLOP/s effective (30 % of H200 dense BF16) that is **≈ 80–100 clip-samples/s per dedicated H200**.
- Budgets:
  - 10–20 M clip-samples (e.g. 20k steps × batch 512–1,024, or 100–200 epochs of 100k clips): **≈ 30–80 H200-hours**.
  - EchoJEPA-L scale (≈ 86 M): **≈ 250–400**.
- Video decode is the practical bottleneck: about 1.5 s per clip measured end to end for feature extraction. Pre-cut 16-frame 256² clips to disk once (NVDEC or CPU), rather than decoding 2,592×1,944 or 2,688×1,520 H.264 every epoch.

**Data movement constraint (computed from the stated 2 Mbit/s)**
- At 2 Mbit/s: 0.25 MB/s = 0.9 GB/h = 21.6 GB/day.
- 260 GB of stills therefore take ≈ 12 days to upload.
- 100k raw Jennette-size clips (290 MB each) would be ≈ 29 TB, more than the 28 TB data disk.
- So video must be fetched directly on the lab server from the archive and reduced on the fly: download, cut 16-frame 256² clips at a few hundred KB each, delete the raw file. 100k such clips would then be ≈ 10–30 GB.

**LoRA for V-JEPA**
- PEFT (Hugging Face `peft`) can wrap the encoder's qkv Linear layers. The JEPA loss is unchanged and the EMA target is the (LoRA-merged) encoder.
- This is unpublished for JEPA (see Gaps), so treat it as an experiment, not a recipe. EchoJEPA's full continuation is the safer published template.

**Distillation path**
- A DINOv3 ViT-L (or 7B) teacher distilled into the ConvNeXt-T student on the 246k buoy views and shore stills keeps the deployed model small.
- The official ConvNeXt distillation code makes this the most direct "domain-adaptive" route for the current production backbone. Cost: about the student's training cost plus one teacher forward per crop. Teacher features can be cached; see Q4 budgets.

**Launch on a Docker-only host**
- V-JEPA 2's `app.main --devices cuda:X cuda:Y` covers a single node.
- DINOv3 needs `torchrun` (or a direct call of `dinov3/train/train.py` in place of submitit). That is an adaptation, not documented.

### Gaps
- No published ExPLoRA-style or LoRA SSL results for V-JEPA / V-JEPA 2, or for DINOv3 specifically (ExPLoRA used DINOv2 and MAE).
- V-JEPA 2 pretraining GPU count, GPU type and GPU-hours are not in the text read; Figure 5's GPU-day values are in the figure and were not extracted.
- EchoJEPA-L wall-clock time was not reported.
- ExPLoRA's "200k iterations" with batch 1,024 via 32 accumulation steps is ambiguous: optimizer steps (≈ 205 M crops) or micro-steps (≈ 6.4 M). The GPU-hours are the reliable anchor.
- Throughput on a GPU shared with other jobs at 70–99 % utilisation has not been measured for training. A 15-minute timing run of each recipe on the lab server would replace the 2–3× guess.
- FINO's per-domain iteration counts and batch sizes are not given in the parts read.

---

## Q3. What are the H200/A100/B200 specs, and what model sizes and training regimes fit in 16–117 GB of free memory?

### Takeaway
- **H200 SXM:** 141 GB HBM3e at 4.8 TB/s, ≈ 989 TFLOPS dense BF16 (1,979 with sparsity).
- **A100:** 40/80 GB at 1.6–2.0 TB/s, 312 TFLOPS dense BF16.
- **B200** (from DGX/HGX totals): ≈ 180 GB per GPU, ≈ 2.25 PFLOPS dense BF16 per GPU.

In 16–117 GB:
- **Frozen inference fits everything up to DINOv3 7B** (13.4 GB of bf16 weights).
- **LoRA/ExPLoRA fits ViT-L and V-JEPA 2 ViT-L/g** with activation checkpointing and gradient accumulation.
- **Full fine-tuning fits ViT-L easily and ViT-g (1 B, ≈ 16 GB of states) on one GPU with ≥ 40–80 GB free.**
- **The 7B needs about 107 GB of optimizer states alone**, so full fine-tuning needs FSDP/ZeRO-3 across ≥ 4 partially free GPUs, or LoRA.

### Cited Findings

**NVIDIA H200 page** — [NVIDIA H200](https://www.nvidia.com/en-us/data-center/h200/). Footnote ² on the tensor rows reads "With sparsity". Figures are "Preliminary specifications".

| | H200 SXM | H200 NVL |
|---|---|---|
| FP64 / FP64 Tensor | 34 / 67 TFLOPS | 30 / 60 |
| FP32 | 67 | 60 |
| TF32² | 989 | 835 |
| BF16/FP16² | 1,979 | 1,671 |
| FP8/INT8² | 3,958 | 3,341 |
| Memory | 141 GB HBM3e, 4.8 TB/s | 141 GB, 4.8 TB/s |
| TDP | up to 700 W | up to 600 W |
| NVLink | 900 GB/s | 900 GB/s (2- or 4-way bridge) |
| PCIe | Gen5 128 GB/s | Gen5 128 GB/s |
| MIG | up to 7 × 18 GB | up to 7 × 16.5 GB |

- An independent tracker gives FP8 dense at 1,979 TFLOPS, i.e. half the sparse headline. — [SemiAnalysis InferenceX](https://inferencex.semianalysis.com/chips/h200) (search snippet)

**NVIDIA A100 datasheet** (PCIe 40/80 GB, SXM 40/80 GB). The starred figure is "With sparsity". — [A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)
- Compute: FP64 9.7, FP64 Tensor 19.5, FP32 19.5, TF32 156 | 312*, BF16 312 | 624*, FP16 312 | 624*, INT8 624 | 1,248* TOPS.
- Memory:

  | Variant | Memory | Bandwidth | TDP |
  |---|---|---|---|
  | 40 GB PCIe | HBM2 | 1,555 GB/s | 250 W |
  | 80 GB PCIe | HBM2e | 1,935 GB/s | 300 W |
  | 40 GB SXM | HBM2 | 1,555 GB/s | 400 W |
  | 80 GB SXM | HBM2e | 2,039 GB/s | 400 W |

- Interconnect and MIG: NVLink 600 GB/s; MIG up to 7 × 5 GB (40 GB) or 7 × 10 GB (80 GB).

**NVIDIA DGX B200** — [NVIDIA DGX B200](https://www.nvidia.com/en-eu/data-center/dgx-b200.md)
- 8 Blackwell GPUs, 1,440 GB total GPU memory, 64 TB/s HBM3e.
- FP8 72 PFLOPS ("Specification shown in sparse. Dense performance is ½ sparse spec shown"); FP4 144 PFLOPS (sparse); NVLink 14.4 TB/s aggregate.
- 2 × Xeon Platinum 8570 (112 cores), 2 TB RAM (configurable to 4 TB), ~14.3 kW.

**NVIDIA HGX B200** (8 GPUs). Footnote: sparse, dense = ½. — [NVIDIA HGX](https://www.nvidia.com/en-us/data-center/hgx/)
- FP16/BF16 36 PFLOPS, TF32 18, FP8/FP6 72, FP4 144 sparse / 72 dense, FP64 296 TFLOPS.
- 1.4 TB memory; NVLink 1.8 TB/s per GPU.

**Memory rules**
- ZeRO: mixed-precision Adam holds fp16 params (2 B) + fp16 grads (2 B) + fp32 params, momentum and variance (12 B) = **16 bytes/parameter**. Their 7.5 B example needs 120 GB per device under plain data parallelism, and 31.4 / 16.6 / 1.9 GB per device with ZeRO stage 1 / 2 / 3 on 64 devices. Stage 3 divides model-state memory by the device count. — [ZeRO, arXiv:1910.02054](https://ar5iv.labs.arxiv.org/html/1910.02054)
- Activation checkpointing cuts activation memory by about the square root of the total, at about 33 % extra compute. — [ZeRO](https://ar5iv.labs.arxiv.org/html/1910.02054)
- Hugging Face's accounting gives 18 bytes/parameter for mixed-precision AdamW: 6 for weights (fp16 + fp32 master), 8 for Adam, 4 for fp32 gradients. Activations are extra and scale with batch × sequence × depth × hidden. 8-bit Adam cuts the optimizer to 2 bytes/parameter. — [HF model memory anatomy](https://huggingface.co/docs/transformers/model_memory_anatomy)
- The V-JEPA 2 ViT-L pretraining config runs 24 clips (16×256²) per GPU with activation checkpointing in bf16. The GPU memory size is not stated. — [vjepa2 config](https://raw.githubusercontent.com/facebookresearch/vjepa2/main/configs/train/vitl16/pretrain-256px-16f.yaml)
- ExPLoRA's ViT-L extended pretraining (micro-batch 32 with 32-step accumulation) ran on a single RTX 6000 Ada or 4 × RTX A4000. — [arXiv:2406.10973](https://arxiv.org/html/2406.10973)

### Inferences

**Derived per-GPU figures**
- H200 SXM dense: BF16 ≈ 989 TFLOPS, FP8 ≈ 1,979, TF32 ≈ 494 (half the sparse headline, per NVIDIA's footnote convention).
- B200 per GPU (system totals ÷ 8): ≈ 180 GB, ≈ 8 TB/s, BF16 dense ≈ 2.25 PFLOPS (36 / 2 / 8), FP8 dense ≈ 4.5 PFLOPS.
- Dense BF16 ratio, H200 : A100 : B200 ≈ 3.2 : 1 : 7.2.
- The H200 has the same Hopper compute as the H100 with more and faster memory, so H100-hour figures from FINO, RAD-DINO and DINOv3 carry over roughly 1:1 for compute-bound jobs. The H100 itself was not fetched; it is assumed to share H200 compute.

**Model-state memory, excluding activations** (estimates: params × bytes; sizes from the DINOv3 and V-JEPA 2 repos and torchvision)

| Model | Params | bf16 weights (frozen inference) | Full FT, 16 B/param | ExPLoRA-style (bf16 base + ~6 % trainable × 16 B) |
|---|---|---|---|---|
| ResNet-18 / ConvNeXt-T / ViT-S | 11.7 / 28.6 / 21 M | 0.02–0.06 GB | 0.2–0.5 GB | — |
| ViT-B (DINOv3) | 86 M | 0.17 GB | 1.4 GB | ≈ 0.25 GB |
| ViT-L (DINOv3, V-JEPA 2) | 300 M | 0.6 GB | 4.8 GB | ≈ 0.9 GB |
| V-JEPA 2 ViT-g | 1 B | 2 GB | 16 GB | ≈ 3 GB |
| V-JEPA 2.1 ViT-G | 2 B | 4 GB | 32 GB | ≈ 6 GB |
| DINOv3 ViT-7B | 6.716 B | 13.4 GB | 107 GB | ≈ 20 GB |

- SSL adds an EMA teacher or target copy: ≈ +2–4 B/param.
- For video, activations dominate. A ViT-L layer at 2,048 tokens × 1,024 dims in bf16 stores ~70 MB of activations per clip, about 1.7 GB per clip across 24 layers without checkpointing. That is an order-of-magnitude estimate, not sourced.
- With checkpointing and the unmasked context only, the official 24-clip micro-batch fits an 80 GB-class GPU.

**What fits where** (estimates)

| Free memory | What fits |
|---|---|
| 16 GB | Frozen inference of anything up to ViT-g comfortably, and the 7B at batch 1–4. ExPLoRA ViT-L at micro-batch 8–32 (images) with checkpointing (ExPLoRA's own 4 × A4000 setup is in this class). V-JEPA 2 ViT-L continuation at micro-batch ~2–6 with accumulation. Full FT of ConvNeXt-T, ViT-B and ViT-L at modest batch. |
| 40–80 GB | Full FT or continuation of V-JEPA 2 ViT-L at micro-batch ~12–24. ExPLoRA on ViT-g. Full FT of ViT-g (16 GB states) with checkpointing. LoRA on the 7B. |
| 117 GB | Full FT of ViT-g at large batch. The 7B still does not fit for full FT on one GPU (107 GB of states before activations). |
| Several partially free GPUs | 7B full FT with FSDP/ZeRO-3 across 4 GPUs → ≈ 27 GB of states each + activations; across 8 → ≈ 13 GB each. Over NVLink 900 GB/s within the node this is communication-feasible, but the 7B is not needed for any target-paper comparison. |

**A100 host (3 × A100)**
- 40 GB cards cover everything up to ViT-g full FT with checkpointing; 80 GB cards handle the V-JEPA 2 ViT-L continuation at the official micro-batch.
- The BF16 rate is ~1/3 of an H200, but the cards are dedicated rather than shared, so the A100 host may match a 70–99 %-busy H200 in wall-clock. That is unmeasured.

### Gaps
- Exact lab server model (DGX H200 with H200 SXM?) not confirmed. NVL numbers differ by about 15 %.
- No published training-throughput benchmark (images/s or clips/s) for DINOv3 or V-JEPA 2 on the H200 itself was found. The throughput figures above are derived from H100 recipes and FLOP counts.
- An official per-GPU B200 spec sheet (vs DGX/HGX system totals) was not fetched. Access to a B200 system is unconfirmed.
- MPS/MIG configuration on the lab server is unknown. It determines whether a co-scheduled job gets a predictable share.

---

## Q4. How do the four target papers' models and hardware compare in parameters and compute with what is feasible now, and what GPU-hour budgets are realistic?

### Takeaway
All four target papers use models of **≤ ~0.15 B parameters and ≤ ~0.55 TMAC per sample**:
- Choi 2020: VGG16, 138 M with FC, 15.5 GMAC per image.
- Kim 2023: GoogLeNet, 6.6 M, plus a ~48 M BiLSTM; ~540 GMAC per 180-s clip.
- Yang 2026: ResNet-18, 11.7 M; ~13–16 GMAC per 7–9-frame window.
- Kamagata 2026: ViT-S, ~22 M, plus a ~7 M head; ~74 GMAC per 16-frame clip.

Only Kamagata states its hardware: a DGX A100, 8 × 40 GB.

Feasible now on one shared H200:
- Frozen V-JEPA 2 ViT-L: 300 M, ~0.83 TMAC per 16×256² clip.
- ViT-g: 1 B, ~2.5 TMAC.
- DINOv3 7B on stills: ~1.36 TMAC per 224² image.

That is **10–40× Kamagata's encoder per sample**, and training any target paper's model takes minutes of H200 time.

Realistic budgets:
1. Frozen extraction: under a few GPU-hours, and bound by decoding and I/O.
2. LP-FT or LoRA with 3–5 seeds: ~2–20 GPU-hours.
3. Continued SSL: ~30–150 GPU-hours for ViT-B/L on 0.25–1 M images ExPLoRA-style, up to ~400–1,000 for full-unfreeze or long schedules; ~30–80 GPU-hours (up to ~400) for V-JEPA 2 ViT-L on 10k–100k clips.

### Cited Findings

**The four papers, from the project's full-text notes (V1)**
- **Choi 2020** — [local note](~/Projects/Wave_Analysis/docs/literature/papers/choi2020.md):
  - Classification: VGG16 (best: 84.05 % ± 0.14), ResNet, Inception v3 and DenseNet with ImageNet transfer and GAP/GMP heads, on one-view lower halves resized to 224².
  - Ship regression: frame differences → pretrained VGG → bidirectional ConvLSTM (256 filters, 3×3) → GAP → FC, on 16-frame clips.
  - Hardware: not recorded in the project note.
- **Kim 2023** — [local note](~/Projects/Wave_Analysis/docs/literature/papers/kim2023.md):
  - Snapshot CNN of 4 blocks (16-32-64-64).
  - GoogLeNet features → BiLSTM (2,000 hidden) → dropout 0.5 → FC.
  - Grayscale frames every 0.5 s; sequences of 6–300 s; best 180 s.
  - Hardware: not recorded.
- **Yang 2026** — [local note](~/Projects/Wave_Analysis/docs/literature/papers/yang2026.md):
  - ResNet-18 per frame + (sin, cos) attitude MLP → multi-head self-attention fusion → mean pool → MLP. 7–9 frames at 2 Hz; ~4,700 simulated samples.
  - Hardware: not recorded.
- **Kamagata 2026** — [local note](~/Projects/Wave_Analysis/docs/literature/papers/kamagata2026.md); [arXiv:2607.11998](https://arxiv.org/abs/2607.11998):
  - **[recent, verified: arXiv abs page, v1 13 Jul 2026, only version]**
  - Model: "V-JEPA" ViT-S (`vit_small_patch16_224`) giving 384-d frame embeddings; Farnebäck optical-flow saliency crops; SlowFast transformer of 2 layers, d = 384 (fast: 16 frames, 4 heads; slow: every 4th frame, 8 heads).
  - Training: 894 clip windows from 6 scenes; AdamW, 50 epochs, early stop at 31.
  - Hardware: "NVIDIA DGX A100" (abstract), 8 × 40 GB (PDF, per the project note).
  - Result: held-out H_s RMSE 0.586 m, R² 0.107.
- The released V-JEPA (2024) checkpoints are ViT-L/16 and ViT-H/16 only, and V-JEPA 2 starts at ViT-L (V-JEPA 2.1 adds a ViT-B 80 M). **No official ViT-S V-JEPA exists.** — [facebookresearch/jepa](https://github.com/facebookresearch/jepa); [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)

**Parameter and compute figures** (torchvision; the "GFLOPS" column is in MACs per the convention note at the top) — [torchvision models](https://docs.pytorch.org/vision/stable/models.html)

| Model | Params | GMAC per 224² image |
|---|---|---|
| VGG16 | 138.4 M | 15.47 |
| GoogLeNet | 6.6 M | 1.5 |
| Inception v3 | 27.2 M | 5.71 |
| ResNet-18 | 11.7 M | 1.81 |
| DenseNet-121 | 8.0 M | 2.83 |
| ConvNeXt-T | 28.6 M | 4.46 |
| ViT-B/16 | 86.6 M | 17.56 |
| ViT-L/16 | 304.3 M | 61.55 |

- Additional sizes: DINOv3 ConvNeXt-T 29 M, ViT-S 21 M, ViT-B 86 M, ViT-L 300 M, ViT-7B 6,716 M — [dinov3](https://github.com/facebookresearch/dinov3). V-JEPA 2 ViT-L 300 M, ViT-H 600 M, ViT-g 1 B; V-JEPA 2.1 ViT-G 2 B — [vjepa2](https://github.com/facebookresearch/vjepa2).

**Measured compute anchors**
- V-JEPA 2 ViT-L frozen feature extraction on the shared H200: ≈ 1.5 s per clip for four arms, decode included. — [local note](~/Projects/Wave_Analysis/docs/experiments/webcoos_video.md)
- FINO ViT-L continued SSL: 10.7–920 GPU-hours per domain (H100/H200). — [arXiv:2606.05107](https://arxiv.org/html/2606.05107)
- ExPLoRA ViT-L: 96–320 GPU-hours (RTX 6000 Ada / 4 × A4000). — [arXiv:2406.10973](https://arxiv.org/html/2406.10973)
- RAD-DINO ViT-B continuation: 37 H100-hours at 100k images, 373 at 1M, 1,307 at 3.5M. — [arXiv:2509.12818](https://arxiv.org/html/2509.12818v1)
- DINOv3 ViT-L ImageNet-1k recipe: ≈ 448 H100-hours. — [dinov3](https://github.com/facebookresearch/dinov3)
- EchoJEPA-L: 8 × H100, ≈ 86 M clip-samples (time not reported). — [arXiv:2602.02603](https://arxiv.org/html/2602.02603)

### Inferences

**Per-sample compute** (estimates; ViT MACs ≈ params × tokens + 2 × layers × tokens² × width)

| Model | Per-sample compute | Basis |
|---|---|---|
| Choi classification | 15.5 GMAC | one 224² view (VGG16) |
| Choi ship regression | ≈ 250 GMAC per clip | 16 frames × VGG, + ~5 GMAC ConvLSTM; assumes 224² |
| Kim snapshot CNN | < 0.1 M params | — |
| Kim CNN-BiLSTM | ≈ 557 GMAC per 180-s clip | 360 frames × 1.5 GMAC GoogLeNet; BiLSTM ≈ 2 × 4 × 2,000 × (1,024 + 2,000) ≈ 48 M params, assuming 1,024-d GoogLeNet features → ~17 GMAC over 360 steps |
| Yang | ≈ 13–16 GMAC per window | 7–9 frames × 1.81 |
| Kamagata | ≈ 74 GMAC per clip | ViT-S/16 ≈ 22 M × 197 tokens + attention ≈ 4.6 GMAC per frame, × 16 frames; SlowFast head ≈ 2 streams × 2 layers × ~1.8 M ≈ 7 M params |
| This project, buoy model | ≈ 27 GMAC per strip | DINOv3 ConvNeXt-T, 29 M; 6 views × 4.46 at 224² (actual input size not checked) |
| V-JEPA 2 ViT-L, 16×256² | ≈ 0.83 TMAC | — |
| V-JEPA 2 ViT-g, 16×256² | ≈ 2.5 TMAC | — |
| V-JEPA 2 ViT-g, 64×384² | ≈ 57 TMAC | 18,432 tokens |
| DINOv3 7B at 224² | ≈ 1.36 TMAC | 201 tokens |

**Ratios**
- V-JEPA 2 ViT-L per clip ≈ 11× Kamagata's encoder and ≈ 1.5× Kim's whole 180-s clip.
- The 7B per still ≈ 88× Choi's VGG16 per view.
- Parameter count: ViT-L is ≈ 14× Kamagata's ViT-S and ≈ 26× Yang's ResNet-18.

**Hardware gap**
- Kamagata's DGX A100 (8 × 40 GB = 320 GB, 8 × 312 = 2.5 PFLOPS dense BF16) is far more than a 22 M-parameter model on 894 clips needs. One epoch is ~66 TMAC of forward compute, ≈ 200 TMAC with the backward pass, i.e. about 1–2 s of one A100's dense BF16 (estimate; their optical-flow preprocessing on CPU would dominate).
- A single partially free H200 exceeds what any of the four papers needed by orders of magnitude.
- None of the four papers used self-supervised pretraining on in-domain data, or any encoder larger than 0.14 B.

**GPU-hour budgets for this project** (estimates; dedicated-GPU numbers; ×2–3 on a 70–99 %-busy H200)

1. **Frozen feature extraction**
   - 246k buoy views through DINOv3 ViT-L at 224²: ≈ 15 PMAC → minutes of compute.
   - Through the 7B: ≈ 0.33 EMAC → ~0.5–1 GPU-hour; 1M shore stills ≈ 2–4 GPU-hours at 224², ×4–5 at 448².
   - 100k shore clips through V-JEPA 2 ViT-L: ≈ 0.08 EMAC of compute, but ≈ 40 GPU-hours end to end at the measured ~1.5 s per clip (decode-bound), or less once clips are pre-cut.
   - The same clips through ViT-g at 64×384²: ≈ 5.7 EMAC ≈ 10–15 GPU-hours.
   - **Budget: ≤ 5 GPU-hours for stills, 10–50 for video, dominated by decode and I/O.**
2. **LP-FT / LoRA, 3–5 seeds**
   - ConvNeXt-T LP-FT on ~200k views × 30 epochs ≈ 6 M images × ~13 GMAC (forward + backward) ≈ 0.08 EMAC → under 1 GPU-hour per seed. Wall-clock is set by the data loader.
   - DINOv3 ViT-L with LoRA: ≈ 6 M × ~180 GMAC ≈ 1.1 EMAC → ~2–4 GPU-hours per seed.
   - V-JEPA 2 attentive probe on cached features: minutes per seed. With LoRA on the encoder over 10k clips × 20 epochs ≈ 0.5 EMAC + decode → ~2–5 GPU-hours per seed.
   - **Budget: 5 seeds ≈ 2–20 GPU-hours per arm.**
3. **Continued SSL pretraining**
   - Images (~250k–1M), ViT-B/L ExPLoRA-style: **≈ 30–150 GPU-hours**. FINO-style full unfreeze, ≥ 50 epochs: ≈ 150–1,000 (anchors: FINO 161 at 77k images; RAD-DINO 373 H100-hours at 1M for a ViT-B).
   - Video (10k–100k clips), V-JEPA 2 ViT-L: **≈ 30–80 GPU-hours** for 10–20 M clip-samples; ≈ 250–400 at EchoJEPA-L scale.
   - Add pre-cutting: one decode pass ≈ 1–2 s per raw clip, CPU or NVDEC.
   - Add 3–5 downstream seeds on top (item 2).
   - FINO's development total (34,673 GPU-hours vs 4,947 for the final runs) is a warning: **ablations and tuning cost ~5–7× the final runs.**

**What follows for the comparison with the four papers**
- Compute is not the constraint relative to the target papers. Labels are: 67 buoys, a handful of shore cameras.
- That points continued SSL at using the abundant unlabelled data, with the strict held-out-station test as the arbiter.
- One frozen-vs-continued pair per backbone is affordable within ~100–200 GPU-hours (stills) or ~100 (video).

### Gaps
- Training hardware for Choi 2020, Kim 2023 and Yang 2026 is not in the project's notes. These notes did not re-open the PDFs, so their GPUs and training times are unknown here.
- Kamagata's "V-JEPA ViT-S" corresponds to no official V-JEPA checkpoint. The actual weights used (perhaps a timm `vit_small_patch16_224` with ImageNet weights) cannot be determined from the abstract or the project note.
- Kim's GoogLeNet feature dimension and input resolution, and Choi's ship-video input resolution, were assumed (1,024-d; 224²). The per-clip MACs for those two rows are approximate.
- The buoy model's actual input resolution and the dataset size per epoch were not checked against the project's configs, so the LP-FT budget is order-of-magnitude.
- No measured training throughput on the lab server for any SSL recipe. Every continued-pretraining budget above is an estimate anchored on H100 recipes and FLOP counts.
