# Pretrained video backbones as feature extractors for shore-camera wave estimation (Hs, Tp, direction)

Scope: video encoders for a small, domain-specific regression task whose signal is periodic physical motion (5-15 s waves, breaking, swash) seen by a fixed, oblique WebCOOS camera (H.264, 2688x1520, 20 fps, 4-10 min clips, one per 30-min CDIP record), tested on unseen days and sites. Compute: one shared H200 (141 GB). Research date 2026-11-18. "Verified" means read in the primary source (paper, official repo/config, or Hugging Face/GitHub API) in this session; "search summary" means taken from a search-engine digest and not opened.

## 1. Which video encoders exist, when were they released, and which are superseded?

### Takeaway
The current frontier of open, motion-strong video encoders is the Meta JEPA line (V-JEPA 2, June 2025; V-JEPA 2.1, March 2026), with InternVideo-Next (Dec 2025) competitive at ViT-L size and Google's 4DS/VideoPrism as the main pixel-MAE / video-text alternatives. V-JEPA (2024), VideoMAE/VideoMAE V2, Hiera, UMT, InternVideo2 and the supervised 2019-2022 models (SlowFast, X3D, TimeSformer, ViViT, MViTv2, VideoSwin) are superseded on motion benchmarks, and several carry non-commercial weight licences.

### Cited Findings

Model landscape (release date, sizes, native input, status):

| Model (release) | Sizes released | Native input | Status in Nov 2026 | Source |
|---|---|---|---|---|
| V-JEPA (Feb 2024) | ViT-L/16 224, ViT-H/16 224, ViT-H/16 384 (no ViT-S/B released) | 16 frames, tubelet 2, 224 or 384 | Superseded by V-JEPA 2/2.1; weights CC BY-NC 4.0 | [V-JEPA paper](https://arxiv.org/abs/2404.08471); [jepa repo README/LICENSE](https://github.com/facebookresearch/jepa) |
| V-JEPA 2 (code 2025-06-11 blog; repo news "2025-06-25 V-JEPA 2 is released") | ViT-L 300M, ViT-H 600M, ViT-g 1B (256 px), ViT-g 1B (384 px); HF API counts 326M / 654M / 1.035B | 2x16x16 tubelets, 3D-RoPE; pretrain 16 f @ 256 then cooldown 64 f @ 384, at 4 fps | Current; Apache-2.0/MIT | [V-JEPA 2 paper](https://arxiv.org/abs/2506.09985); [vjepa2 README](https://github.com/facebookresearch/vjepa2); [Meta blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/); [HF API](https://huggingface.co/api/models/facebook/vjepa2-vitg-fpc64-384) |
| V-JEPA 2-AC | ViT-g encoder + action-conditioned predictor | robot video | Robotics planner; not relevant for regression | [vjepa2 README](https://github.com/facebookresearch/vjepa2) |
| V-JEPA 2.1 (repo news 2026-03-16; arXiv v3 2026-06-11) | ViT-B 80M and ViT-L 300M (distilled from ViT-G), ViT-g 1B, ViT-G 2B, all 384 px | 16 f @ 256 then 64 f @ 384 (images 512) | Current; torch.hub only (official HF not listed; a third-party HF conversion exists) | [V-JEPA 2.1 paper](https://arxiv.org/abs/2603.14482); [vjepa2 README](https://github.com/facebookresearch/vjepa2); [third-party HF](https://huggingface.co/apiantonio/vjepa2.1-vit-giant-384) |
| VideoMAE (Mar 2022, NeurIPS 2022) | ViT-S/B/L/H | 16 f @ 224 | Superseded; weights CC BY-NC 4.0 | [VideoMAE](https://arxiv.org/abs/2203.12602); [HF API](https://huggingface.co/api/models/MCG-NJU/videomae-large) |
| VideoMAE V2 (Mar 2023, CVPR 2023) | ViT-B ... ViT-g (1B) | 16 f | Superseded in frozen setting; weights CC BY-NC 4.0, code MIT | [VideoMAE V2](https://arxiv.org/abs/2303.16727); [HF API](https://huggingface.co/api/models/OpenGVLab/VideoMAEv2-giant) |
| UMT (Mar 2023, ICCV 2023) | ViT-L/16 headline | - | Superseded (its recipe fed InternVideo2); HF/GitHub MIT | [UMT abstract](https://arxiv.org/abs/2303.16058); [GitHub API](https://api.github.com/repos/OpenGVLab/unmasked_teacher) |
| Hiera (Jun 2023) | B 51M, B+ 69M, L 213M, H 672M (video) | 16 f @ 224 | Efficient but superseded; weights CC BY-NC 4.0 (code Apache-2.0) | [Hiera](https://arxiv.org/abs/2306.00989); [HF API](https://huggingface.co/api/models/facebook/hiera-huge-224-hf) |
| InternVideo2 (Mar 2024, ECCV 2024) | 1B, 6B | - | Superseded as an encoder by InternVideo-Next; Stage2-1B HF Apache-2.0, gated | [InternVideo2](https://arxiv.org/abs/2403.15377); [HF API](https://huggingface.co/api/models/OpenGVLab/InternVideo2-Stage2_1B-224p-f4) |
| InternVideo2.5 (Jan 2025) | video MLLM | long context | An MLLM, not a stand-alone encoder | [InternVideo2.5](https://arxiv.org/abs/2501.12386) |
| InternVideo-Next (Dec 2025; v2 2026-03-30) | ViT-B, ViT-L (patch 14) | Stage 1 16 f, Stage 2 32 f, 224 px | Current; paper v2 says weights "will be released"; no HF repo found by API | [InternVideo-Next](https://arxiv.org/abs/2512.01342) |
| InternVideo3 (Jun 2026) | 8B instruct MLLM (agent) | long video | MLLM/agent, not a feature encoder (search summary) | [InternVideo3 arXiv 2606.12195](https://arxiv.org/abs/2606.12195) |
| VideoPrism (paper Feb 2024; public weights Jun-Jul 2025) | public B 114M (16 f @ 288), L 354M (8 f @ 288); paper ViT-g not released | 288 px; other frame counts via interpolated temporal pos-emb | Current for video-text; JAX/Flax; Apache-2.0 | [VideoPrism model card](https://huggingface.co/google/videoprism); [GitHub API](https://api.github.com/repos/google-deepmind/videoprism) |
| 4DS (Dec 2024; v2 Jul 2025) | released: 4DS-B-dist-e 88M, 4DS-L 305M, 4DS-e 3.8B (JAX .npz); paper scales to 22B | video MAE | Current pixel-MAE family; repo Apache-2.0 | [4DS paper](https://arxiv.org/abs/2412.15212); [representations4d README](https://github.com/google-deepmind/representations4d) |
| RVM (Dec 2025) | ViT-S 34M ... ViT-H 743M (encoder+decoder) | 256 px | New recurrent video MAE, claims strong small-model efficiency | [representations4d README](https://github.com/google-deepmind/representations4d); [RVM arXiv](https://arxiv.org/abs/2512.13684) |
| Cosmos-Embed1 (May 2025) | 224p, 336p, 448p (448p = 1.197B params by HF API) | optimised for 8 frames at 1-2 fps; arbitrary non-square resolution | Video-text retrieval embedder; NVIDIA Open Model License | [HF card](https://huggingface.co/nvidia/Cosmos-Embed1-448p); [NVIDIA page](https://research.nvidia.com/labs/cosmos-lab/cosmos-embed1/) |
| Cosmos Tokenizer (Oct 2024) | continuous/discrete video tokenizers for generation | - | Compression tokenizer; code Apache-2.0 | [GitHub API](https://api.github.com/repos/NVIDIA/Cosmos-Tokenizer) |
| TimeSformer (2021), ViViT (2021), SlowFast (2019), MViTv2 (2022), VideoSwin (2021), X3D (2020) | supervised Kinetics-era models | 8-64 frames, 224 | Superseded on SSv2 by self-supervised models (numbers in section 2) | [VideoMAE SSv2 table](https://arxiv.org/abs/2203.12602); [Hiera tables](https://arxiv.org/abs/2306.00989) |

- V-JEPA 2 was pretrained on VideoMix22M (22M samples, "over 1 million hours" of video plus images); the paper's Table 1 lists YT1B at 1.6M hours while Section 2.3 says 1.4M, an internal inconsistency. — [V-JEPA 2](https://arxiv.org/html/2506.09985)
- V-JEPA 2.1 adds a "dense predictive loss" (all tokens, visible and masked, contribute) and "deep self-supervision" (losses at several intermediate layers, fused by an MLP). — [V-JEPA 2.1](https://arxiv.org/html/2603.14482)
- InternVideo-Next is pretrained without video-text supervision on about 1.1M public videos (K-Mash) with a SigLIP2-1B semantic teacher; the authors call it the first such model to beat video-text models on both K400 and SSv2. — [InternVideo-Next](https://arxiv.org/html/2512.01342v2)
- The VideoPrism card says "Our public models perform slightly worse than the paper models"; the paper's ViT-g (SSv2 68.5, Diving-48 71.3 frozen) is not public. — [VideoPrism model card](https://huggingface.co/google/videoprism)
- Image-only encoders that appear as baselines in the video tables: DINOv2-g, SigLIP2, Perception Encoder PE-core-G (Meta, repo created 2025-04-16, Apache-2.0) and DINOv3 ViT-7B. — [V-JEPA 2 Table 4](https://arxiv.org/html/2506.09985); [V-JEPA 2.1](https://arxiv.org/html/2603.14482); [GitHub API](https://api.github.com/repos/facebookresearch/perception_models)
- Video-generative backbones are now also being used as perception encoders, e.g. GenCeption on WAN 2.1 (14B and 1.3B checkpoints released). — [representations4d README](https://github.com/google-deepmind/representations4d)

### Inferences
- For a 2026 study, V-JEPA 2 (ViT-L for screening, ViT-g-384 for the best run) is the default choice. V-JEPA 2.1 is the upgrade if dense, per-patch features matter (they likely do for localising breaking), and 4DS/VideoMAE V2 serve as the pixel-reconstruction contrast. V-JEPA v1 should not be used: it is superseded, non-commercial, and has no small model.
- **Kamagata et al. 2026's "V-JEPA ViT-S" is suspect.** The official V-JEPA release has only ViT-L/H, and the paper names the backbone `vit_small_patch16_224`, a timm ImageNet identifier, without giving a checkpoint source (section 6). Its R²=0.107 may therefore reflect an image backbone or an unofficial one, not V-JEPA.

### Gaps
- Fine-tune SSv2/K400 numbers for UMT, InternVideo2 and X3D were not retrieved.
- The V-JEPA 2.1 weight licence is not stated separately from the MIT code licence.
- InternVideo-Next weights: no public HF repository was found (the HF API returned no model for the obvious names), so availability is unconfirmed.
- The Cosmos tokenizers' weight licence and any use of them as regression features: no source found.

## 2. Which models capture fine motion and physical dynamics best?

### Takeaway
On the motion-centric benchmarks (SSv2, Diving-48) under frozen attentive probes, the V-JEPA 2/2.1 family leads clearly: SSv2 77.3-77.7 and Diving-48 about 89-90, against 68-70 SSv2 for InternVideo2/VideoPrism and 50-56 for image models and VideoMAE V2. InternVideo-Next-L slightly beats V-JEPA 2-L in its own protocol. "Intuitive physics" (IntPhys 2) remains near chance for every model. Where the signal is low-level dynamics (point tracking, weather fields, cyclone pressure), pixel-MAE models (4DS, VideoMAE V2) match or beat V-JEPA in the one cross-domain scientific benchmark (SciVid).

### Cited Findings

Frozen attentive-probe top-1 (%), V-JEPA 2 Table 4. "Same protocol" means Meta re-ran it with their probe; "lit." means a number copied from the literature.

| Model | Params | SSv2 | Diving-48 | K400 | Source |
|---|---|---|---|---|---|
| V-JEPA 2 ViT-L (256) | 300M | 73.7 | 89.0 | 85.1 | [V-JEPA 2](https://arxiv.org/html/2506.09985) |
| V-JEPA 2 ViT-H (256) | 600M | 74.0 | 89.8 | 85.3 | same |
| V-JEPA 2 ViT-g (256) | 1B | 75.3 | 90.1 | 86.6 | same |
| V-JEPA 2 ViT-g (384) | 1B | 77.3 | 90.2 | 87.3 | same |
| V-JEPA ViT-H (same protocol) | 600M | 74.3 | 87.9 | 84.5 | same |
| InternVideo2s2-1B (same protocol) | 1B | 69.7 | 86.4 | 89.4 | same |
| VideoPrism-g (lit.) | 1B | 68.5 | 71.3 | 87.6 | same |
| InternVideo2-6B (lit.) | 6B | 67.7 | - | 88.8 | same |
| VideoMAE V2-g (lit.) | 1B | 56.1 | - | 82.8 | same |
| PE-core-G, image (same protocol) | 1.9B | 55.4 | 76.9 | 88.5 | same |
| DINOv2, image (same protocol) | 1.1B | 50.7 | 82.5 | 83.6 | same |
| SigLIP2, image (same protocol) | 1.2B | 49.9 | 75.3 | 87.3 | same |
| V-JEPA 2.1 ViT-g / ViT-G (384) | 1B / 2B | 76.9 / 77.7 | 89.0 / 89.2 | 87.0 / 87.7 | [V-JEPA 2.1](https://arxiv.org/html/2603.14482) |
| DINOv3 ViT-7B, image | 7B | 70.1 | - | 87.8 | same |
| VideoPrism public B / L | 114M / 354M | 62.2 / 64.6 | 64.3 / 67.6 | 82.9 / 85.0 | [VideoPrism card](https://huggingface.co/google/videoprism) |

- InternVideo-Next's own frozen protocol (16 frames): InternVideo-Next-L SSv2 73.0 vs V-JEPA 2-L 72.0, InternVideo2-L 65.9, VideoMAE V2-1B 56.1; Base 70.1. — [InternVideo-Next Table 5](https://arxiv.org/html/2512.01342v2). (V-JEPA 2-L is 72.0 here and 73.7 in Meta's protocol, so the protocols differ by about 1-2 points.)
- The V-JEPA (2024) paper's own frozen protocol: V-JEPA ViT-H/16 SSv2 71.4 (384 px: 72.2) vs VideoMAE-H 66.2, MVD-L 66.5, OmniMAE-H 65.4, Hiera-H 64.7, VideoMAE V2-g 61.2, DINOv2-g 50.6, OpenCLIP-G 34.8. The paper attributes its "over +21 points" SSv2 gain over image models to motion understanding, and notes that K400 labels can often be "inferred using appearance-based cues". — [V-JEPA](https://arxiv.org/html/2404.08471)
- Fine-tuned SSv2 top-1, older and supervised models: TimeSformer ViT-B 59.5 (8 f), ViT-L 62.4 (64 f); SlowFast R101 63.1; ViViT-L FE 65.9; Motionformer-L 68.1; VideoSwin-B 69.6; MaskFeat MViT-L@312 75.0; VideoMAE ViT-S/B/L 66.8/70.8/74.3 (no external data), ViT-L 32 f 75.4. — [VideoMAE Table 6](https://arxiv.org/abs/2203.12602)
- Fine-tuned SSv2: Hiera-L 75.1 (2 clips), Hiera-L32 76.5; MViTv2-L MaskFeat@312 74.4; ViT-L MAE 74.3. — [Hiera Table 6](https://arxiv.org/abs/2306.00989). VideoMAE V2-g 77.0 SSv2, 68.7 SSv1, 90.0 K400. — [VideoMAE V2 abstract](https://arxiv.org/abs/2303.16727). V-JEPA ViT-L/16 fine-tuned 75.1. — [V-JEPA](https://arxiv.org/html/2404.08471)
- Frozen V-JEPA 2 ViT-g-384 (77.3 SSv2) matches or exceeds fully fine-tuned VideoMAE V2-g (77.0). This is a comparison across two sources and two protocols. — [V-JEPA 2](https://arxiv.org/html/2506.09985); [VideoMAE V2](https://arxiv.org/abs/2303.16727)
- Dense and low-level dynamics: V-JEPA 2.1 lifts dense-task quality far above V-JEPA 2 with little change on SSv2. ViT-g DAVIS J&F 52.5 -> 68.1, NYUv2 depth RMSE 0.642 -> 0.350, ADE20K mIoU 24.4 -> 47.8. — [V-JEPA 2.1](https://arxiv.org/html/2603.14482). InternVideo-Next-L tracking (Waymo mIoU) 72.4 vs V-JEPA 2-L 68.9, InternVideo2-L 63.0, DINOv3-L 59.7. — [InternVideo-Next Table 7](https://arxiv.org/html/2512.01342v2)
- Intuitive physics (IntPhys, 2025): a V-JEPA model trained on natural video reaches about 98% zero-shot on IntPhys via prediction-error "surprise", and about 62% on InfLevel (one review says 66%); multimodal LLMs and pixel-generative models fall short. — [Garrido et al. 2025](https://arxiv.org/abs/2502.11831) (search summary of the paper)
- IntPhys 2 (2025): Meta states current video models "are at or close to chance" while humans are near perfect. — [Meta blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/). The Decoder reports V-JEPA 2 at "44.5% paired accuracy". The V-JEPA 2 paper's introduction gives 44.5 as an MVP (Minimal Video Pairs) score, so the Decoder likely conflated the two benchmarks. — [The Decoder](https://the-decoder.com/metas-latest-model-highlights-the-challenge-ai-faces-in-long-term-planning-and-causal-reasoning/); [V-JEPA 2](https://arxiv.org/html/2506.09985); Quanta also reports only slightly-better-than-chance performance — [Quanta](https://www.quantamagazine.org/how-one-ai-model-creates-a-physical-intuition-of-its-environment-20251003/)
- Layerwise frozen probing (temporal attentive probe, best layer), IntPhys2 VOE / MVP pair accuracy: V-JEPA-H 56.9 / 94.0, V-JEPA 2-g 58.8 / 93.3, V-JEPA 2.1-G 66.7 / 93.7, VideoMAE-H 58.8 / 92.0, VideoMAE V2-g 15.7* / 91.1 (*undertrained), LTX-Video 13B (diffusion) 47.1 / 84.3. MVP collapses under frame shuffling, which shows genuine temporal sensitivity. — [Punzo et al. 2026](https://arxiv.org/abs/2606.09646)
- V-JEPA 2 (L, g) and VideoMAE V2-g encode camera motion with >90 ROC-AUC at 60-70% of network depth (largely recoverable even at 20% depth), but stay near chance on intuitive-physics tasks with linear probes. — [Musa et al. 2026](https://arxiv.org/abs/2609.01551)
- Physical-property readout: frozen generative (DynamiCrafter) and self-supervised video features predict elasticity and viscosity well, and transfer to real videos (a real viscosity set of 12 liquids). Friction is weak (AUC 0.47-0.58 without adaptation), and fine-tuning on real data helps. — [Zhan, Ma, Xie, Zisserman 2025](https://arxiv.org/abs/2510.02311)
- Fluid PDE systems (The Well: shear flow, Rayleigh-Benard, active matter), frozen encoders plus attentive probe for parameter estimation (MSE): a JEPA (CNN, pretrained per dataset) beats VideoMAE (ViT-tiny, per dataset) by 28-51%. On shear flow, JEPA with 10% of the data (0.57) beats VideoMAE with all of it (0.67). A physics operator model (DISCO) is best overall. — [Qu et al. 2026, Polymathic AI](https://arxiv.org/abs/2603.13227)
- SciVid (frozen backbones, same protocol): Digital Typhoon pressure RMSE 4DS-L 3.84, 4DS-e 4.23, V-JEPA-H 4.16, VideoMAE V2-g 4.53, DINOv2-L 6.31. WeatherBench2 Z500 wRMSE VideoMAE V2-g 594, 4DS-L 597, 4DS-e 601, V-JEPA-H 619, DINOv2-L 658. Point tracking (STIR) 4DS-e 0.513, V-JEPA-H 0.443, VideoMAE V2-g 0.344. "No single backbone is best across all tasks"; pixel-space MAE models did best on weather fields. — [SciVid, Hasson et al. 2025](https://arxiv.org/abs/2507.03578)

### Inferences
- Wave estimation sits between "action" motion (SSv2) and "field" dynamics (weather, fluids). The evidence points to V-JEPA 2/2.1 for temporal and motion structure. A pixel-MAE model (VideoMAE V2-g or 4DS-L) is the necessary second arm, because SciVid shows the MAE family winning on continuous geophysical fields.
- Image-only encoders (DINOv2/v3, SigLIP2) are about 20-25 SSv2 points behind on motion. Use them only as a per-frame baseline: that comparison measures how much the temporal signal adds beyond appearance (whitecap texture).
- "Intuitive physics" benchmarks test object permanence and continuity, not oscillatory fluid dynamics, so IntPhys scores say little about wave period or height. The fluid-parameter (Polymathic) and viscosity (Zhan et al.) results are closer analogues.

### Gaps
- No benchmark tests periodic or oscillatory motion (frequency, period) with frozen video encoders. Repetition-counting benchmarks (e.g. RepCount, Countix) were not searched in this session and would be the closest analogue.
- Diving-48 fine-tune numbers for the supervised models were not retrieved.
- No published comparison of V-JEPA 2 vs 4DS on water or fluid video was found.

## 3. How many frames and what temporal span or fps do the models accept? How to cover a 10-15 s wave period, and what longer context costs

### Takeaway
V-JEPA 2/2.1 is the only family whose pretraining already spans a wave period: 64 frames at 4 fps = 16 s (cooldown), with 3D-RoPE allowing other lengths. Most other encoders are trained on 8-32 frames covering about 1-4 s. Kamagata's 16 frames at 10 fps cover 1.6 s, less than one wave period. Under self-attention, cost grows faster than linearly with frames: going from 16 f@256 to 64 f@384 is about 22x the FLOPs per clip for ViT-g.

### Cited Findings
- V-JEPA 2 pretraining config: `fps: 4`, 16 frames at 256 px, tubelet 2, patch 16 (pretrain-256px-16f.yaml); cooldown `fps: 4`, 64 frames at 384 px, per-GPU batch 6 with activation checkpointing (cooldown-384px-64f.yaml). — [vjepa2 pretrain config](https://github.com/facebookresearch/vjepa2/blob/main/configs/train/vitg16/pretrain-256px-16f.yaml); [cooldown config](https://github.com/facebookresearch/vjepa2/blob/main/configs/train/vitg16/cooldown-384px-64f.yaml)
- V-JEPA 2: raising resolution (256 -> 384) and duration (16 -> 64 frames) in pretraining and evaluation lifts the six-task average to 88.2 (+4.0 over the ViT-L/16 baseline). Longer cooldown clips help even with 16-frame evaluation (+0.7). "128 or 256 frames gave no further gains beyond 64". — [V-JEPA 2](https://arxiv.org/html/2506.09985)
- V-JEPA 2 evaluation clips: SSv2 16x2x3 (ViT-g-384: 64x2x3, frame_step 2); Diving-48 32x4x3; K400 16x8x3. The ViT-L SSv2 config uses frame_step 4. The probe wrapper accepts `max_frames: 128`. — [V-JEPA 2 Table 4 caption](https://arxiv.org/html/2506.09985); [eval config vitg-384/ssv2](https://github.com/facebookresearch/vjepa2/blob/main/configs/eval/vitg-384/ssv2.yaml); [eval config vitg-384/diving48](https://github.com/facebookresearch/vjepa2/blob/main/configs/eval/vitg-384/diving48.yaml); [eval config vitl/ssv2](https://github.com/facebookresearch/vjepa2/blob/main/configs/eval/vitl/ssv2.yaml)
- The HF V-JEPA 2 checkpoints are named "fpc64" (64 frames per clip), and the usage example samples 64 frames. — [HF vjepa2-vitg-fpc64-384](https://huggingface.co/facebook/vjepa2-vitg-fpc64-384)
- The Meta blog says V-JEPA 2 "learns and makes predictions at a single time scale" and names hierarchical multi-time-scale models as future work. — [Meta blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)
- V-JEPA (2024): 16 frames with temporal stride 4. The paper is inconsistent on duration: "about 3 seconds" in Sec. 3.4 vs "roughly 2 seconds at 30 fps" in Appendix B. — [V-JEPA](https://arxiv.org/html/2404.08471)
- VideoPrism public: B trained on 16 frames, L on 8 frames, at 288x288; other frame counts work by interpolating the temporal positional embeddings. — [VideoPrism card](https://huggingface.co/google/videoprism)
- InternVideo-Next: 16 frames (Stage 1), 32 frames (Stage 2), 224 px. — [InternVideo-Next](https://arxiv.org/html/2512.01342v2)
- Cosmos-Embed1: "optimized for 8 frames, sampled at 1-2 frames per second". — [Cosmos-Embed1 HF](https://huggingface.co/nvidia/Cosmos-Embed1-448p)
- Hiera and VideoMAE: 16 frames at 224 by default. Hiera-L32 and VideoMAE-L 32-frame variants gain about 1.4 and 1.1 SSv2 points over 16 frames (76.5 vs 75.1; 75.4 vs 74.3). — [Hiera](https://arxiv.org/abs/2306.00989); [VideoMAE](https://arxiv.org/abs/2203.12602)
- Kamagata 2026: "overlapping video patches of 16 frames with stride 8 at 10 frames per second", input 224x224, slow pathway 4 frames. Clip duration is not stated (16/10 = 1.6 s). — [Kamagata et al. 2026](https://arxiv.org/html/2607.11998v1)
- Kamagata et al. (arXiv:2606.13302) used 60-frame windows for peak period (ViViT; held-out RMSE 0.80 s). — [arXiv 2606.13302](https://arxiv.org/abs/2606.13302) (details per the project's verification log, `docs/literature/source_verification.md`)
- Compute anchor: ViT-L/16 at 16x224 = 597 GFLOPs per view and ViT-H = 1192 G (FLOPs counted as multiply-adds). — [Hiera Table 4](https://arxiv.org/abs/2306.00989). Hiera-L 413 G (32 f: 1029 G), and ViT-L 32 f = 1436 G. — [Hiera Table 6](https://arxiv.org/abs/2306.00989)

### Inferences
- **Sampling recommendation: 64 frames at 4 fps (one frame in five from 20 fps), giving 16 s per window.** This matches V-JEPA 2's pretraining fps, so the learned dynamics sit in their native time scale. It spans 1-3 periods for Tp = 5-15 s, and 2 Hz Nyquist is about 10x the highest swell frequency (0.2 Hz). Each tubelet covers 0.5 s at 4 fps, which is coarse for the sub-second breaking-onset motion, so the second arm should be 32 f at 10-20 fps (1.6-3.2 s) for breaking and swash detail. Concatenate the two views or pass them to the probe as two token sets.
- A 4-10 min record holds 15-37 non-overlapping 16-s windows. The record-level label (30-min buoy statistic) should be predicted from all of them: a second attentive pooling over window embeddings (or over the token sets) estimates Hs/Tp from about 20 wave periods, which matches how the buoy statistic is defined. 16 f at 10 fps (Kamagata) sees under one wave period per sample, a likely contributor to its low held-out skill.
- Token count = (frames/2) x (H/16) x (W/16). Estimated forward cost per window (2 FLOPs per multiply-add, linear + attention terms, dims from the vjepa2 source; this matches Hiera's ViT-L 597 GMAC = 1.19 TFLOP):

| Encoder | 16 f@256 (2,048 tok) | 64 f@256 (8,192) | 32 f@384 (9,216) | 64 f@384 (18,432) | 64 f@384x672 (32,256) |
|---|---|---|---|---|---|
| ViT-L (1024x24) | 1.6 TF | 11.5 TF | 13.9 TF | 44.5 TF | 122 TF |
| ViT-H (1280x32) | 3.3 TF | 21.3 TF | 25.5 TF | 78.9 TF | 211 TF |
| ViT-g (1408x40) | 5.1 TF | 31.7 TF | 37.7 TF | 113.7 TF | 300 TF |
| ViT-G (1664x48) | 8.9 TF | 51.6 TF | 61.1 TF | 176 TF | 451 TF |

  Going from 16 to 64 frames at fixed resolution costs about 6-7x (ViT-L/g) rather than 4x, because of the attention term. At 64 f, 384 px costs about 3.6-3.9x as much as 256 px.
- V-JEPA 2's finding that frames beyond 64 add nothing was measured on action benchmarks. For waves, whether 64 frames at 4 fps outperforms 64 frames at 2 fps (32 s) should be tested directly, since 2 fps would still be 10x the swell Nyquist rate.

### Gaps
- No source measures V-JEPA 2 or 2.1 accuracy as a function of input fps (e.g. 2 vs 4 vs 8 fps) at fixed frame count.
- The VideoMAE / VideoMAE V2 / InternVideo2 pretraining frame strides were not verified in this session.

## 4. Frozen probe vs fine-tuning on small datasets, and which probe heads to use

### Takeaway
Across V-JEPA's low-shot tables, EchoJEPA, the Polymathic fluid benchmark and SciVid, frozen features with a cross-attention (attentive) probe degrade much less with few labels than pixel-MAE features, and beat linear probes by large margins. Fine-tuning (or LoRA) helps mainly for low-level or continuous-field tasks (tracking, weather), which argues for frozen-first, then LoRA. Meta's own probe is large (about 50M params on ViT-L, about 90M on ViT-g, estimated), which is oversized for a few thousand clips.

### Cited Findings
- V-JEPA low-shot frozen evaluation (SSv2 with 5% / 10% / 50% of labels): V-JEPA-H/16-384 54.0 / 59.3 / 67.9, VideoMAE-H 41.4 / 48.1 / 60.5, VideoMAE V2-g 28.0 / 37.3 / 54.0, MVD-L 42.9 / 49.5 / 61.0. On K400 5%: V-JEPA-H-384 68.2 vs VideoMAE V2-g 37.0. — [V-JEPA Table 7](https://arxiv.org/html/2404.08471)
- V-JEPA (2024) attentive probe: one cross-attention layer with a learnable query, residual to the query, two-layer MLP with GeLU, LayerNorm, linear classifier. — [V-JEPA](https://arxiv.org/html/2404.08471)
- V-JEPA 2 attentive probe: four transformer blocks, the last one cross-attention with a single learnable query (three queries for EK100 anticipation), logits averaged over clips. — [V-JEPA 2](https://arxiv.org/html/2506.09985). Config: `num_probe_blocks: 4`, `num_heads: 16`; the ViT-g-384 probe uses the "multilevel" wrapper with `out_layers: [24, 29, 34, 39]` of a 40-layer encoder, `use_pos_embed: false`, `use_rope: true`. Learning rates are swept over 0.005-0.0001 in parallel heads. — [vjepa2 eval config](https://github.com/facebookresearch/vjepa2/blob/main/configs/eval/vitg-384/diving48.yaml)
- Hugging Face parameter totals: vjepa2-vitl-fpc16-256-ssv2 375.5M vs encoder vjepa2-vitl-fpc64-256 326.0M; vjepa2-vitg-fpc64-384-ssv2 1.128B vs encoder 1.035B. — [HF API ssv2 L](https://huggingface.co/api/models/facebook/vjepa2-vitl-fpc16-256-ssv2); [HF API ssv2 g](https://huggingface.co/api/models/facebook/vjepa2-vitg-fpc64-384-ssv2)
- V-JEPA ViT-L/16 on SSv2: frozen probe 69.5 vs full fine-tune 75.1. — [V-JEPA](https://arxiv.org/html/2404.08471)
- VideoMAE is data-efficient for in-domain pretraining from scratch: it "achieves impressive results on very small datasets (i.e., around 3k-4k videos) without using any extra data". HMDB51 is about 61% from pretraining alone vs 18.0% from scratch; UCF101/HMDB51 91.3/62.6 with in-domain pretraining vs 96.1/73.3 with K400 pretraining. — [VideoMAE](https://arxiv.org/abs/2203.12602)
- EchoJEPA (echocardiography, periodic cardiac motion; V-JEPA 2 adapted to the domain, frozen backbones plus attentive probe, depth 4, 16 heads). LVEF MAE: EchoJEPA-G 4.26 vs EchoPrime 5.33, PanEcho 5.43, EchoMAE-L 8.15. View accuracy with 1% of labels: EchoJEPA-G 78.6% vs about 21-22% for the baselines. In a controlled latent-vs-pixel comparison (same data and compute), EchoJEPA-L beats EchoMAE-L by 26.7% on LVEF MAE. Domain changes: frame rate raised from 4 to 24 fps, narrower aspect-ratio and crop-scale augmentation. Cross-site (Chicago) LVEF MAE 5.44 vs 6.71 (EchoPrime). — [EchoJEPA](https://arxiv.org/html/2602.02603)
- Polymathic fluid benchmark: frozen encoders plus attentive probes (100 epochs). JEPA at 10% of the data beats VideoMAE at 100% on shear flow (MSE 0.57 vs 0.67). — [Qu et al. 2026](https://arxiv.org/abs/2603.13227)
- SciVid: frozen backbones by default (the full suite runs in under a day on one H100). Cross-attention readout vs linear readout on FlyVsFly: 0.894 vs 0.568. Fine-tuning 4DS-e on STIR: 51.3/57.8 -> 61.2/69.2. On WeatherBench2, fine-tuning "helps significantly". LoRA rank 32 (about 2.3% of parameters) comes close to full fine-tuning (STIR 58.4 vs 59.3). With less data every model degrades (Digital Typhoon at 1/8 data: 4DS-e 7.23 vs 4DS-L 8.29 RMSE). — [SciVid](https://arxiv.org/abs/2507.03578)
- Layerwise probing: the temporal attentive probe (1 self-attention + 1 cross-attention layer, 16 heads) beats linear and MLP probes. Linear probes are weak on MVP (about 37-50%). The best layer is often at 50-75% depth, not the last. Frame-shuffle and single-frame-repeat controls are recommended. — [Punzo et al. 2026](https://arxiv.org/abs/2606.09646). Camera motion peaks at 60-70% depth. — [Musa et al. 2026](https://arxiv.org/abs/2609.01551)
- The wildfire fixed-camera paper gets label efficiency by learning each camera's normal dynamics from unlabeled footage (JEPA-like next-frame feature prediction with an EMA target), with per-camera and per-time-of-day calibration and label-free recalibration on new cameras. — [Muksimova and Umirzakova 2026, Sensors 26:6020](https://pmc.ncbi.nlm.nih.gov/articles/PMC13667166)

### Inferences
- The probe-parameter estimates (about 50M on ViT-L, about 93M on ViT-g) come from subtracting the HF totals; they assume the difference is the probe. With a few thousand labelled records, a probe that size is likely to overfit to site and day.
- **Recommended head for the wave task:** take tokens from 3-4 intermediate-to-late layers, as in Meta's multilevel probe (e.g. layers at about 60%, 75%, 87% and 100% depth). Then a 1-2-block attentive probe with d=256-384, 3 query tokens (Hs, Tp, direction, the last as sin/cos), heavy dropout, and a second attentive pooling over the 15-37 windows of a record. Sweep learning rates in parallel heads, as Meta's config does.
- Order of experiments: (1) frozen V-JEPA 2-L at 64 f/4 fps with this probe; (2) the same with ViT-g-384 and with VideoMAE V2-g / 4DS-L; (3) LoRA (rank 16-32) on the best backbone if frozen skill plateaus, since SciVid shows LoRA reaching about full fine-tune quality; (4) continued self-supervised pretraining on unlabeled WebCOOS video (EchoJEPA-style domain adaptation, wildfire-style per-camera priors). WebCOOS provides far more unlabeled than labelled clips, which is exactly the regime where domain SSL paid off in EchoJEPA.
- Add the controls from the probing papers to every run: frame-shuffled and single-frame-repeated inputs. If shuffled clips predict Hs as well as ordered ones, the model is using appearance (whitecap coverage, texture), not wave motion.

### Gaps
- No published study probes frozen video encoders for regression of a periodic quantity under cross-site splits in a geophysical setting; EchoJEPA (cardiac) is the closest analogue, and it did not test off-the-shelf V-JEPA 2.
- The "Attention, please! Revisiting attentive probing" line of work (efficient probes) was not read in this session.

## 5. Resolution handling: native resolutions vs 2688x1520 frames, and tiling or cropping to keep small breaking waves

### Takeaway
Native resolutions are 224-384 px squares: V-JEPA 2 256/384, V-JEPA 2.1 384, VideoPrism 288, InternVideo-Next/VideoMAE/Hiera 224, Cosmos-Embed1 224-448 with arbitrary aspect. Resizing a 2688x1520 frame to 256 makes each 16-px patch about 95 source pixels wide, which erases small distant breakers. Use ROI crops of the surf and breaking zone at 384, or a fixed tile grid at native scale, rather than full-frame resizing. RoPE-based V-JEPA 2 can take non-square grids, but at higher token cost.

### Cited Findings
- V-JEPA 2: 384-px evaluation and pretraining (ViT-g-384) gives SSv2 77.3 vs 75.3 at 256 and EK100 action R@5 39.7 vs 38.0. — [V-JEPA 2](https://arxiv.org/html/2506.09985). V-JEPA 2.1 trains its final phase at 384 px for video and 512 px for images. — [V-JEPA 2.1](https://arxiv.org/html/2603.14482)
- V-JEPA 2 uses 3D-RoPE, and the probe wrapper runs with `use_pos_embed: false`, `use_rope: true`. — [V-JEPA 2](https://arxiv.org/html/2506.09985); [eval config](https://github.com/facebookresearch/vjepa2/blob/main/configs/eval/vitg-384/diving48.yaml)
- Cosmos-Embed1 checkpoints "support arbitrary non-square resolutions", defaulting to 224/336/448p. — [Cosmos-Embed1 HF](https://huggingface.co/nvidia/Cosmos-Embed1-448p)
- VideoPrism public models take (num_frames, 288, 288, 3) inputs. — [VideoPrism card](https://huggingface.co/google/videoprism)
- Kamagata 2026 feeds 224x224 inputs with "saliency-guided cropping" biased "towards high energy areas, such as active surf and breaking regions", the saliency coming from Farneback optical-flow energy. — [Kamagata et al. 2026](https://arxiv.org/html/2607.11998v1)
- Wildfire smoke on fixed mountaintop cameras (native 1536x2048 or 2048x3072): frames split into a 4x4 grid of 224x224 tiles "to keep small plumes from being diluted by full-frame pooling", with overlapping tiles for FIgLib and a per-tile, per-camera calibrated score. — [Sensors 2026, 26:6020](https://pmc.ncbi.nlm.nih.gov/articles/PMC13667166)
- EchoJEPA narrowed crop-scale (0.5-1.0) and aspect-ratio (0.9-1.1) augmentation for its fixed-geometry domain. — [EchoJEPA](https://arxiv.org/html/2602.02603)

### Inferences
- Geometry: 2688x1520 has a 1.77 aspect ratio. Full-frame resize to 384 short side gives 679x384, i.e. 42x24 patches, each covering about 63x63 source px; a centre-crop to 256 discards 44% of the width and makes each patch about 95 source px. A 2688-wide frame resized to 224 (Kamagata-style) makes each patch about 190 px.
- Options, ordered by cost-effectiveness:
  - (a) A fixed per-camera ROI crop of the breaking and swash zone. The camera is fixed, so one crop box per view is chosen once, e.g. a 1152x1152 or 768x768 source region downsampled 2-3x to 384. This is also a cheap guard against the model reading sky, horizon or burned-in overlays.
  - (b) 2-4 ROI tiles (surf zone, outer break, swash) at 384, encoded separately and fused by the record-level attentive pooling. This costs 2-4x a single crop.
  - (c) The full frame at 384x672 (32,256 tokens; about 300 TFLOP per 64-f window for ViT-g, 122 for ViT-L). This is out of the pretraining distribution in aspect ratio and is expensive.
  - (d) A full tiling at native scale (about 7x4 tiles of 384) costs about 28x a single crop and is impractical for ViT-g over thousands of records.
- Random-crop and scale augmentation should be narrow (fixed geometry, as in EchoJEPA). For transfer to unseen sites, pixel geometry is site-specific, so including per-site metric scale (pixels per metre at the breaker line) as an input, or normalising crops to similar metric footprints, may matter more than raw resolution.
- WebCOOS H.264 at 2688x1520 must be fully decoded even when sampling 4 fps. Pre-transcoding each clip once to the ROI crop at 4 fps and at 10-20 fps (ffmpeg crop+scale) will likely dominate wall time less than repeated decoding during probe sweeps.

### Gaps
- No source evaluates V-JEPA 2/2.1 at non-square or above-384 resolutions, so quality at 384x672 is untested.
- No study quantifies how many pixels a breaking crest needs for detection by a frozen ViT (minimum object scale vs patch size) in water imagery.

## 6. Applications of video foundation models to fluid, ocean, coastal, weather or other geophysical video

### Takeaway
Direct applications to coastal video are few and weak. The two Kamagata 2026 preprints (V-JEPA ViT-S plus SlowFast for Hs/Tp/direction, and ViViT for Tp) are the only ones found; neither has instrument labels, and the first has an unclear backbone provenance. Rip-current, river-flow, swash and wave-breaking work still uses optical flow, CNNs and RNNs. The strongest transferable evidence comes from scientific benchmarks (SciVid: weather and cyclones; Polymathic: fluid PDEs; viscosity readout) and from domain-adapted JEPAs on other periodic or fixed-camera video (EchoJEPA, wildfire cameras).

### Cited Findings
- Kamagata et al. 2026 (arXiv:2607.11998, IEEE HPC-related venue). Backbone: "a pretrained Vision Transformer backbone trained with V-JEPA's latent predictive objective", named `vit_small_patch16_224`, checkpoint source not stated. Frozen vs fine-tuned is not explicitly stated, and there are no ablations of backbone, frames or resolution. Training-scene count is inconsistent within the paper ("66 scenes" vs "six"). Tested on 5 held-out plunging-breaker videos (Oak Island E/W, Xishawan). — [Kamagata et al. 2026](https://arxiv.org/html/2607.11998v1). Held-out Hs RMSE 0.586 m, R² 0.107, Pearson 0.436; Tp RMSE 1.42 s; direction RMSE 37°. The per-clip labels cannot be traced to an instrument. — [arXiv 2607.11998](https://arxiv.org/abs/2607.11998) (per the project's verification log `docs/literature/source_verification.md`)
- The official V-JEPA (2024) release lists only ViT-L/16 (224), ViT-H/16 (224) and ViT-H/16 (384) checkpoints. — [facebookresearch/jepa README](https://github.com/facebookresearch/jepa)
- Kamagata et al. (arXiv:2606.13302) estimate Tp only, using a ViViT backbone (LtViViT best), 60-frame windows, Airy-wave synthetic pretraining (>1,000 clips) and expert/timestack/optical-flow pseudo-labels on stock and live-stream footage. Held-out RMSE 0.80 s on 4 scenes; data "on request". — [arXiv 2606.13302](https://arxiv.org/abs/2606.13302) (details per the project's verification log)
- The nearshore wave-breaking video dataset has 9,000 labelled clips from 15 cameras across six coastal sites (spilling, plunging, surging), with a CNN+RNN baseline rather than a video foundation model. — [Yin et al. 2025, Sci. Data](https://www.nature.com/articles/s41597-025-06005-5) (search summary)
- Rip currents from UAV video: optical flow plus a CNN for water segmentation, no video foundation model. — [PMC13718950](https://pmc.ncbi.nlm.nih.gov/articles/PMC13718950) (search summary)
- River surface velocity: deep multi-frame optical flow (RivVideoFlow) with orthorectification, and RAFT on UAV river video (which underestimates up to 0.5 m/s in high-variance quartiles). No DINOv2, V-JEPA or VideoMAE application found. — [Water 2024, 16:2784](https://mdpi.com/2073-4441/16/19/2784); [IJAEOG 2024 via DOAJ](https://doaj.org/article/7d5590cb8c66490c97b5181e6d9cf298) (search summaries)
- Swash-zone waterline detection from UAV video applied to wave period measurement. — [Remote Sensing 18:1385 (2026)](https://doi.org/10.3390/rs18091385) (search summary)
- 3D ocean-wave free-surface reconstruction from stereo cameras with an attention-augmented pyramid network (no pretrained video backbone). Millimetre-level elevation in the central region, dominant-frequency error below 0.01 Hz. — [Liu et al. 2025, arXiv 2512.06024](https://arxiv.org/abs/2512.06024)
- SciVid (Google DeepMind, 2025) applies frozen general video encoders to weather (ERA5 WeatherBench2, 16 frames = 8 days; Digital Typhoon infrared, 12 hourly frames). 4DS-L is the best learned model on typhoon pressure; VideoMAE V2-g and 4DS-e are best on WB2 but remain far behind GraphCast, GenCast and HRES. Frame shuffling hurts weather forecasting second only to tracking. — [SciVid](https://arxiv.org/abs/2507.03578)
- Fluid PDEs (shear flow, Rayleigh-Benard convection): latent-prediction JEPA features are "more physically informative and more sample-efficient" than VideoMAE for parameter estimation. — [Qu et al. 2026](https://arxiv.org/abs/2603.13227)
- Viscosity of real liquids is readable from frozen video foundation-model features. — [Zhan et al. 2025](https://arxiv.org/abs/2510.02311)
- Wildfire smoke onset on 101 fixed HPWREN cameras: a 4.9M-param CNN plus factorised transformer with a self-supervised scene-dynamics prior (no foundation model). It transfers zero-shot to unseen camera networks (PyroNear-2025 recall 69.7% -> 72.8% after label-free recalibration). — [Sensors 2026, 26:6020](https://pmc.ncbi.nlm.nih.gov/articles/PMC13667166)
- EchoJEPA: a V-JEPA 2-style model adapted to a periodic-motion medical video domain beats domain baselines and generalises across sites. — [EchoJEPA](https://arxiv.org/html/2602.02603)
- The coastal-video literature itself flags "a lack of standardized evaluation protocols and publicly available expert-annotated coastal video benchmarks". — [arXiv 2606.13302](https://arxiv.org/html/2606.13302v1) (search summary)

### Inferences
- A WebCOOS clip-to-CDIP study with a frozen V-JEPA 2 / 4DS comparison under site-held-out splits would be, as far as these searches show, the first instrument-labelled evaluation of modern video foundation models on coastal wave video. That is publishable even with modest skill.
- The fixed-camera lessons from the wildfire paper carry over directly: per-camera conditioning or calibration, learning each camera's normal dynamics from unlabeled footage, and event-level evaluation.

### Gaps
- No application of any video foundation model to clouds and sky cameras, smoke plumes, rip currents or river flow was found in these searches. The searches were brief, so absence is not proof.
- DINO-world ("Back to the Features: DINO as a Foundation for Video World Models", arXiv 2507.19468) appeared in results but was not read.

## 7. Licence, weight availability, parameter counts, and memory/throughput on one H100/H200

### Takeaway
Commercially permissive, easily loaded weights exist for V-JEPA 2 (MIT/Apache-2.0, HF Transformers plus torch.hub), V-JEPA 2.1 (torch.hub; MIT repo), VideoPrism public B/L (Apache-2.0, JAX), InternVideo2-1B (Apache-2.0, gated), 4DS (Apache-2.0 repo, JAX) and Cosmos-Embed1 (NVIDIA Open Model License). VideoMAE/VideoMAE V2, Hiera, TimeSformer and V-JEPA v1 weights are CC BY-NC 4.0. On one H200, frozen extraction with ViT-L at 64 f/256 is cheap: hours for tens of thousands of 16-s windows. ViT-g-384 at 64 f is about 10x more and needs ROI crops. The practical bottleneck is storing token maps, not GPU memory.

### Cited Findings

| Model | Licence (weights unless noted) | Where | Params | Source |
|---|---|---|---|---|
| V-JEPA 2 ViT-L / ViT-H (fpc64, 256) | MIT | HF `facebook/vjepa2-vitl-fpc64-256`, `-vith-`; torch.hub `vjepa2_vit_large`/`_huge` | 326.0M / 653.9M | [HF API L](https://huggingface.co/api/models/facebook/vjepa2-vitl-fpc64-256); [HF API H](https://huggingface.co/api/models/facebook/vjepa2-vith-fpc64-256); [README](https://github.com/facebookresearch/vjepa2) |
| V-JEPA 2 ViT-g (256, 384) | Apache-2.0 | HF `facebook/vjepa2-vitg-fpc64-256`, `-384`; torch.hub `vjepa2_vit_giant(_384)` | 1.035B | [HF API g-384](https://huggingface.co/api/models/facebook/vjepa2-vitg-fpc64-384); [HF card](https://huggingface.co/facebook/vjepa2-vitg-fpc64-384) |
| V-JEPA 2 probe checkpoints (SSv2 L and g-384, Diving-48 L and g-384, EK100) | MIT | HF `...-ssv2`, `...-diving48`; README links | - | [HF API](https://huggingface.co/api/models/facebook/vjepa2-vitl-fpc32-256-diving48); [README](https://github.com/facebookresearch/vjepa2) |
| V-JEPA 2 / 2.1 code | MIT (3 files Apache-2.0) | GitHub | - | [README](https://github.com/facebookresearch/vjepa2); [GitHub API](https://api.github.com/repos/facebookresearch/vjepa2) |
| V-JEPA 2.1 B/L/g/G (384) | not separately stated | torch.hub `vjepa2_1_vit_base_384` ... `_gigantic_384` | 80M / 300M / 1B / 2B | [README](https://github.com/facebookresearch/vjepa2) |
| V-JEPA (2024) | CC BY-NC 4.0 | dl.fbaipublicfiles.com | about 200M / 630M (paper) | [jepa LICENSE](https://github.com/facebookresearch/jepa/blob/main/LICENSE); [V-JEPA](https://arxiv.org/html/2404.08471) |
| VideoMAE (v1) | CC BY-NC 4.0 | HF `MCG-NJU/videomae-*` | L 343M (HF) | [HF API](https://huggingface.co/api/models/MCG-NJU/videomae-large) |
| VideoMAE V2 | CC BY-NC 4.0 (code MIT) | HF `OpenGVLab/VideoMAEv2-Base` ... `-giant` | giant 1.026B, Base 86M | [HF API giant](https://huggingface.co/api/models/OpenGVLab/VideoMAEv2-giant); [GitHub API](https://api.github.com/repos/OpenGVLab/VideoMAEv2) |
| InternVideo2 Stage2 1B (224, f4) | Apache-2.0, gated (auto) | HF `OpenGVLab/InternVideo2-Stage2_1B-224p-f4` | 1B | [HF API](https://huggingface.co/api/models/OpenGVLab/InternVideo2-Stage2_1B-224p-f4) |
| UMT | MIT, gated (auto) | HF `OpenGVLab/UMT` | - | [HF API](https://huggingface.co/api/models/OpenGVLab/UMT) |
| VideoPrism public B / L (+ LvT text variants) | Apache-2.0 (other materials CC BY 4.0) | HF `google/videoprism-base-f16r288`, `-large-f8r288`; JAX/Flax | 114M / 354M | [VideoPrism card](https://huggingface.co/google/videoprism); [HF API](https://huggingface.co/api/models/google/videoprism-large-f8r288) |
| Hiera | CC BY-NC 4.0 (code Apache-2.0) | HF `facebook/hiera-*-hf` | Base 50.8M, Huge 670.7M | [HF API](https://huggingface.co/api/models/facebook/hiera-huge-224-hf); [GitHub API](https://api.github.com/repos/facebookresearch/hiera) |
| TimeSformer | CC BY-NC 4.0 | HF `facebook/timesformer-*` | 121M (B) | [HF API](https://huggingface.co/api/models/facebook/timesformer-base-finetuned-k400); [LICENSE](https://github.com/facebookresearch/TimeSformer/blob/main/LICENSE) |
| ViViT | MIT (HF tag); Scenic code Apache-2.0 | HF `google/vivit-b-16x2-kinetics400` | - | [HF API](https://huggingface.co/api/models/google/vivit-b-16x2-kinetics400) |
| SlowFast / X3D / MViTv2 (PySlowFast), VideoSwin | Apache-2.0 (code repos) | GitHub model zoos | - | [GitHub API SlowFast](https://api.github.com/repos/facebookresearch/SlowFast); [GitHub API VideoSwin](https://api.github.com/repos/SwinTransformer/Video-Swin-Transformer) |
| 4DS-B-dist-e / L / e | repo Apache-2.0 (weights not separately stated) | GCS .npz (JAX) | 88M / 305M / 3.8B | [representations4d README](https://github.com/google-deepmind/representations4d) |
| Cosmos-Embed1 224p/336p/448p | NVIDIA Open Model License ("ready for commercial use") | HF `nvidia/Cosmos-Embed1-*` | 448p 1.197B | [HF card](https://huggingface.co/nvidia/Cosmos-Embed1-448p); [HF API](https://huggingface.co/api/models/nvidia/Cosmos-Embed1-448p) |

- Meta's blog says V-JEPA 2 code and checkpoints are available "for commercial and research applications". — [Meta blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)
- HF Transformers loads V-JEPA 2 via `AutoModel` / `AutoVideoProcessor` with `model.get_vision_features(...)`; the card shows 64-frame sampling with torchcodec. — [HF card](https://huggingface.co/facebook/vjepa2-vitg-fpc64-384)
- Architecture dims (vjepa2 `vision_transformer.py`): ViT-L 1024x24 (16 heads), ViT-H 1280x32, ViT-g 1408x40 (22 heads, MLP ratio 48/11), ViT-G ("gigantic") 1664x48 (26 heads, MLP ratio 64/13). — [vjepa2 source](https://github.com/facebookresearch/vjepa2/blob/main/src/models/vision_transformer.py)
- Measured throughput (A100, fp16, 16x224 clips): ViT-B 47.1, ViT-L 17.8, ViT-H 11.3 clips/s; Hiera-B 133.6, Hiera-L 40.8, Hiera-H 20.9; MViTv2-L 20.5. — [Hiera Tables 1 and 17](https://arxiv.org/abs/2306.00989)
- V-JEPA 2 progressive-resolution training cut GPU time by 8.4x vs training at full resolution from scratch. — [V-JEPA 2](https://arxiv.org/html/2506.09985). The cooldown at 64 f/384 px runs a per-GPU batch of 6 with activation checkpointing; GPU type not stated in the config. — [cooldown config](https://github.com/facebookresearch/vjepa2/blob/main/configs/train/vitg16/cooldown-384px-64f.yaml)
- SciVid's whole frozen-backbone suite (all models, all five tasks) runs in under a day on one H100. — [SciVid](https://arxiv.org/abs/2507.03578)
- UMT pretraining took "6 days on 32 A100 GPUs". — [UMT](https://arxiv.org/abs/2303.16058)

### Inferences
- Throughput estimates use the FLOP table in section 3 and two bounds. The pessimistic bound scales Hiera's measured A100 ViT-L rate (17.8 clip/s x 1.19 TFLOP, about 21 TFLOP/s effective) by 2-3x for H200. The optimistic bound assumes 40% of an assumed about 1 PFLOP/s dense BF16 peak; NVIDIA's H100/H200 spec was not fetched in this session.
  - ViT-L, 64 f @ 256 (11.5 TF): about 0.03-0.3 s per 16-s window.
  - ViT-g, 64 f @ 384 (114 TF): about 0.3-2.7 s per window.
  - Example: 3,000 records x about 25 windows = 75k windows. That is about 0.6-6 GPU-hours for ViT-L 64 f/256, and about 6-56 GPU-hours for ViT-g-384 64 f, per crop.
- Memory, frozen inference: bf16 weights are about 0.65 GB (L) and 2.1 GB (g). With memory-efficient attention (the code uses xformers variants), live activations for ViT-g at 18,432 tokens are about 0.2-0.3 GB per layer per sample (the MLP intermediate 18,432 x 6,144 x 2 bytes = 226 MB). Batch size is limited by compute, not by the 141 GB.
- Memory, full fine-tune of ViT-g: AdamW states plus master weights are about 16 bytes/param, i.e. about 17 GB. Activations at 64 f/384 without checkpointing are roughly 20+ GB per sample (estimate). Batch 4-8 per H200 with activation checkpointing is plausible, consistent with Meta's batch 6 for the heavier pretraining step. LoRA reduces optimizer state to under 1 GB.
- **Feature caching is the real constraint.** One full token map for one layer, at bf16, is 18,432 x 1,408 x 2 bytes = 52 MB for ViT-g 64 f/384 and 8,192 x 1,024 x 2 bytes = 17 MB for ViT-L 64 f/256. For 75k windows that is about 3.9 TB and 1.3 TB per layer.
  - (a) Run the frozen backbone online during probe training (each epoch re-pays the extraction FLOPs).
  - (b) Cache spatially pooled tokens (e.g. 2x2 or 4x4 average-pooled, or per-time-step means: 32 x d per window, about 90 KB). This loses fine spatial detail.
  - (c) Cache a subset of layers at reduced precision. A pilot comparing (a) and (b) on one fold would settle it.
- On a shared H200, ViT-L or V-JEPA 2.1 ViT-L (distilled from ViT-G) at 64 f is the efficient workhorse for sweeps. Reserve ViT-g-384 and 4DS-e (3.8B) for final runs.
- For licences: if any commercial use of the pipeline is foreseeable, avoid VideoMAE/VideoMAE V2, Hiera, TimeSformer and V-JEPA v1 weights (CC BY-NC), or confine them to research-only comparisons.

### Gaps
- No published, measured V-JEPA 2/2.1 feature-extraction throughput or memory on H100/H200 was found; all numbers above are estimates.
- 4DS and V-JEPA 2.1 weight licences are not separately stated beyond their repositories' code licences.
- Whether `google/videoprism` has a working PyTorch/Transformers port was not verified. A Transformers documentation page for VideoPrism appeared in search results, while the model card lists only JAX/Flax.
