# Long-record aggregation and long-video architectures for sea-state estimation from shore video

Scope: how much sea-surface record an Hs/Tp estimate needs, how to pool many short-window embeddings into one record-level estimate, and which long-video model families (SSM/Mamba, memory transformers, token merging, streaming, recurrent heads) are usable on a shared H200. Encoders themselves (V-JEPA 2, VideoMAE V2, etc.) are out of scope; they appear only where their evaluation protocol bears on aggregation. Researched 2026-11-18. Items released June-November 2026 are tagged **[2026 Jun-Nov]**.

## Q1. How long a record (seconds / number of waves) does Hs and Tp estimation need statistically, and what does that imply for model input length?

### Takeaway
For a single-point record, Hs from a 17-min record is uncertain by about ±12% and peak frequency by about ±5% (90% confidence) (Donelan & Pierson 1983). Standard Gaussian-process theory, checked against those published figures, gives a 1-sigma Hs error of about 40-60% for one 16-s window and about 8-13% for 4-10 min. A single 16-s, 64-frame window cannot pin down Hs or Tp statistically. A model has to integrate minutes of video, by pooling windows or by using a long-context model. The 30-min buoy label itself carries an irreducible sampling error of about 4-5% (1 sigma).

### Cited Findings
- Donelan & Pierson (1983, JGR 88(C7):4381-4392), using laboratory data and a Lake Ontario tower, found that stationary quasi-Gaussian process theory predicts sampling variability accurately. For a 17-min record, "commonly used in wave monitoring at sea", the uncertainties in significant height and peak frequency are about **±12% and ±5%** at 90% confidence. They also found that the spectral peak is generally overestimated. — [EarthRef abstract record](https://earthref.org/ERR/46206/); [paper PDF (403 to automated fetch)](https://data-ww3.ifremer.fr/BIB/Donelan_Pierson_JGR1983.pdf)
- A later review (Bitner-Gregersen & Magnusson 2014) cites Donelan & Pierson as obtaining "8% variability" for Hs, and higher variability for very narrow spectra. It gives closed-form COV expressions for Hs under Pierson-Moskowitz and JONSWAP spectra as functions of peak frequency and record length. It notes that the Tucker formula does not capture the increase of COV with Hs seen in measurements. — [Bitner-Gregersen & Magnusson 2014, Ocean Dynamics (seen via search snippet; full text returned 403)](https://data-ww3.ifremer.fr/BIB/Bitner_Gregersen_Magnusson_OD2014.pdf)
- Worked example quoted in the literature: for a Pierson-Moskowitz spectrum with a 0.1 Hz peak frequency and a **1,024 s record**, the COV of Hs from Tucker (1992), as used by Forristall et al. (1996), is **about 5%**. This is roughly the value from Bitner-Gregersen & Hagen (1990). — [via search snippet of Bitner-Gregersen & Magnusson 2014; not read in full](https://data-ww3.ifremer.fr/BIB/Bitner_Gregersen_Magnusson_OD2014.pdf)
- Forristall et al. (1996, J. Waterway Port Coastal Ocean Eng. 122(5):216-225) found that finite-record sampling variability in Hs positively biases storm-peak Hs and extreme-value design heights, so a correct hindcast can look too low against measured peaks. They recommend continuous recording and comparisons against *smoothed* measured series. — [ASCE abstract](https://ascelibrary.com/doi/10.1061/(ASCE)0733-950X(1996)122:5(216))
- Measured Hs is usually computed from 17.5- or 30-min records. Shorter records give larger variability, and variability is higher for higher hourly Hs. — [search summary of related literature incl. EGU2013 abstract](https://meetingorganizer.copernicus.org/EGU2013/EGU2013-13980.pdf)
- A search-result summary reported empirical buoy Hs uncertainty of about 10.65% for NDBC and about 7.06% for CDIP Waverider, relative to mean Hs. This includes more than pure sampling error. **The originating study was not identified in the snippet, so treat these as unverified.** The same search surfaced De Carlo & Ardhuin (2024, JGR Oceans), which works through the sampling uncertainty of a 20-min buoy record and notes that variability grows with Hs. Its full text returned 403 or exceeded fetch size. — [De Carlo & Ardhuin 2024](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023JC020832?af=R)
- The two 2026 shore-video wave papers by Kamagata et al. use very short windows: 60 frames of about 2 s for Tp, "at least one wave cycle", in the June 2026 preprint. The July 2026 preprint reports Hs Pearson r = 0.451 and best R² = 0.246. — [arXiv 2606.13302](https://arxiv.org/html/2606.13302); [arXiv 2607.11998](https://arxiv.org/abs/2607.11998)
- The project's own design notes record that the current plan takes 16 s around each clip midpoint at 4 fps (64 frames), labelled from the CDIP 30-min record. The project doc describes Kamagata et al. 2026 as using 1.6-s windows (Hs RMSE 0.586 m, R² 0.107). — local file `docs/experiments/webcoos_video.md` (lines 10-30)

### Inferences
- **Theory used.** For a zero-mean Gaussian sea surface with one-sided spectrum S(f), the variance estimate m0 from a record of length T has CV²(m0) ≈ ∫S(f)²df / (T·m0²). Equivalently, the record has ν ≈ 2·T·B degrees of freedom, with equivalent bandwidth B = m0²/∫S²df. Then CV(Hs) ≈ CV(m0)/2, because Hs = 4√m0. This is the standard asymptotic result. I computed it numerically (scratch script, not a cited source) and checked it against the two published numbers above:
  - PM spectrum, fp = 0.1 Hz, T = 1,024 s gives **CV(Hs) = 4.7%**, against the published "about 5%".
  - JONSWAP (γ = 3.3), Tp = 10 s, T = 1,020 s gives CV = 6.1%, so a 90% interval of **±10%**, against Donelan & Pierson's ±12%.
- **Predicted Hs sampling CV (1 sigma), point record, JONSWAP γ = 3.3.** It is smaller for broader PM-like seas. For T = 16 s the asymptotic formula is only indicative.

  | Record T | Tp = 6 s | Tp = 10 s | Tp = 14 s | Approx. dof (Tp = 10 s) |
  |---|---|---|---|---|
  | 16 s (one V-JEPA 2 window) | 38% | 49% | 58% | about 2 |
  | 64 s | 19% | 24% | 29% | about 8 |
  | 240 s (4 min) | 10% | 13% | 15% | about 32 |
  | 600 s (10 min) | 6% | 8% | 9% | about 79 |
  | 1,800 s (30 min, buoy label) | 3.6% | 4.6% | 5.4% | about 237 |

- **Tp.** A single window of length T has a frequency resolution of 1/T. A 16-s window has bins at 0.0625 Hz spacing (periods 16 s, 8 s, 5.3 s ...), so on frequency content alone it cannot tell Tp = 10 s from 12 s. Reaching the ±5% peak-frequency precision Donelan & Pierson report at 17 min requires minutes of record. A 16-s window holds about 1-3 waves at Tp = 6-14 s, so Tp cannot be read off one window. Any Tp skill from one window must come from priors such as wavelength in the image, the dispersion relation, or the site's climatology, not from period statistics.
- **Camera vs point record.** A camera sees an area spanning many wavelengths, so spatial averaging adds effective degrees of freedom beyond the point-record figures above. The single-window error may therefore be smaller than the 16-s row suggests. Pixel intensity is not surface elevation, though, and nearshore transformation and the 20-30 km buoy offset add mismatch. No source quantifying camera-area degrees of freedom was found.
- **Implications for model input length:**
  - The model should integrate at least 4-10 min, ideally the whole available clip, which is about 240-600 s, or 15-37 non-overlapping 16-s windows.
  - Even a perfect estimator of the *nearshore, in-view* sea would show about 4-5% (1 sigma) Hs scatter against a 30-min buoy label from label sampling noise alone, before adding the spatial offset.
  - Forristall et al. imply that labels should be smoothed across adjacent buoy records (for example a centred 3-record mean) when used as targets or for evaluation.
- **The pooling design should be exchangeable over windows.** Hs is a variance statistic, so the record-level estimate is a pooled variance. Averaging per-window Hs² (energy), or log-energy features, is statistically closer to correct than averaging per-window Hs. The pooled head should be built to pool energy-like quantities.

### Gaps
- Could not read the full Donelan & Pierson (1983), Bitner-Gregersen & Magnusson (2014) or Forristall et al. (1996) papers (HTTP 403 or paywall), so their closed-form COV formulas and Tp/peak-period variability curves are not quoted directly. Tucker & Pitt (2001) and Goda's statistical-variability results (e.g., PARI 1985) were not accessible.
- No published source was found quantifying the extra degrees of freedom from spatially extended image or stereo observations versus point buoys for Hs.
- No source was found on how CDIP computes Tp from its 64-band spectrum (bandwidths 0.005-0.01 Hz), which sets the label's Tp quantisation.

## Q2. Which aggregation designs (MIL attention pooling, Perceiver, Set Transformer, hierarchical, recurrent) are proven for turning many clip embeddings into one video-level regression target, and with what evidence?

### Takeaway
All of these are well-established permutation-invariant or learned-query poolers with public MIT-licensed code: attention-MIL (Ilse 2018), Set Transformer PMA/ISAB, Perceiver cross-attention, and the attentive probe V-JEPA and EchoJEPA already use. The strongest direct evidence for physical-quantity video regression comes from echocardiography. Two recipes have worked there:
- simple averaging of 32-frame clip predictions per beat (EchoNet-Dynamic, R² 0.81);
- early fusion, which concatenates the tokens of many clips or views with clip and view embeddings into one frozen-encoder attentive probe (EchoJEPA 2026).

For waves, the only head-to-head (Kamagata 2026) found that a small ViViT beat ConvLSTM/LSTM on accuracy, while the recurrent heads were steadier in time.

### Cited Findings
- **Attention-based deep MIL (Ilse, Tomczak & Welling, ICML 2018).** A permutation-invariant pooling takes a weighted average of instance embeddings, with weights from a small attention network. It also has a gated variant (tanh × sigmoid), and the attention weights show which instances drive the bag label. It was evaluated on MUSK/Fox/Tiger/Elephant, MNIST-bags and histopathology. It became the standard aggregator for whole-slide imaging. The work is classification-framed, and regression needs a different head and loss. Code is MIT. — [arXiv 1802.04712](https://arxiv.org/pdf/1802.04712); [summary of follow-up adoption](https://www.emergentmind.com/topics/attention-based-multiple-instance-learning-aggregator); [AMLab-Amsterdam/AttentionDeepMIL (MIT, per GitHub API)](https://github.com/AMLab-Amsterdam/AttentionDeepMIL)
- **Set Transformer (Lee et al., ICML 2019).**
  - It introduces Pooling by Multihead Attention (PMA), in which k learnable seed vectors attend over the set features.
  - It also introduces inducing-point self-attention (ISAB), which reduces set self-attention from quadratic to linear in set size.
  - It frames multiple-instance learning as a set problem and argues that simple pooling under-fits.
  - Code is MIT.
  — [arXiv 1810.00825](https://arxiv.org/pdf/1810.00825v2); [juho-lee/set_transformer (MIT, per GitHub API)](https://github.com/juho-lee/set_transformer)
- **Perceiver (Jaegle et al., ICML 2021).** Asymmetric cross-attention distils a large input array into a small latent array, so cost is linear in input size and depth is decoupled from input size. It attends directly to 50,000 pixels to match ResNet-50/ViT on ImageNet, and is competitive on point clouds, audio, video and video+audio. The authors name the Set Transformer as the closest prior work. Perceiver IO (arXiv 2107.14795) adds output queries of arbitrary size. — [arXiv 2103.03206](https://arxiv.org/abs/2103.03206v2); [Wikipedia summary for Perceiver IO](https://en.wikipedia.org/wiki/Perceiver)
- **The V-JEPA 2 attentive probe is a PMA-style pooler.** It has 4 transformer blocks, and the last replaces self-attention with cross-attention from a learnable query: 1 query for classification, 3 for EK-100 verb/noun/action.
  - Multi-clip evaluation **averages the logits** of separately probed clips (late fusion). Examples: SSv2 16 frames × 2 clips × 3 crops; K400 16 × 8 × 3; COIN 32 × 8 × 3.
  - Pretraining on 64 rather than 16 frames gave +0.7 points on average, and **128 or 256 frames gave no further gain**.
  - The encoder uses 2×16×16 tubelets at 256².
  — [V-JEPA 2, arXiv 2506.09985](https://arxiv.org/html/2506.09985v1)
- **EchoJEPA (Feb 2026, arXiv 2602.02603).** It adapts V-JEPA 2 to echocardiography and keeps the backbone **frozen**.
  - The probe has depth 4 (three self-attention blocks, then cross-attention), 16 heads, a single learnable query and a linear head.
  - Multi-view and multi-clip aggregation is **early fusion**: tokens from all streams are concatenated with factorised view and clip embeddings, and a key-padding mask handles missing views. Late averaging appears only as an RVSP ablation.
  - LVEF MAE on the Stanford site: EchoJEPA-G 3.97, EchoJEPA-L 5.76, EchoPrime 4.87, PanEcho 5.10.
  - EchoJEPA-L (ViT-L, 300M) is pretrained on 525K public MIMIC-IV-Echo clips at 16 frames and 8 fps. EchoJEPA-G (ViT-g 1.1B, 18.1M proprietary videos) appears unreleased.
  - The paper states no weight licence; the GitHub repo is Apache-2.0 per the GitHub API (last push 2026-06-18).
  — [arXiv 2602.02603](https://arxiv.org/html/2602.02603); [bowang-lab/EchoJEPA](https://github.com/bowang-lab/EchoJEPA)
- **EchoNet-Dynamic (Ouyang et al., Nature 2020).** It regresses ejection fraction from 32-frame clips and averages clip estimates over every beat as test-time augmentation. Against expert annotation it reaches MAE 4.1%, RMSE 5.3% and R² 0.81. Code is MIT. Later EchoNet-Dynamic regressors are similar: CoReEcho MAE 3.90 and R² about 0.82; EchoVisionFM MAE 4.18 and R² 0.80. — [PMC8979576](https://pmc.ncbi.nlm.nih.gov/articles/PMC8979576); [CoReEcho arXiv 2403.10164](https://arxiv.org/html/2403.10164v2); [echonet/dynamic (MIT, per GitHub API)](https://github.com/echonet/dynamic)
- **Hierarchical / feature-based long-video heads.** ViS4mer (Islam & Bertasius, ECCV 2022) runs a ViT on short clips and a multi-scale S4 decoder over the clip features. It is 2.63× faster and uses 8× less GPU memory than its self-attention counterpart, and is best on 6 of 9 LVU tasks. Code is MIT. — [arXiv 2204.01692](https://arxiv.org/pdf/2204.01692v3.pdf); [md-mohaiminul/ViS4mer (MIT, per GitHub API)](https://github.com/md-mohaiminul/ViS4mer)
- **Recurrent heads in wave-from-video work:**
  - A Surfline surf-cam study found plain ConvNets poor for wave statistics, while a ConvLSTM "demonstrated significant improvement". Its ground truth was pressure sensors beside the camera. — [Stanford course report](https://galenegan.github.io/files/final_report.pdf)
  - An IEEE paper estimates Hs and fp from wave-impact video with CNN + ConvLSTM + self-attention memory. — [IEEE 10403732](https://ieeexplore.ieee.org/document/10403732)
  - An Ocean Engineering 2023 paper runs a CNN on snapshots, then LSTM+CNN over snapshot sequences, for wave-height classes. — [AMiner record](https://www.aminer.org/pub/648d4a50d68f896efac6eea2/wave-height-classification-via-deep-learning-using-monoscopic-ocean-videos)
- **Kamagata et al. 2026 [2026 Jun-Nov] (Tp from 2-s coastal video windows).**
  - Lightweight ViViT reaches test RMSE 0.80 s and MAE 0.56 s.
  - PtLSTM 1.22 s, WaveConvLSTM 1.86 s, attention-CNN+LSTM/GRU 1.41 s.
  - The authors say transformers were more accurate per window, while recurrent-convolutional models had higher temporal stability.
  - Window-to-record aggregation is not described, and one table shows MAE > RMSE, which is impossible.
  — [arXiv 2606.13302v2 (19 Jul 2026)](https://arxiv.org/html/2606.13302)

### Inferences
- **Recommended head.** Cache frozen per-window tokens, then pool over the whole record with a Set-Transformer/Perceiver-style head: a few self-attention blocks over all window tokens, then one or a few learned queries. This is the EchoJEPA early-fusion recipe with "view" replaced by "window index / time offset".
  - It generalises the current attentive probe from 1 window to 15-37 windows at negligible cost: 37 × 128 tokens is about 4.7k tokens.
  - It lets the model weight windows with glare, rain or occlusion down. Gated attention-MIL gives the same benefit in a cheaper form.
- **Baselines.** Two baselines should be reported beside it:
  - mean of per-window predictions in log space, which the project already does across seeds;
  - mean of per-window predicted Hs², the energy average, which follows from Q1.
  These are the EchoNet- and V-JEPA-2-style late-fusion baselines. A learned pooler has to beat them to justify itself.
- **Uncertainty weighting.** Train per-window heads to output a mean and a log-variance (heteroscedastic Gaussian NLL). Inverse-variance weighting of windows is then the statistically right pooling under the Q1 model. No wave-specific source tests this; it is a design suggestion.
- **No direct evidence yet favours one learned pooler over another for physical regression.** The evidence is: averaging works for EF; early-fusion attentive probes work for EF and RVSP; and a ViViT beats ConvLSTM per window for Tp. The choice between attention-MIL, PMA and Perceiver should be made empirically. With 3k-10k records, the simplest pooler that beats mean pooling is the safe pick.

### Gaps
- No published paper was found that applies attention-MIL, Set Transformer or Perceiver to pool clip embeddings into a *regression* target for physical or ocean video specifically.
- No published wave-from-video paper was found that pools over a full multi-minute record; the 2026 wave papers use 1.6-2 s windows.
- No head-to-head of xLSTM heads on video regression was found.

## Q3. Do Mamba/SSM video models beat transformer pooling on long-video understanding, at what memory cost, are weights/licences usable, and are there results on physical/periodic video?

### Takeaway
End-to-end SSM video models beat older *feature-based* transformer/S4 heads on Breakfast, COIN and LVU, and use far less memory than full space-time attention. VideoMamba-S/M scores 95-97% on Breakfast and 88-90% on COIN, against ViS4mer at 88.2 and 88.4. VideoMamba claims 40× less memory than TimeSformer at 64 frames. Pure Mamba models over-fit as they scale, and hybrids (VideoMAP, Vamba) are now the trend. There is no evidence that an SSM *pooling head* beats an attention pooling head over a modest number (tens) of cached clip embeddings. For periodic physical signals, Mamba heads on per-frame-pooled features work well for remote PPG over 2.6-60 s videos (RhythmMamba). VideoMamba, Vim, RhythmMamba, Vamba and ViS4mer weights or code are under permissive licences (Apache-2.0 or MIT). MeMViT and ToMe are CC-BY-NC-4.0, and MC-ViT has no official code.

### Cited Findings
- **VideoMamba (Li et al., ECCV 2024, arXiv 2403.06977).** It is a pure Mamba video backbone in three sizes: Ti 7M, S 26M, M 74M. A 98M Base was dropped because of sub-optimal optimisation, and larger pure-Mamba models **overfit**, which needed self-distillation from a smaller teacher.
  - K400 (IN-1K init): Ti 78.1-80.3, S 80.8-82.7, M 81.9-83.3. M with masked pretraining reaches 85.0 at 64 frames and 384².
  - **Long-video results (end-to-end, f32/f64):**

    | Model | Breakfast | COIN |
    |---|---|---|
    | VideoMamba-Ti | 94.3 / 94.3 | 86.2 / 87.0 |
    | VideoMamba-S | 95.3 / 97.4 | 88.4 / 88.7 |
    | VideoMamba-M | 94.8 / 95.8 | 88.3 / 89.5 |
    | VideoMamba-M† (masked pretraining) | 97.9 / 96.9 | 89.6 / 90.4 |
    | ViS4mer (feature-based) | 88.2 | 88.4 |
    | Distant Supervision (TimeSformer features) | 89.9 | 90.0 |
    | Turbo (end-to-end VideoMAE-B) | 86.8 | 82.3 |

  - **LVU (only Ti reported) vs ViS4mer:** Relation 62.5 vs 57.1, Scene 70.4 vs 67.4, Director 67.3 vs 62.6, Genre 65.2 vs 54.7, Writer 53.0 vs 48.8, Year 48.2 vs 44.8, Speaking 40.4 vs 40.8; View MSE 2.90 vs 3.63.
  - **Efficiency:** about 6× faster and about 40× less GPU memory than TimeSformer-Ti at 64 frames (A100-80G, batch 128, 224²). The absolute figures appear only in Fig. 1.
  - Weights are on HF (OpenGVLab/VideoMamba, **Apache-2.0**) and the GitHub code is Apache-2.0.
  — [arXiv 2403.06977v2](https://arxiv.org/html/2403.06977v2); [OpenGVLab/VideoMamba](https://github.com/OpenGVLab/VideoMamba); HF API licence check (apache-2.0)
- **VideoMambaPro ("Snakes and Ladders", arXiv 2406.19006).** It fixes "historical decay" and "element contradiction" with masked backward computation and elemental residuals. It gains +1.6-2.8% top-1 on K400 and +1.1-1.9% on SSv2 over VideoMamba. It has no long-video benchmarks in the abstract and no official code or weights link. Versions run from v1 (27 Jun 2024) to v4 (13 Nov 2024). — [arXiv abs](https://arxiv.org/abs/2406.19006)
- **VideoMAP (arXiv 2503.12332, 2025).** It is a hybrid of Mamba and Transformer layers at a **4:1** ratio, built to counter pure-Mamba overfitting. It pretrains by frame-wise autoregressive prediction of CLIP embeddings. It reports gains on K400, SSv2, Breakfast and COIN, and scaling to about 300M parameters without overfitting (secondary summary). Code is at yunzeliu/MAP (licence not checked). — [arXiv 2503.12332](https://arxiv.org/html/2503.12332v1); [memories.ai summary](https://memories.ai/research/videomap)
- **Video Mamba Suite (arXiv 2403.09626).** It evaluates 14 Mamba models and modules in four roles, including as a temporal module, across 12 tasks and 13 datasets, and reports "promising efficiency-performance trade-offs". It is an evaluation suite, not a new method. — [arXiv 2403.09626](https://arxiv.org/pdf/2403.09626)
- **S4-based models:**
  - ViS4mer: 2.63× faster and 8× less memory than self-attention; best on 6/9 LVU tasks; MIT code. — [arXiv 2204.01692](https://arxiv.org/pdf/2204.01692v3.pdf)
  - S5 (Wang et al., CVPR 2023) adds a mask generator guided by a momentum S4 that selects informative tokens. It gains up to +9.6% on LVU over the S4 method with 23% less memory; a secondary source says 25% at 60 frames. — [arXiv 2303.14526](https://ar5iv.labs.arxiv.org/html/2303.14526)
- **Vim (Vision Mamba) code** is Apache-2.0 (hustvl/Vim), and the **mamba** kernels are Apache-2.0 (state-spaces/mamba, active as of 2026-11-08). — [hustvl/Vim](https://github.com/hustvl/Vim); [state-spaces/mamba](https://github.com/state-spaces/mamba) (GitHub API licence checks)
- **Vamba (ICCV 2025, arXiv 2503.11579).** It is a hybrid Mamba-2/Transformer video LMM: Mamba-2 encodes the video tokens and the LLM is a transformer.
  - It encodes **more than 1,024 frames at 640×360 on a single GPU without token reduction**, against 256 frames for transformer LMMs.
  - Training and inference memory drop by ≥50%, and training steps run about 2× faster.
  - It gains +4.3% on LVBench (v1 abstract: 4.6%).
  - Code and HF weights (TIGER-Lab/Vamba-Qwen2-VL-7B) are **MIT**, and the repo was active in Oct 2026.
  - It is a 7B video-LLM, not a lightweight encoder.
  — [ICCV 2025 open access](https://openaccess.thecvf.com/content/ICCV2025/html/Ren_Vamba_Understanding_Hour-Long_Videos_with_Hybrid_Mamba-Transformers_ICCV_2025_paper.html); [HF papers page](https://huggingface.co/papers/2503.11579); [TIGER-AI-Lab/Vamba](https://github.com/TIGER-AI-Lab/Vamba)
- **Periodic physical video: remote photoplethysmography (rPPG) with Mamba.**
  - **RhythmMamba (AAAI 2025):**
    - Design: it pools each frame to one token (global average pooling) and runs Mamba over the T-token temporal sequence. It models the periodic signal's phase as state transitions, with multi-temporal-scale constraints and a frequency-domain feed-forward.
    - Efficiency: 319% throughput and 23% peak memory vs RhythmFormer (20.09 vs 6.30 kfps; 6.66 vs 29.06 "M" per frame; 128² input).
    - Results: heart-rate MAE 0.23 bpm on PURE, 0.50 on UBFC, 4.30 on VIPL-HR, 3.16 on MMPD.
    - Length test: tested at arbitrary lengths on MMPD, accuracy is roughly flat from 2.6 s to 60 s (MAE 3.10-3.51) but degrades at 1 s (MAE 5.50).
    - Code is MIT.
    — [arXiv 2404.06483v2](https://arxiv.org/html/2404.06483v2); [AAAI paper page](https://ojs.aaai.org/index.php/AAAI/article/view/33204); [zizheng-guo/RhythmMamba (MIT)](https://github.com/zizheng-guo/RhythmMamba)
  - PhysMamba (two separate 2024 papers): one uses Temporal-Difference Mamba (Chaoqi31/PhysMamba), and the other is a Mamba-2 dual-stream time-frequency model (arXiv 2408.01077). — [arXiv 2408.01077](https://arxiv.org/html/2408.01077v1); [arXiv 2409.12031](https://arxiv.org/abs/2409.12031v1)
- **Memory-augmented transformers:**
  - **MeMViT (CVPR 2022)** caches keys and values from earlier iterations as memory and processes video online. It gets **30× longer temporal support for 4.5% more compute**; naive extension would need >3,000% more. It is state of the art on AVA and EPIC-Kitchens-100 (2022), noting that most models handle <5 s of video. Code and models are **CC-BY-NC-4.0**. — [arXiv 2201.08383](https://arxiv.org/pdf/2201.08383.pdf); [facebookresearch/MeMViT](https://github.com/facebookresearch/MeMViT) (licence file checked)
  - **MC-ViT (ICML 2024):**
    - It uses 16-frame segments with self-attention inside each segment and cross-attention to a consolidated memory (k-means or coreset over past activations; K = 128-512 memories per segment).
    - Five segments × 512 memories (2,560 tokens) lost no accuracy on Diving48.
    - It matches joint space-time attention at about 10× less inference memory and FLOPs, and needs 3.5× less training memory.
    - Diving48: 89.7 (B) and 91.0 (L). EgoSchema subset/full: 61.2/42.3 (B) and 62.6/44.4 (L). Perception Test: 47.0 (B) and 48.1 (L).
    - It fine-tunes existing pretrained video ViTs with short schedules.
    - No official code or weights were found; there is only an unofficial reimplementation (kyegomez/MC-ViT).
    — [arXiv 2402.05861v2](https://arxiv.org/html/2402.05861v2); [unofficial repo](https://github.com/kyegomez/MC-ViT)

### Inferences
- **For this project, SSM vs attention matters mainly at the backbone level, not the pooling head.**
  - With a frozen V-JEPA 2 producing cached tokens, a whole 10-min record is only about 37 windows, or about 1,200 temporal steps if every 2-frame tubelet is kept. Either size is trivial for an attention head, so an SSM head brings no memory benefit at that scale.
  - Mamba's advantage shows up when (a) the backbone itself must see minutes of raw frames end-to-end (VideoMamba/Vamba style), or (b) per-frame features at the native 20 fps are streamed (12,000 steps per 10-min clip).
- **The rPPG evidence matches the wave case.** It is a periodic signal, a regression target, and pooled per-frame features run through a 1-D temporal model. It suggests a cheap complementary arm: per-frame (or per-tubelet) spatially pooled features, then a Mamba or GRU over the full 4-10 min at 4-20 fps. That arm sees every wave cycle in the record, unlike a 16-s window.
- **Practical caveats for end-to-end VideoMamba:**
  - VideoMamba-M is only 74M parameters and was pretrained on K400-scale data.
  - Training it end-to-end on 3k-10k records risks the same overfitting the authors report.
  - Its 224² input would downsample 2688×1520 frames heavily.
  - It would replace, not complement, the frozen V-JEPA 2 pipeline, so it should be a later ablation.
- **Licences:** VideoMamba, Vim, mamba, RhythmMamba, ViS4mer, Vamba, Set Transformer, AttentionDeepMIL and EchoNet are all usable for research and publication. MeMViT and ToMe are non-commercial (CC-BY-NC-4.0), which is fine for academic research but worth noting.

### Gaps
- No Mamba/SSM paper was found on ocean waves, fluids, weather video or vibration; the closest periodic-physical evidence is rPPG.
- No paper was found that directly compares an SSM pooling head with an attention pooling head over *cached frozen-encoder clip embeddings* for regression.
- VideoMamba's absolute memory and throughput values (GB, clips/s) at 64+ frames appear only in its Fig. 1 and were not extracted.
- No VideoMambaPro official weights were found.
- The VideoMAP licence and its Breakfast/COIN numbers were not verified.
- LVU clip durations were not verified from the LVU paper.

## Q4. What are the practical memory/throughput numbers at 64-1,000+ frames, and what would each option cost on an H200 with 16-117 GB free for 3,000-10,000 records of 4-10 min?

### Takeaway
Published ratios include:
- VideoMamba: 40× less memory and 6× faster than TimeSformer at 64 frames.
- ViS4mer: 8× less memory than self-attention.
- MC-ViT: about 10× less inference memory than joint attention.
- MeMViT: 30× temporal reach for +4.5% compute.
- ToMe: 2.2× throughput on video ViT-L for -0.2-0.3% accuracy.
- Vamba: more than 1,024 frames at 640×360 on one GPU.

For this project, the binding cost is running the frozen ViT-L over every 16-s window, plus decoding the 4-MP video, not the aggregator. Covering whole records multiplies backbone compute by the number of windows: about 15-37× more than one window per record. It is still single-digit GPU-hours on an H200, and memory fits easily within 16 GB free.

### Cited Findings
- V-JEPA 2 uses 2×16×16 tubelets at 256² input, so a 64-frame clip yields 32×16×16 = 8,192 tokens (arithmetic from the stated tubelet and resolution). — [arXiv 2506.09985](https://arxiv.org/html/2506.09985v1)
- A ViT-L JEPA encoder has about 300M parameters (EchoJEPA-L). — [arXiv 2602.02603](https://arxiv.org/html/2602.02603)
- **Measured in this project:** on the H200 smoke test (2026-11-18, 24 clips), feature extraction for all four arms took **about 1.5 s per clip**. The full sample comes to about 1.5 GPU-hours, and source clips are about 290 MB each (2592×1944). — local file `docs/experiments/webcoos_video.md` (Progress section)
- VideoMamba is about 6× faster and uses about 40× less GPU memory than TimeSformer-Ti for 64-frame video (A100-80G, batch 128, 224²). — [arXiv 2403.06977v2](https://arxiv.org/html/2403.06977v2)
- Vamba encodes more than 1,024 frames (640×360) on a single GPU without token reduction, while transformer LMMs reach 256 frames. It cuts memory by ≥50% and runs training steps about 2× faster. — [HF papers 2503.11579](https://huggingface.co/papers/2503.11579)
- MC-ViT is about 10× cheaper in inference memory and FLOPs than joint space-time attention at matched accuracy, and 3.5× cheaper in training memory with 3.2× fewer training FLOPs. Memory is bounded by segment size, and compute is linear in video length. — [arXiv 2402.05861v2](https://arxiv.org/html/2402.05861v2)
- MeMViT has 30× longer temporal support for +4.5% compute. — [arXiv 2201.08383](https://arxiv.org/pdf/2201.08383.pdf)
- ViS4mer is 2.63× faster with 8× less memory than self-attention, and S5 uses 23% less memory than ViS4mer. — [arXiv 2204.01692](https://arxiv.org/pdf/2204.01692v3.pdf); [arXiv 2303.14526](https://ar5iv.labs.arxiv.org/html/2303.14526)
- **Token merging:**
  - ToMe gives 2.2× throughput on video ViT-L (Spatiotemporal MAE, K400) for a 0.2-0.3% accuracy drop, and halves fine-tuning time.
  - A CVPR 2026 Findings paper (MaMe) reports VideoMAE-L at 6.25 videos/s baseline, 6.97 with ToMe (r = 32), and 9.28 with MaMe (81.47% accuracy). That is 48.5% faster for a 0.84% accuracy loss.
  - vid-TLDR is training-free, merges background tokens by saliency, and beats ToMe on video-text retrieval.
  - ToMe code is CC-BY-NC-4.0.
  — [ToMe arXiv 2210.09461](https://arxiv.org/pdf/2210.09461); [MaMe, CVPR 2026 Findings](https://openaccess.thecvf.com/content/CVPR2026F/papers/Huo_MaMe_Matrix-Based_Token_Merging_CVPRF_2026_paper.pdf); [vid-TLDR review](https://liner.com/review/vidtldr-training-free-token-merging-for-lightweight-video-transformer)
- RhythmMamba's cost grows linearly with video length because of Mamba plus per-frame pooling. — [arXiv 2404.06483v2](https://arxiv.org/html/2404.06483v2)

### Inferences
All figures below are back-of-envelope estimates; measure before relying on them.

- **One 64-frame window through frozen ViT-L.** About 300M parameters and 8,192 tokens give:
  - linear-layer FLOPs ≈ 2·3e8·8,192 ≈ 5 TFLOP;
  - attention FLOPs ≈ 4·N²·d·L = 4·8,192²·1,024·24 ≈ 6.6 TFLOP;
  - about 12 TFLOP per window.

  At an effective 300-400 TFLOP/s in bf16 on an H200, that is about 0.03-0.04 s per window. Inference memory is about 0.6 GB of weights plus well under 5 GB of activations at batch 8 with SDPA/FlashAttention, so it fits in 16 GB free.
- **Whole-record coverage with non-overlapping 16-s windows.** A record has 15-37 windows (4-10 min):
  - 3,000 records: about 45k-110k windows, roughly 0.5-1.2 GPU-h of encoder time;
  - 10,000 records: about 150k-370k windows, roughly 1.5-4 GPU-h.

  This is consistent with the measured about 1.5 s per clip for four arms on one window, which is likely dominated by decoding and I/O. **Decoding is the likely bottleneck.** A 10-min, 20 fps clip has 12,000 frames of 4-5 MP. Use GPU NVDEC decode (PyNvVideoCodec, torchcodec or DALI), or keep every 5th frame for 4 fps, and benchmark on one clip first.
- **Feature-cache sizes (bf16):**
  - full tokens: 8,192 × 1,024 × 2 B ≈ 16.8 MB per window, so 370k windows ≈ 6.2 TB (too large);
  - 8×4×4 pooled tokens × 4 layers ≈ 1 MB per window ≈ 370 GB (feasible on disk);
  - one token per tubelet-time (32 × 1,024) ≈ 64 KB per window ≈ 24 GB (fits in RAM).

  The record-level aggregator then trains on cached features in minutes on any GPU. 37 windows × 128 tokens is about 4.7k tokens per record, about 1 GB at batch 32.
- **Feeding a whole 10-min record (2,400 frames at 4 fps) to one ViT-L attention context.** That is about 307k tokens, and attention FLOPs ≈ 4·(3.07e5)²·1,024·24 ≈ 9 PFLOP per record, about 80× the windowed cost (about 30 s per record).
  - Inference memory is fine with FlashAttention.
  - Fine-tuning would need roughly 250 GB of activations without checkpointing, about 25-30 GB with per-layer checkpointing.
  - V-JEPA 2 showed no gain from 128-256-frame pretraining, so this path is costly with no evidence of benefit.
- **End-to-end SSM backbones** (VideoMamba-S/M at 224², 64-1,024 frames) would fit in 16-40 GB for training by the published 40× ratio. They would require retraining a weaker backbone on 3k-10k labels.
- **Token merging (ToMe-style, about 2× speed)** only matters if the encoder becomes the bottleneck, for example with 20 fps input or the ViT-g encoder.

### Gaps
- No published absolute GB or clips/s numbers were found for V-JEPA 2 ViT-L inference on H100/H200. The figures above are FLOP-count estimates.
- No measured NVDEC throughput was found for 2592×1944 or 2688×1520 H.264 on H200. This needs benchmarking.
- MC-ViT and VideoMamba memory curves exist only as figures.

## Q5. What 2025-2026 work (including June-November 2026 releases) on long-context video encoders or aggregation is relevant?

### Takeaway
The 2025-2026 trends are hybrid Mamba-Transformer models (VideoMAP, Vamba) and, above all, bounded-memory streaming designs for video-LLMs (StreamFlow, KeyRec, ReMem, Aug-Oct 2026). All are built for QA, not regression. The most relevant 2026 items for this project are:
- EchoJEPA (Feb 2026): a frozen V-JEPA 2 with an attentive probe and early fusion of many clips to regress a physical quantity;
- two Kamagata et al. coastal-video preprints (Jun/Jul 2026): short windows, weak Hs skill.

No June-November 2026 encoder was found that natively ingests minutes of video for dense regression.

### Cited Findings
- **EchoJEPA (arXiv 2602.02603, v4 10 Feb 2026).** It uses V-JEPA 2 pretrained on 18M echo videos, with a frozen backbone, an attentive probe and early fusion of views and clips. It reports about 20-27% better LVEF than baselines and drops only 2% under acoustic perturbations versus 17% for competitors. EchoJEPA-L is released (repo Apache-2.0). — [arXiv 2602.02603](https://arxiv.org/html/2602.02603); [emergentmind summary](https://www.emergentmind.com/topics/echojepa)
- **[2026 Jun-Nov] Kamagata et al. (2026), "Physics-Guided Spatiotemporal Learning for Coastal Wave Peak Period Estimation from Video".** arXiv 2606.13302; listed as submitted 11 Jun 2026, v2 19 Jul 2026.
  - It uses 2-s windows (60 frames) with a hinge physics penalty outside 2-20 s and sim-to-real transfer.
  - ViViT gives Tp test RMSE 0.80 s against ConvLSTM 1.86 s.
  - The data are 13 expert-labelled scenes with 6,926 windows.
  — [arXiv HTML](https://arxiv.org/html/2606.13302); [alphaXiv listing](https://www.alphaxiv.org/abs/2606.13302)
- **[2026 Jun-Nov] Kamagata et al. (2026), "HPC-Enabled Video-based Coastal Wave Parameter Estimation Using V-JEPA and Deep Spatiotemporal Learning".** arXiv 2607.11998, submitted 13 Jul 2026.
  - It combines V-JEPA ViT-S, a SlowFast dual stream, Farneback optical flow, and multi-task regression with an Airy-dispersion constraint (λp = 0.1).
  - It trained on 6 scenes on a DGX A100.
  - Pearson r: Hs 0.451, Hmax 0.578, Tp 0.643, Tz 0.680, direction 0.832. Best R² is 0.246.
  - Clip length and aggregation are not stated in the abstract.
  — [arXiv abs](https://arxiv.org/abs/2607.11998)
- **[2026 Jun-Nov] KeyRec: Bounded Visual Memory for Streaming and Long-Video Understanding.** arXiv 2609.32182, 26 Sep 2026, Google-affiliated authors.
  - It is training-free.
  - It keeps a recent-frame cache plus an event bank that grows by novelty-based add, merge and evict.
  - It uses about 10% of the dense visual-token budget and is best among compressed methods in 13 of 15 settings.
  - No code link was seen.
  — [arXiv abs](https://arxiv.org/abs/2609.32182)
- **[2026 Jun-Nov] StreamFlow** (arXiv 2608.10949, Aug 2026). It has a dynamics-aware mid-term memory that filters redundant frames *before* encoding, plus a latent long-term memory. It scores 67.73% on StreamingBench. — [arXiv abs](https://arxiv.org/abs/2608.10949v1)
- **[2026 Jun-Nov] ReMem** (arXiv 2610.05940, Oct 2026). It is training-free, with a fixed-size streaming context memory plus retrieved vision memory. — [arXiv PDF (seen in search results only)](https://arxiv.org/pdf/2610.05940)
- **[2026 Jun-Nov] Other titles seen only in search listings (not opened):**
  - GROVE (2608.02392)
  - StreamTTT (2608.13416)
  - Progressive Latent Memory Evolution (2609.04131)
  - Matryoshka frame selection (2608.05707)
  - CASE cost-aware stopping (2610.05400)

  All target video-LLM QA. — [GROVE](https://arxiv.org/pdf/2608.02392); [StreamTTT](https://arxiv.org/pdf/2608.13416); [2609.04131](https://arxiv.org/pdf/2609.04131); [2608.05707](https://arxiv.org/pdf/2608.05707); [2610.05400](https://arxiv.org/pdf/2610.05400)
- **[2026 Jun-Nov] arXiv 2606.29166 (June 2026).** It does self-supervised clustering of video encoding complexity on top of V-JEPA, at about 1.19% extra pretraining cost. It is not a long-context method. — [arXiv abs](https://arxiv.org/abs/2606.29166)
- **Earlier 2026:**
  - "Learning Compact Video Representations for Efficient Long-form Video Understanding" (arXiv 2602.17869, Feb 2026): an SSM-based compressor (linear complexity) with autoencoder pretraining, giving +2-4% across benchmarks. — [arXiv HTML](https://arxiv.org/html/2602.17869v1)
  - StateKV (arXiv 2605.31598, May 2026): linear-scaling inference for pretrained video VLMs. — [arXiv HTML](https://arxiv.org/html/2605.31598v1)
  - "Attend Before Attention: autoregressive gazing" (arXiv 2603.12254, Mar 2026). — [arXiv PDF](https://arxiv.org/pdf/2603.12254)
- **2025:**
  - Vamba (ICCV 2025; repo active Oct 2026; MIT). — [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Ren_Vamba_Understanding_Hour-Long_Videos_with_Hybrid_Mamba-Transformers_ICCV_2025_paper.html)
  - VideoMAP (2025; 4:1 hybrid). — [arXiv 2503.12332](https://arxiv.org/html/2503.12332v1)
  - V-JEPA 2 (Jun 2025; MIT repo, active Oct 2026). — [arXiv 2506.09985](https://arxiv.org/html/2506.09985v1); [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)

### Inferences
- **Transferable ideas from the 2026 streaming-memory work:**
  - Filter redundant frames *before* encoding (StreamFlow). For waves this could mean dropping frames with glare, rain or no visible sea.
  - Keep bounded event memory with novelty-based merging (KeyRec), which is essentially online coreset or k-means consolidation, as in MC-ViT.

  Both are designed for LLM question answering. For a scalar regression target with ≤37 windows per record, plain cached-feature pooling (Q2) is simpler and adequate.
- **The two 2026 coastal-video preprints are the closest prior art, and both work on very short windows.** One uses 2-s windows; the other's window length is not stated in its abstract, though the project doc records 1.6 s. Their weak Hs skill (R² ≤ 0.25) is consistent with the Q1 sampling argument. A record-level aggregator over 4-10 min is a clear, statistically motivated way to improve on them.

### Gaps
- No June-November 2026 *encoder* release was verified that targets minute-scale dense video regression. Most 2026 long-video work is video-LLM QA.
- Code or licence status for KeyRec, StreamFlow and ReMem was not established.
- Several Aug-Oct 2026 titles were seen only as search-listing titles and were not opened.
- No 2025-2026 xLSTM-based video model results were found.
