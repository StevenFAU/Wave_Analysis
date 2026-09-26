# Source Verification Log

This log records how each literature claim used by the project was checked
against primary sources, and where the project's technical summary
(`Wave_Analysis_Project_Technical_Summary.md`) needed correction or caveats.
It exists so that no number in this repository has unknown provenance.

**Verification levels**

| Level | Meaning |
|---|---|
| **V1 – primary full text** | Checked against the full text of the paper (e.g. arXiv PDF). |
| **V2 – primary abstract/metadata** | Checked against the publisher abstract, Crossref/Semantic Scholar metadata, or the authors' dataset record. |
| **V3 – project notes only** | Taken from the project technical summary (prepared from the full text), not re-checked because the full text is paywalled. Must be re-checked before being quoted in a publication. |

Last full review: **2026-09-25**.

---

## Kamagata et al. (2026) — arXiv:2607.11998 · **V1**

Checked against the arXiv v1 PDF (the HTML rendering has text-extraction errors;
see below).

| Claim in technical summary | Status | Primary-source finding |
|---|---|---|
| Five targets Hs, Hmax, Tp, Tz, θ | ✅ confirmed | |
| V-JEPA ViT-Small, F ∈ ℝ^{B×T×384}, 224×224 input | ✅ confirmed | Backbone `vit_small_patch16_224`. |
| SlowFast: 2 layers, d=384, dropout 0.1, fast 16 frames/4 heads, slow α=4/8 heads | ✅ confirmed | The arXiv **HTML** renders "2 layers" as "22 layers" and "6 scenes" as "66 scenes"; the PDF is authoritative. |
| MLP 384→128→5, ŷ = 5·Softplus(·) | ✅ confirmed | Hidden activation GELU. |
| L = L_MSE + λ_p L_phys, λ_p = 0.1, ε = 1e-4, T_p² vs 6 H_s | ✅ confirmed | See physical interpretation below. |
| "~six annotated training scenes" | ✅ confirmed | 6 scenes → 894 clip windows (16 frames, stride 8, 10 fps). Training H_s range 0.85–2.80 m. |
| PCC / RMSE / MAE / R² table | ⚠️ **caveat** | The summary's table is the paper's **"full-set"** Table I, evaluated on all 894 windows, which **include the training data**. The genuinely held-out results are Table II (5 plunging-breaker videos): H_s RMSE 0.586 m, R² 0.107, PCC 0.436; T_p RMSE 1.422 s, R² 0.203; direction RMSE 36.98°, R² 0.158. Cite Table II for generalisation claims. |
| Ground truth: "annotated wave-parameter dataset" | ⚠️ **unresolved** | The paper does not state how per-clip H_s/T_p/θ labels were obtained. Its only cited video source, Yin et al. (2025, *Sci. Data* 12:1722), provides **breaking-type labels only** (spilling/plunging/surging); that paper's site wave conditions come from **ERA5 reanalysis climatology**, not co-located instruments. Kamagata's labels therefore cannot currently be traced to a physical sensor. |
| NRMSE definition | ➕ added | NRMSE(H_s) = 28.81 % equals RMSE / (training range 2.80 − 0.85 m). Range-normalised NRMSE is not comparable to mean-normalised NRMSE or the scatter index. |
| Train/validation split | ➕ added | Not described; early stopping at epoch 31 on an unspecified validation set. |
| Code / data availability | ➕ added | No statement in the paper. |

**Physical interpretation of the regulariser (project analysis).**
T_p² = 6 H_s is equivalent to a fixed deep-water peak steepness
s_p = 2πH_s/(gT_p²) = 2π/(6g) ≈ 0.107. That is steeper than the steepest
observed wind seas (s_p ≲ 0.05–0.07) and 3–10× typical swell values. For
H_s = 2 m it pulls T_p toward about 3.5 s. The paper calls this term a
dispersion-relation constraint, but it is a fixed-steepness prior. It is
consistent with the reported negative T_p bias (−0.43 s), though that does not
prove it causes it. See `src/wave_analysis/models/losses.py` for the exact
reproduction and for band-limited alternatives.

## Choi et al. (2020) — *Ocean Eng.* 201, 107129 · **V2** (abstract) / **V3** (details)

Crossref and Semantic Scholar metadata confirm the title, all seven authors
(H. Choi, M. Park, G. Son, J. Jeong, J. Park, K. Mo, P. Kang), volume/article
and date (April 2020). The abstract confirms:

- CNN classification of SWH level from a single raw image, with four CNN
  structures and two performance-improvement methods; **84 %** accuracy on the
  NDBC dataset;
- ConvLSTM regression from sequential images; **MSE 0.0177** on a dataset of
  serial images captured from a **container ship**;
- motivation: fuel-efficient ship routing.

Details held at **V3** (full text paywalled): 20 NDBC buoys; 44,620 image/label
pairs; images about 10 min past the hour and wave records about 40 min past
(30-minute offset); thresholds of 1.52 m and 3.05 m; brightness-based
exclusion; 97,458 ship images at 1 Hz, 3,019 16-frame clips (2,675/344);
radar reference; frame differences; VGG encoder; BiConvLSTM with 256 filters.

*Observation:* 1.52 m and 3.05 m are exactly 5 ft and 10 ft, so the class
thresholds were probably defined in imperial units.

*Consistency check with NDBC documentation (V1, NDBC acq.shtml):* camera
images in the current NDBC buoycam listing are stamped at minute 10
(e.g. `W04A_2026_09_26_0010.jpg`). This is consistent with the "10 minutes past
the hour" image timing.

**Errata to the technical summary.** §5.1 writes the ConvLSTM candidate state
with a sigmoid. The standard formulation (Shi et al. 2015) uses **tanh**, which
is what the project implements. Check the paper's own equation if you need an
exact replication.

## Kim, Cho & Lee (2023) — *Ocean Eng.* 288, 116002 · **V2** / **V3**

Crossref confirms the title, authors (Y.-H. Kim, S. Cho, P.-S. Lee),
volume/article and date (November 2023). The SSRN preprint (id 4349127) and
ScienceDirect abstracts confirm: a monoscopic camera; wave height measured by
an **ADCP installed in the southwestern area of Korea**; single-snapshot CNN
estimation of sea state and average wave height; then CNN+LSTM on sequential
snapshots; data augmentation found effective.

Details held at **V3**: Jeju Island wave-energy-converter site; Nortek AWAC at
~18 m; 07:00–11:00 collection; grayscale; frames every 0.5 s; five classes
0.75–3.25 m in 0.5 m bins; 180 s sequences best; precision/recall/accuracy
0.783 → 0.904 → 0.932; Bi-LSTM with 2,000 hidden units; the inconsistent
data-availability statement.

## Yang et al. (2026) — *Ocean Eng.* 352, 124517 · **V2** / **V3**

Crossref confirms the title, authors (X. Yang, H. Wang, R. Zhang, S. Sun,
M. Zhang), volume/article and date (2026). Publisher abstract snippets confirm:
an attitude-aware spatiotemporal framework for shipborne SWH; a time-synchronised
dataset of sequential images and IMU attitude, **with SWH measured by (virtual)
wave buoys as labels**; multi-head self-attention for cross-modal dependencies;
a digital-twin dataset of **47 sea states**; robustness on unseen states and
seas above 2 m; sliding-window real-time inference.

Details held at **V3**: ResNet-18; sin/cos attitude encoding; ~4,700 pairs at
2 Hz; zero-upcrossing and spectral labels agreeing within about 0.05 m; the
M1–M6 ablation table; MAE 0.025 m / RMSE 0.032 m / R² 0.994; 7–9 frame optimum.
All reported metrics are **in simulation**.

## Related work identified during verification

- **Yin et al. (2025)**, *Sci. Data* 12:1722, doi:10.1038/s41597-025-06005-5; data at
  doi:10.6084/m9.figshare.28814993 (**CC BY-NC-ND 4.0**). 9,000 breaking-type clips
  from 15 cameras at 6 sites, including Currituck, Jennette's Pier and Oak Island,
  NC. The camera identifiers match WebCOOS naming, so these U.S. sites could be
  paired with nearby instruments. Registered as `yin2025_vwbt`.
- **arXiv:2606.13302**, *Physics-Guided Spatiotemporal Learning for Coastal Wave
  Peak Period Estimation from Video*: found in search results next to Kamagata
  et al.; not yet reviewed (open question L-1 in `research_gaps.md`).
