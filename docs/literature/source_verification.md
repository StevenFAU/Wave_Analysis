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
- **arXiv:2606.13302** (Kamagata, Jat, Gamundani, Srivastava, Saravanakumar;
  v1 2026-06-11, v2 2026-07-19), *Physics-Guided Spatiotemporal Learning for
  Coastal Wave Peak Period Estimation from Video*. **V1** (arXiv PDF, read
  2026-09-26):
  - Target is T_p only (2–20 s).
  - Gold set: 13 scenes (6,926 windows of 60 frames). Sources are a GitHub
    wave-tracking repository, a Zenodo storm-video record
    (doi:10.5281/zenodo.4295611), Surfline live streams, and the team's own
    Namibian recordings.
  - Gold labels: "expert visual assessment" combined with timestack analysis.
    A hybrid target uses the expert value where available and otherwise the
    FFT peak of the ROI mean intensity (0.05–0.5 Hz).
  - Silver set: 20 scenes (10,655 windows) from stock-footage libraries
    (Pixabay, iStock, Vecteezy, Adobe Stock) with optical-flow pseudo-labels.
  - Synthetic pretraining: more than 1,000 Airy-wave clips.
  - Evaluation: scene-level split, 9 train/validation scenes and 4 held-out.
    Held-out RMSE is 0.80 s (LtViViT) and 1.41 s (PtAttnCNN), SI about 0.1.
  - Data and code availability: "available from the corresponding author on
    reasonable request".
  - **No instrument ground truth**, so G-2 applies.

## NDBC wave processing (Earle 2003) — NDBC TD 03-01 · **V1**

Read in full on 2026-10-01 from <https://www.ndbc.noaa.gov/wavemeas.pdf> (55
pages). The repository previously cited TD 96-01 (Earle 1996). TD 03-01
(January 2003) is its update. It describes the 1996 edition as now out of
date and says it should no longer be used, so the citation is now `Earle2003`.

| Fact used by the project | Finding |
|---|---|
| Spectra are computed on board | "Transmission of raw time series data is not feasible" over satellite links, so "much of the data analysis is performed onboard" and key parameters are relayed to shore, where further processing and QC are done. **No raw heave, pitch or roll series is published.** This is why no NDBC record gives a camera's attitude at the moment of an image. |
| Sensors | Fixed accelerometers, Datawell HIPPY 40 (heave, pitch, roll), 3-axis magnetometers, angular-rate and tilt sensors, mast-axis inclinometers, depending on the payload (GSBP, DACT, VEEP, WPM, DWPM, NDWPM; Table 1) |
| Record length and sampling | 20 min (1200 s) for most systems, 40 min for one; sampling 1.50, 2.00, 2.56 or 1.7066 Hz by system; non-directional spectra cut off at 0.40–0.50 Hz and directional spectra at 0.35 Hz on some systems |
| H_m0 | 4√m₀ with m₀ summed over the bands of the non-directional spectrum, consistent with the WVHT = H_m0 check (`docs/datasets/ndbc.md`) |
| H_m0 vs H₁/₃ | H_m0 values "typically being about 5% to 10% greater" than H₁/₃ (citing Longuet-Higgins 1980) |
| Uncertainty of one record | 90 % confidence intervals for H_m0 are "generally about −10% to +15%". NDBC does not publish per-record intervals |
| Peak period, direction | T_p is the centre of the band with the most energy. NDBC's "mean wave direction" is the mean direction of that band, which others call the dominant direction |
| Scope | Payloads up to 2003. Systems deployed later are not covered, so conventions must still be checked against data (as done for WVHT) |

## Landscape review sources (2026-09-26)

The broader review ([landscape_review.md](landscape_review.md)) adds two
levels: **S** (screened by title only, not yet evidence) and **Q** (a data
service queried programmatically). DOIs, author lists, volumes, and pages
of every bibliography entry it added were checked against the Crossref API on
2026-09-26.

## Verification index

One row per bibliography key whose verification is recorded above or in the
landscape review. The dashboard reads this table, and a test checks that
every key exists in [bibliography.bib](bibliography.bib). Keys not listed
(textbooks, standards, architecture papers) have no logged check yet.
`V1/V2` means the review read the source but did not record which of
the two levels applies.

| Key | Level | Basis |
|---|---|---|
| Kamagata2026 | V1 | arXiv v1 PDF; see the section above |
| Kamagata2026Tp | V1 | arXiv:2606.13302 PDF, read 2026-09-26 |
| Choi2020 | V2/V3 | Crossref and Semantic Scholar abstract; details from project notes |
| Kim2023 | V2/V3 | publisher abstract; details from project notes |
| Yang2026 | V2/V3 | publisher abstract; details from project notes |
| Yin2025 | V1 | *Sci. Data* article and figshare record |
| Guimaraes2020 | V1 | landscape review, Tier 1 table |
| Kwon2023 | V1 | landscape review, Tier 1 table |
| Buscombe2020 | V2 | landscape review, Tier 1 table |
| Stringari2021 | V2 | landscape review, Tier 2 table |
| Vorkapic2024 | V2 | landscape review, Tier 2 table |
| Nanda2024 | V2 | landscape review, Tier 2 table (KOLOMVERSE) |
| Quach2021 | V2 | landscape review, satellite table (Sentinel-1 WV) |
| CienfuegosZenodo2025 | V2 | Zenodo record 17192781 |
| SalinZenodo2023 | V2 | Zenodo record 7942014 |
| BuscombeCarini2019 | V1/V2 | landscape review §8; Crossref metadata 2026-09-26 |
| Brumer2017 | V1/V2 | landscape review §8; Crossref metadata 2026-09-26 |
| SchwendemanThomson2015a | V1/V2 | landscape review §8; Crossref metadata 2026-09-26 |
| SchwendemanThomson2015b | V1/V2 | landscape review §8; Crossref metadata 2026-09-26 |
| HolmanHaller2013 | V1/V2 | landscape review §8; Crossref metadata 2026-09-26 |
| Holman2013cBathy | V1/V2 | landscape review §8; Crossref metadata 2026-09-26 |
| Almar2012 | V1/V2 | landscape review §8; Crossref metadata 2026-09-26 |
| Kudryavtsev2017 | V1/V2 | landscape review §8; Crossref metadata 2026-09-26 |
| Bodnar2025 | V1/V2 | landscape review §8; Crossref metadata 2026-09-26 |
| Assran2025 | V1/V2 | landscape review §8; arXiv record |
| Earle2003 | V1 | NDBC wavemeas.pdf (TD 03-01), read 2026-10-01; see the section above |
