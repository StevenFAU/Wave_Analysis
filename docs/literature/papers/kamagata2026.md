# Kamagata et al. (2026): V-JEPA + SlowFast + physics loss, five wave parameters

**Citation:** Kamagata, A. H., Jat, D. S., Gamundani, A. M., Paramasivam, S.,
Sani, B., & Zakariyya, A. (2026). HPC-Enabled Video-based Coastal Wave Parameter
Estimation Using V-JEPA and Deep Spatiotemporal Learning. arXiv:2607.11998.
· key `Kamagata2026`

**Verification:** **V1** (arXiv v1 PDF). The HTML rendering garbles numbers
("22 layers", "66 scenes"); the PDF says 2 and 6.

## Setup
Monocular coastal video → [H_s, H_max, T_p, T_z, θ]. **6 annotated scenes →
894 clip windows** (16 frames, stride 8, 10 fps). Training H_s range
0.85–2.80 m. Held-out test: **5 plunging-breaker videos** (Oak Island East/West,
NC; Xishawan C4, China) from Yin et al. (2025). Hardware: NVIDIA DGX A100
(8 × 40 GB). AdamW, 50 epochs, early stop at 31, seed 42.

## Method
Pretrained V-JEPA ViT-S (`vit_small_patch16_224`), frame embeddings in ℝ³⁸⁴.
Farnebäck optical-flow saliency cropping. SlowFast transformer (2 layers,
d = 384, dropout 0.1; fast: 16 frames, 4 heads; slow: every 4th frame, 8 heads)
→ concat → Linear(384). MLP 384→128 (GELU)→5, output 5·Softplus.
Loss = MSE + 0.1·|log(T_p²+ε) − log(6H_s+ε)|.

## Results
| | Full set (includes training) | **Held-out test** |
|---|---|---|
| H_s RMSE / R² / PCC | 0.562 m / 0.134 / 0.451 | **0.586 m / 0.107 / 0.436** |
| H_max | 0.992 m / 0.121 / 0.578 | 1.035 m / 0.096 / 0.568 |
| T_p | 1.412 s / 0.246 / 0.643 | 1.422 s / 0.203 / 0.629 |
| T_z | 0.975 s / 0.226 / 0.680 | 0.982 s / 0.163 / 0.669 |
| θ | 33.99° / 0.193 / 0.832 | 36.98° / 0.158 / 0.839 |

H_s regression slope 0.12 (regression dilution). Direction bias +17.9°,
attributed by the authors to missing camera-bearing calibration.
NRMSE = RMSE / training range.

## Critical notes (project analysis)
1. **Label provenance is not stated.** The cited video source provides
   breaking-type labels only; its wave climate comes from ERA5.
2. **Full-set metrics include training data.** Cite the held-out table.
3. **The "dispersion" loss implies a fixed steepness** s_p ≈ 0.107, far steeper
   than real seas (`models/losses.py`).
4. Direction metrics (RMSE, R², PCC in degrees) appear to be linear. No circular statistics are described, which matters near 0°/360°.
5. The validation split for early stopping is not described.
6. No code or data availability statement.

## Consequences for Wave_Analysis
- Architecture has outpaced labelled data (hypothesis H2 in the README).
- Reproducible modules: `models/multimodal/slowfast_head.py`,
  `models/losses.py::kamagata_physics_loss`.
- WebCOOS camera naming in the test set suggests a route to instrument labels
  (`docs/datasets/webcoos.md`).
