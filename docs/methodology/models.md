# Models

Deep learning is optional (`uv sync --extra ml`); baselines are mandatory
(ADR 0007). The reference implementations are for **controlled replication and
ablation on this project's data and splits**, not claims of reproducing the
papers' numbers, whose data are unavailable or simulated.

## Statistical baselines (`models/baselines/statistical.py`)

| Baseline | What it tests |
|---|---|
| `MeanBaseline` | The R² = 0 floor |
| `ClimatologyBaseline(["station_id", "month"])` | Site and season climatology, the natural competitor in cross-time tests |
| `PersistenceBaseline` | Causal last-known reference value (gap-filling or nowcasting framing) |
| `RidgeBaseline` | Whether simple covariates (image statistics, wind) already explain the target |

## Reference architectures (torch)

| Module | Paper | Structure | Unverified details (exposed as parameters) |
|---|---|---|---|
| `baselines/choi2020.py::Choi2020Regressor` | Choi et al. 2020 | frame differences → VGG-16 trunk → BiConvLSTM (256, 3×3) → GAP → FC → H_s | FC widths, resolution, optimiser |
| `baselines/kim2023.py::Kim2023SnapshotCNN` | Kim et al. 2023 | 4 conv blocks (16-32-64-64) → FC → softmax | kernel sizes, FC width |
| `baselines/kim2023.py::Kim2023CNNBiLSTM` | Kim et al. 2023 | GoogLeNet features → BiLSTM(2000) → dropout 0.5 → FC | frozen vs fine-tuned encoder |
| `multimodal/yang2026.py::Yang2026Fusion` | Yang et al. 2026 | ResNet-18 + sin/cos attitude MLP → fusion → sinusoidal PE → transformer encoder → mean pool → MLP; `sliding_window_predict` | fusion operator (concat vs sum), depth, heads |
| `multimodal/slowfast_head.py::SlowFastWaveHead` | Kamagata et al. 2026 (verified V1) | fast (16 frames, 4 heads) + slow (α = 4, 8 heads) transformers, 2 layers, d = 384 → Linear → MLP 384-128-5 (GELU) → 5·Softplus | frame encoder supplied externally (V-JEPA weights not bundled) |

Improvement options (off by default, so the paper configurations stay
reproducible): per-target output scales and a sin/cos direction head in
`SlowFastWaveHead`; image-only mode in `Yang2026Fusion` (ablation M3/M4-style).

## Physics-informed losses (`models/losses.py`)

| Loss | Constraint | Status |
|---|---|---|
| `kamagata_physics_loss` | \|log(T_p²+ε) − log(6H_s+ε)\| | Exact reproduction. **Implies fixed s_p ≈ 0.107**, an unrealistically steep prior (source_verification.md) |
| `steepness_band_loss` | hinge on log s_p outside [0.005, 0.07] | project alternative: penalises only the implausible |
| `ordering_loss` | H_s ≤ H_max; T_m02 ≤ T_m01 | always true for one record |
| `spectral_consistency_loss` | H_s = 4√m₀(Ŝ) | for joint spectrum + bulk heads |
| `circular_loss` | 1 − cos(Δθ) | periodic direction loss |

Research question (research_gaps.md Q-G4): *which physical priors improve
cross-station generalisation, not just in-distribution fit?*

## Recommended experimental ladder

1. Baselines on the cross-station split.
2. Single-frame CNN (snapshot).
3. Temporal model on the same frames (the effect of temporal context, as in
   Kim et al. 2023 and Yang et al. 2026's M1→M3).
4. Multimodal inputs (wind, attitude), declared as such.
5. Self-supervised pretraining on unlabelled video, with test stations excluded
   from the pretraining corpus.
6. Physics priors, ablated one at a time.

Each step changes one factor and is evaluated with the full reporting set
(`evaluation.md`).
