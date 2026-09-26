# Reproducibility

## Environment

- Python ≥ 3.11 (CI tests 3.11, 3.12, 3.13).
- Dependencies are locked in `uv.lock`. `uv sync --group dev` recreates the
  exact environment. Optional extras: `netcdf`, `imaging`, `ml`
  (`uv sync --extra ml`).
- For PyTorch without CUDA, install CPU wheels explicitly:
  `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

## Determinism

| Source of randomness | Control |
|---|---|
| Splits | `seed` parameter recorded in `SplitResult.params` |
| Bootstrap CIs | `seed` argument |
| Synthetic seas (`simulate_*`) | `seed` argument; random-amplitude model documented |
| Model training | seed in experiment config; record `torch.backends.cudnn.deterministic` in run metadata |

## Rebuilding from scratch

```bash
uv sync --group dev
uv run pytest                                  # offline test suite (fixtures are real, trimmed files)
uv run pytest -m network                       # live upstream-format checks
uv run wave-analysis ndbc inventory            # regenerates data/registry/ndbc_*.{parquet,csv}
uv run wave-analysis ndbc download 41010 --years 2023
uv run wave-analysis ndbc standardize 41010
uv run python scripts/validate_ndbc_hm0.py 41010 2023
```

Raw data are not in Git. They are regenerated from the manifests, and
checksums prove they are identical. If an upstream file has changed, the
download status says `changed`, and results depending on it must be re-run and
versioned.

## Versioning

- **Software:** Semantic Versioning (`pyproject.toml`), tagged releases, changes
  in `CHANGELOG.md`.
- **Datasets/manifests:** independent identifiers, e.g.
  `ndbc_registry_v0.1.0`, `ndbc_buoycam_hs_cross_station_v0`, so a software
  bug-fix release does not silently change a dataset version, and the reverse.
- **Configurations** are hashed (`config_sha256`) into every provenance record.

## Reporting checklist for results

A result is reportable only when it states:

- [ ] manifest name and version, split protocol, and leakage-audit outcome;
- [ ] reference instrument and H_s estimator of the labels;
- [ ] time-offset tolerance and distribution of |Δt|;
- [ ] baselines (mean, climatology, ridge) on the same split;
- [ ] metrics from `evaluation.regression_report` (and `circular_report` for
      directions), with dependence-aware bootstrap CIs;
- [ ] stratified results (station, illumination, sea state, season);
- [ ] Git commit (clean tree) and config hash.
