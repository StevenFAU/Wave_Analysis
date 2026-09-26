# Example: offline end-to-end pipeline on real fixture data

`pipeline_demo.py` runs the whole v0.1 chain **without network access**, on the
real NDBC 41010 fixtures (1–5 March 2023):

1. parse the directional spectra and derive H_m0, T_p, and D_m;
2. create a *mock* image table at minute 10 of each hour, as NDBC buoycams
   would produce, with simulated per-frame brightness;
3. pair images with spectra (`pair_nearest`), recording Δt and sync flags;
4. annotate illumination from solar geometry;
5. build a manifest with declarative filters and an exclusion ledger;
6. split by day with an embargo, and audit for leakage;
7. evaluate climatology and ridge baselines with a block-bootstrap CI.

The images are mocked: there is no public NDBC image archive for 2023. This
demonstrates the *mechanics*. Run `wave-analysis ndbc cameras` hourly to
collect real images (`docs/datasets/ndbc.md`).

```bash
uv run python examples/synchronized_sample/pipeline_demo.py
```
