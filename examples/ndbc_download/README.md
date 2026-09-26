# Example: NDBC station download → standardized tables

Downloads one year of standard meteorological and directional spectral data
for NDBC 41010 (Canaveral East), standardizes it, computes spectral bulk
parameters, and runs QARTOD QC. Requires network access (about 6 requests).

```bash
uv run python examples/ndbc_download/run.py            # default 41010 2023
uv run python examples/ndbc_download/run.py 42036 2022
```

Outputs are written to a temporary directory and printed as summaries. For a
persistent, manifest-recorded download use the CLI:

```bash
uv run wave-analysis ndbc download 41010 --years 2023
uv run wave-analysis ndbc standardize 41010
```
