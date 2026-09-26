# Synchronisation

Code: `processing/synchronize.py`, `processing/geospatial.py`; schema
`schemas/sample.py`.

## Rule

A synchronised sample **always stores both original timestamps, their offset,
the spatial separation, and a synchronisation flag**:

$$
\Delta t = t_{\text{target}} - t_{\text{reference}}.
$$

Choi et al. (2020) paired NDBC camera images (about 10 min past the hour) with
wave records (about 40 min past), a 30-minute offset. That may be acceptable for
coarse classes, but it matters for regression, spectra, and event detection, and
it is invisible unless recorded per sample.

## Timestamp semantics come first

Before pairing, establish what each timestamp *means*:

| Source | Timestamp meaning |
|---|---|
| NDBC meteorology | End of an 8-min (buoy) / 2-min (C-MAN) window (since 2004) |
| NDBC waves | **Rounded** to the hour/half-hour; start vs end depends on payload (GSBP start at :40, DACT/VEEP end at :50); ~20-min record |
| NDBC products | The same record can carry different stamps in `.spec` (xx:40) and `.data_spec` (xx:50) |
| CDIP | Explicit window (`waveTimeBounds`) |
| NDBC buoycam | Minute 10 (file name and burned-in caption) |
| Video | Clip start/end; clock source and uncertainty recorded in `ImageRecord` |

Observations therefore carry `time_reference` (`start`/`end`/`center`/`nominal`)
and `averaging_period_s`.

## Pairing modes

**Nearest in time** (`pair_nearest`): each target (image/clip) gets the nearest
reference within a tolerance. Unmatched targets are kept with
`sync_qc_flag = 9`, offsets beyond `suspect_offset` are flagged 3, and input
order is preserved. `direction="backward"` gives causal pairing for real-time
evaluation, where a model may not use a reference measured after the image.

**Window overlap** (`pair_interval_overlap`): used when both sides have windows.
It reports the overlap duration, the overlap fraction of the target window, and
the centre offset. A 3-min clip fully inside a 20-min buoy record has
fraction 1.0.

## Spatial separation and propagation lag

The great-circle distance to the reference (`haversine_m`) is stored per sample.
When the reference is offshore of a coastal camera, the wave state takes time to
arrive. `propagation_lag` estimates the group-velocity travel time of the peak
component:

$$
\tau = d / c_g(f_p, h).
$$

This is an order-of-magnitude correction: one depth, no refraction or shoaling.
Its uncertainty belongs in the error analysis. It is not a license to
treat distant buoys as co-located. Nearshore transformation (shoaling,
refraction, breaking, bathymetry) means an offshore H_s is **not** the H_s in
front of a beach camera. Pairings over large distances must be labelled as such,
and their evaluation stratified by distance.

## Tolerances

Tolerances are experiment parameters (`configs/experiments/*.yaml`), not
constants. Report the distribution of |Δt| for every dataset, along with results
stratified by |Δt|. A model whose error grows with |Δt| is learning weather
persistence, not waves.
