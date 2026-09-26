# Satellite SAR (Sentinel-1) and Altimetry

Registry id `sentinel1_ocn` · planned **v0.4.0** · status **candidate**.

## Role

Synthetic aperture radar does not need daylight and sees through cloud. Level-2
ocean (OCN) products include wave-mode ocean swell spectra, surface wind, and
radial velocity. Revisit time is days, not seconds, so SAR cannot replace
local video. It provides:

- an **all-weather, large-scale context** layer (swell systems approaching a
  coast);
- **independent validation** of regional wave climate;
- a potential weak supervisory signal for swell direction and period.

Satellite altimetry (along-track H_s; Jason, Sentinel-3/6, SWOT) is a second
independent large-scale H_s reference. Its footprint is kilometres, so it does
not match coastal camera scales.

## Rules

- Satellite products are `context` (or weak `reference`) and never replace
  co-located in-situ ground truth.
- Store the satellite acquisition time and footprint geometry; pair by
  space–time overlap with explicit tolerances.
