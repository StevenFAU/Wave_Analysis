# USGS CoastCam

Registry id `usgs_coastcam` · planned **v0.3.0** · status **documented**.

## Provider and first site

USGS St. Petersburg Coastal and Marine Science Center. First target:
**Madeira Beach, Florida** (data release doi:10.5066/P9SGRTF1; portal
<https://cmgds.marine.usgs.gov/data/madeirabeach/>).

## Why calibrated cameras matter

A CoastCam release contains the **intrinsic (IO) and extrinsic (EO)
calibration**, so pixels map to world coordinates. That enables:

- physically scaled measurements (wave lengths, celerity from pixel timestacks,
  runup elevation);
- orthorectification, whose absence Kamagata et al. (2026) name as a cause of
  their +17.9° direction bias;
- transfer across sites that does not depend on memorising one camera's view.

## Products

Argus-convention image products (Holman & Stanley 2007): **snap**
(instantaneous), **timex** (time-exposure mean), **var** (variance), **bright**,
**dark**, and **runup timestacks** (a cross-shore pixel transect sampled over
about 17 minutes). Timex and variance images encode where waves break, which
indicates the sea state. Timestacks encode wave periods and runup directly.

## Pairing

Eastern Gulf of Mexico candidates for Madeira Beach must be selected from the
NDBC/CDIP inventories by distance, depth, and exposure (e.g. NDBC 42036 West
Tampa has 28 years of directional spectra but lies about 180 km offshore; a
nearshore reference would be far better). Every pairing records the separation
and a propagation-lag estimate (`processing.synchronize.propagation_lag`).

## Open questions

- Which CoastCam sites overlap in time with nearshore wave instruments?
- Is raw video available for any site/period, or only products?
