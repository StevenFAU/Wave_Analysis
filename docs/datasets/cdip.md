# Coastal Data Information Program (CDIP)

Registry id `cdip` · planned **v0.2.0** · access verified 2026-09-25.

## Provider

Scripps Institution of Oceanography (UC San Diego), with support from USACE and
California State Parks. Operates a network of **Datawell Directional Waverider**
buoys, the reference-class instrument for wave measurement. Many CDIP buoys are
also listed as NDBC stations (e.g. station 41114 off Fort Pierce, FL is CDIP 134).

## Why CDIP matters here

1. **Reference quality.** Waveriders are the standard against which other wave
   sensors are validated.
2. **Explicit acquisition windows.** `waveTimeBounds` gives start and end of each
   record, so synchronisation can use interval overlap instead of rounded
   timestamps.
3. **Raw displacement.** Archive files include the heave time series
   (`xyzZDisplacement`, typically 1.28 Hz). Both spectral H_m0 and
   zero-crossing H₁/₃, H_max, and T_z can then be derived from the *same*
   record, as in the Yang et al. (2026) dual-labelling strategy, but from a
   physical rather than a virtual buoy.
4. **Kamagata-style H_max targets** need the time series, which only a
   displacement record provides.

## Access (verified)

OPeNDAP via THREDDS, no authentication:

```
https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/{stn}p1_rt.nc
https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/{stn}p1/{stn}p1_historic.nc
```

Variables confirmed from the DDS of station 134: `waveTime`, `waveTimeBounds`,
`waveFlagPrimary/Secondary`, `waveHs`, `waveTp`, `waveTa`, `waveDp`, `waveTz`,
`wavePeakPSD`, `waveFrequency` (64 bands), `waveFrequencyBounds`,
`waveBandwidth`, `waveEnergyDensity`, `waveMeanDirection`,
`waveA1Value`/`waveB1Value`/`waveA2Value`/`waveB2Value`.

## Conventions to handle

- CDIP's a₁, b₁, a₂, b₂ use the nautical angle convention directly, so
  α₁ = atan2(b₁, a₁). Convert with `processing.spectra.ndbc_from_fourier` to the
  project's canonical α/r parameters.
- CDIP publishes `waveBandwidth` and uses it for integration. Midpoint bandwidths
  are only a fallback.
- Flags: CDIP primary flags follow a 1–4/9 scheme. The mapping to QARTOD must be
  confirmed from CDIP documentation.

## Open questions

- CDIP's preferred acknowledgement text and terms for redistribution of derived data.
- Which CDIP buoys lie within a few km of calibrated cameras (CoastCam, WebCOOS,
  USACE FRF Argus)?
