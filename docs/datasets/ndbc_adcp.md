# NDBC ADCP (Acoustic Doppler Current Profiler) Records

Part of the NDBC source ([ndbc.md](ndbc.md)); this page covers what is
specific to currents.

## Availability (inventory 2026-09-26)

- 167 stations with historical `adcp` files and 5 with `adcp2` (168 distinct);
  50 stations publish realtime `.adcp`.
- **48 stations** have ADCP and spectral density in the same year.
- **35** stations with a camera today also have ADCP history.
- Per-station table: `data/registry/ndbc_adcp_availability.csv`
  (`scripts/inventory_adcp.py`).

## Content

`adcp` (`a`) files give, per record, a variable number of
`DEPnn DIRnn SPDnn` triplets: bin depth [m], current direction [degT, "towards"
per oceanographic convention — to be confirmed per NDBC documentation], and
speed [cm/s]. Parsed to long form by `sources.ndbc.parsers.parse_adcp`
(`time_utc, bin, depth_m, direction_degT, speed_cm_s`).

`adcp2` (`b`) files carry per-bin diagnostics (error and vertical velocity,
percent-good, echo intensity EI1–4, correlation CM1–4, flags) and embed position
lines between records. They are archived raw and not yet parsed.

## Role in this project

Kim et al. (2023) used a seafloor AWAC as a **wave reference**. NDBC buoy ADCPs
are hull-mounted current profilers. They provide **currents, not waves**, so
their role here is *context / input* (wave–current interaction, Doppler shifting
of apparent wave periods seen by cameras), never the H_s label.

## Open questions

- Direction convention (towards vs from) and magnetic-declination handling per
  deployment. Verify against NDBC ADCP documentation before use.
- Can the `adcp2` echo intensity near the surface act as a proxy for bubbles
  and breaking?
- The QARTOD current-QC manual (doi:10.7289/V5WM1BMZ) tests apply to `adcp2`
  diagnostics; implementation is planned with the `adcp2` parser.
