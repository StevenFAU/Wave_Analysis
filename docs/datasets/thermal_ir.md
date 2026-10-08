# Thermal Infrared (LWIR) Imagery

Research direction · no source registered yet.

## Motivation

Thermal imagery has two advantages for this project:

- it **works at night**, the condition most often excluded from the reviewed
  vision datasets;
- breaking waves disrupt the cool surface skin layer and expose warmer bulk
  water, and fresh foam has a different emissivity and temperature. Active
  breaking therefore has a thermal signature distinct from its visible
  appearance.

It may also reveal fronts, upwelling, rip-current plumes, and mixing that RGB
does not show.

## Requirements for a useful dataset

- **Radiometric** thermal video (calibrated brightness temperature, not
  colour-mapped video), with emissivity assumption and atmospheric correction
  recorded (`schemas.imagery.ThermalMetadata`).
- A **co-located physical wave instrument** (Waverider/Spotter/ADCP).

No public dataset combining these was found during project review. A
self-collected radiometric-thermal + wave-buoy dataset would be novel. See
`docs/literature/research_gaps.md` (G-5).

The closest public record, found on 2026-10-08, is Carini et al.'s Duck 2016
dataset: surf-zone thermal IR fused with line-scanning lidar wave profiles,
with wave conditions from an FRF ADOP in 3.5 m depth, over two days
([research datasets](external_research_datasets.md#carini-et-al-duck-2016-thermal-ir-and-lidar)).
Whether its IR products are radiometric is not stated in the readmes; it is a
method check, not a training set.
