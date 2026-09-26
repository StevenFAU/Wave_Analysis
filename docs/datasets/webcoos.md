# WebCOOS — Webcam Coastal Observation System

Registry id `webcoos` · planned **v0.2.0** · status **candidate**.

## Why it matters

WebCOOS (SECOORA and partners, NOAA IOOS funded) runs coastal webcams in the
U.S. Southeast with archived imagery and video and programmatic access
(API, bulk download, live streams). The camera IDs in the **Yin et al. (2025)
VWBT dataset**, the held-out test set of Kamagata et al. (2026), match WebCOOS
naming (`currituck_hampton_inn`, `jennette_north`, `jennette_south`,
`oakisland_east`, `oakisland_west`). WebCOOS could therefore give instrument
labels to the same camera views used in the literature, which those papers
lacked.

## Candidate pairings to investigate

| Camera site | Nearby instrument candidates |
|---|---|
| Currituck / Duck area, NC | USACE FRF Duck instruments and Waveriders; NDBC/CDIP buoys off Duck |
| Jennette's Pier, Nags Head, NC | NDBC/CDIP buoys off Oregon Inlet / Cape Hatteras |
| Oak Island, NC | NDBC/CDIP buoys off Cape Fear / Frying Pan Shoals (41013 has 23 yrs of directional spectra) |

Distances, depths, and time overlap must be quantified with the
inventories before any pairing is accepted.

## Open questions

- API authentication, rate limits, licence, and redistribution terms.
- Frame rate, resolution, and calibration availability per camera.
