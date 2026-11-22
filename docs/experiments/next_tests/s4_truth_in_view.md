# S4 · Truth in the camera's view: are shore errors the method or the labels?

Fixed 2026-11-21, before any run. Shared rules: [README](README.md). Status: not started.

## Question

Every shore label so far comes from a buoy 6 to 29 km offshore. Waves change on the
way in, so part of each shore error may be label mismatch and not the method. The USGS
publishes cross-shore pixel lines sampled at 2 Hz (timestacks) at beaches with an
instrument much closer. If the period methods do clearly better there, the labels were
the limit.

## Data (open; not re-checked since the source review, so verify first)

| Site | Imagery | Truth |
|---|---|---|
| Tres Palmas, Puerto Rico | 331 timestack files, about 50 MB | CDIP 181, 3.5 km away, with raw displacement |
| Madeira Beach, Florida | 6,329 timestack files, 10 to 17 minutes each daylight hour (2017 to 2022) | A current meter 1.7 km away (USGS data release 10.5066/P9HZKIWB) |

Step 0 is a download check: list the files, confirm sizes and licence (public domain),
record each download in the ledger through the existing `Downloader`, and add the
collection to `data/registry/data_dictionary.yaml`. Start with Tres Palmas. Measure
Madeira's total volume before fetching it.

## Arms

| Arm | Method |
|---|---|
| Typical | The median period of the training days at that site |
| B brightness spectrum | The project's pixel-spectrum peak, applied to the timestack with no change |
| T space-time | The two-dimensional spectrum of the timestack (the same method as S3's arm T) |

Folds over whole local dates within each site. The only fitted step is the line from
signal period to instrument period, fitted on training dates.

## Pass marks

- **Primary.** At Tres Palmas, peak-period RMSE of the better of B and T (chosen on a
  development third of the dates) on the remaining dates: below the Typical arm with
  the interval's upper end under zero, **and** at or below 2.0 s. For comparison, the
  best method scores 2.59 s against offshore labels at the four WebCOOS cameras.
- **Secondary.** The same at Madeira Beach, a calm site.
- Reported: significant height from the timestack's variance against the instrument,
  and the wave-by-wave agreement at Tres Palmas if clocks allow it.

## What the result means

Met at both sites: label distance is a real part of the shore error, and the case for
truth in view (a tower with lidar gauges, or our own site) is made with numbers. Not
met: the method is the limit, and S3 matters more.

## Cost and where

Desktop CPU and small downloads. One connection, polite pacing.

## Build

Fill `src/wave_analysis/sources/coastcam.py` (a placeholder today), a datasheet under
`docs/datasets/`, `scripts/coastcam_timestacks.py`. Outputs to `runs/next-s4-<stamp>/`.
