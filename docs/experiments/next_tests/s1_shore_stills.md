# S1 · One model for all shore stills

Fixed 2026-11-21, before any run. Shared rules: [README](README.md). Status: not started.

## Question

Shore stills have only been tested with frozen probes and one single-camera model. On
buoy strips, fine-tuning cut error by about a fifth against frozen features. Three
questions follow. Does fine-tuning help on shore too? Does a shore model work at a
beach it never saw? Does starting from the buoy-trained network help, which would mean
"seeing waves" is one skill across platforms?

## Data

- **Stage 1 (on disk now).** Package `sensing_timex_v1`: 30,931 stills at four cameras
  (Jennette's Pier south 9,667; Currituck 8,722; Oak Island east 7,199 and west 5,343),
  labels from CDIP 243, 433 and 200. Splits: the package's held-out year 2025; and
  leave-one-site-out with three sites (the two Oak Island cameras share a beach and a
  buoy, so they leave together).
- **Stage 2 (needs the data drive and new pairings).** All eight cameras' 241,229
  stills. First pair the four unpaired cameras with a buoy and write the pairing table.
- Shore labels are measured 6 to 29 km offshore. Every result carries that caveat.

## Arms (three seeds each, gentle recipe)

| Arm | Start | Training |
|---|---|---|
| P frozen probe | Public DINOv3 | Ridge on frozen features. The reference: 26.5 cm on the held-out year |
| A fine-tuned | Public DINOv3 | Linear head 2 epochs, then the whole network at one tenth the learning rate |
| C from buoys | The buoy-trained DINOv3-gentle backbone (station split, the same seed) | As A |

No-camera comparators, fitted on the training years or sites only: the typical
conditions of the training data (61.3 cm on the held-out year), and the ERA5 line at
each buoy (`data/raw/era5/webcoos_*`).

## Pass marks

All on total height, with intervals over whole local dates.

- **S1-a, fine-tuning.** Held-out year, A minus P: upper end below zero and a gain of
  at least 1.5 cm.
- **S1-b, an unseen beach.** Leave-one-site-out, pooled over the three sites: the
  fixed 50/50 mean of arm A and the ERA5 line, minus the ERA5 line alone: upper end
  below zero. The ERA5 line for a held-out site is fitted on the other sites. Arm A
  alone is reported beside it.
- **S1-c, one encoder.** Held-out year, C minus A: upper end below zero and a gain of
  at least 1 cm. Reported as well: C minus A on the unseen beaches, and the reverse
  direction as frozen probes only (features of the shore-trained backbone on the buoy
  folds against public DINOv3 features).

## Controls

A shuffled-label run of arm A must fall to the comparator. Report per camera and per
site; the Currituck lens has been hazy since mid-2025.

## Stop rule

If S1-a is not met after stage 1, do not build stage 2; report that frozen features
are enough on shore.

## Cost and where

A run on 30,931 stills is under an hour on one GPU. Stage 1: 9 held-out-year runs and
18 leave-one-site-out runs for arms A and C, about 10 to 20 GPU-hours. Check first
whether the package is already on the server (the summary-image test ran there).

## Build

`scripts/train_shore_stills.py` on the pattern of `scripts/train_sea_state.py` and
`scripts/sensing_timex_probe.py`, a config, a scorer with the date bootstrap. Outputs
to `runs/next-s1-<stamp>/`.
