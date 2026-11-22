# S3 · Label-free wave period from shore video

Fixed 2026-11-21, before any run. Shared rules: [README](README.md). Status: not
started. Branch from `claude/video-whole-clip-491c90`
(`src/wave_analysis/processing/pixel_spectrum.py` lives there). Same session as S2, one
after the other: both edit the video code.

## Question

The only thing that improved shore peak period was a hand-built spectrum of water
brightness (2.76 s for typical conditions down to 2.59 s). The strongest published
period methods are hand-built too. Can a signal learned from unlabelled clips, with a
loss that rewards a single clean periodicity, or a space-time image along a line across
the surf, do better?

## Data

The 3,685 clips, used **without labels** for learning. Labels (CDIP peak period and
mean zero-crossing period) are used only to score, in the five date folds.

## Arms

| Arm | Signal |
|---|---|
| B brightness spectrum | The existing pixel spectrum's peak. The reference |
| T space-time image | Brightness along a fixed cross-shore line per camera over time; the period from its two-dimensional spectrum. The lines are drawn once per camera from a mean image, before any score, and committed |
| L learned periodic signal | A small network maps a clip to a one-dimensional signal, trained with periodicity losses (energy inside the 3 to 25 s band, a sparse spectrum, variance across clips) and no labels. The period is the signal's spectral peak |
| Typical | The training dates' median period |

For arms B, T and L the only fitted step is a line from the signal's period to the
buoy's, fitted on the training dates.

## Pass mark

- **Primary.** Peak-period RMSE, the better of T and L (chosen on the 18 development
  dates) minus B, on the other 28 dates: upper end below zero, intervals over whole
  local dates.
- Reported: both T and L; mean zero-crossing period; per camera; the share of clips
  where each method returns no clear peak.
- The 28 dates were opened for earlier bars, so a pass is a finding. It is confirmed
  only on clips from dates downloaded after 2026-11-21, scored once.

## Controls

Shuffle the frames of each clip: the period signal must disappear. Check arm L on
synthetic clips with a known period before any real score.

## Cost and where

Arms B and T on the desktop CPU from frames on disk or on the server. Arm L under 10
GPU-hours.

## Build

A space-time module and a periodicity-loss trainer beside `pixel_spectrum.py`, with
unit tests on synthetic waves. Outputs to `runs/next-s3-<stamp>/`.
