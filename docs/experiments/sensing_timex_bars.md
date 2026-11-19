# Test 9: time-exposure and brightest images against one still

Status: **bar T9 fixed 2026-11-19, before any image was packaged or any feature
extracted.** Code: `scripts/sensing_timex_package.py` (desktop: triples and package),
`scripts/sensing_timex_probe.py` (GPU server: frozen features, probes, scores); config
`configs/experiments/sensing_timex_v1.yaml`.

## Question

WebCOOS publishes, beside each camera's stills, a time-exposure (`timex`: the mean of
the frames of ten minutes) and a brightest-pixel image (`brt`: their per-pixel maximum)
for the ten minutes from each stamp. No experiment has used them. Breaking waves leave a
bright band in both, whose width and brightness grow with wave height, which one still
catches only by chance. **Does a `timex` or `brt` image estimate the buoy's H_s better
than a still from the same ten minutes, in a year the model never saw?**

## Data

- **Cameras:** the four with CDIP-paired stills: Currituck Hampton Inn (CDIP 433),
  Jennette's Pier south (243), Oak Island east and west (200; CDIP 200 has no data after
  2026-07-20).
- **Triples:** a `timex` and a `brt` image with the same stamp, and the camera's still
  whose time lies inside their ten minutes (the one nearest the middle if several), with
  the still's CDIP pairing complete (`exclusion` empty) and the sun at or above the
  horizon at the still. Counted 2026-11-19 from file names and the pairing tables (no
  label read): **31,069 triples on 939 days**; 2024: 8,974, 2025: 11,077, 2026: 11,018.
  Currituck 8,735, Jennette's 9,688, Oak Island east 7,242, west 5,404. About half the
  `timex` stamps have no still inside their window; they are left out and counted.
- **Label:** the still's ADR 0010 label (`Hs_label_m`: the mean of the good CDIP records
  centred within 30 min of the still), the same number for all three images of a triple.
  The buoys are 6-29 km from the cameras: the label is the incident sea.
- **Images:** each resized to 256 px high (aspect kept), JPEG quality 90; Currituck's
  "Hampton Inn" caption blacked out first in all three (the box the video work uses,
  560 x 170 px in the lower left of the 2,688 x 1,520 frame). The package holds only these
  images and a table; it reaches the server through R2.

## Split: a held-out year

By date only, before any label is read:

| Part | Dates |
|---|---|
| train | 2024 (from the first `timex`, 2024-03/04), except the validation weeks |
| validation | 15 % of the ISO weeks of 2024, drawn with seed 0, whole weeks |
| **test** | 2025, except 1-3 January (a 72-h embargo after the training year) |
| later test | 2026 (to 2026-10-22; Oak Island to 2026-07-20) |

All four cameras train together. Every arm uses exactly these triples.

## Arms (one encoder, three inputs)

Frozen DINOv3 ViT-L/16 (`vit_large_patch16_dinov3.lvd1689m`), the shore video's `image`
recipe unchanged: the image as two 256-px squares (its left and right edges), tokens of
layers 11, 15, 19 and 23 pooled to 4 x 4 per square.

| Arm | Input |
|---|---|
| `still` | the still |
| `timex` | the time-exposure |
| `brt` | the brightest-pixel image |
| `all3` (reported, not a bar) | the tokens of all three side by side |

Head: the shore video's attentive probe and its settings (`AttentiveProbe`, width 256,
4 heads, dropout 0.2; AdamW 3e-4, weight decay 0.05, batch 32, 60 epochs, one-cycle),
target log H_s standardised on the training triples, Huber loss; the validation weeks
pick the epoch (mean squared error); seeds 0, 1, 2, predictions averaged in log space.
`climatology`: the training triples' geometric-mean H_s.

## Bars

On the 2025 test triples, RMSE of H_s (m), paired bootstrap over whole local days
(2,000 draws, seed 0). Two inputs are tried against the still, so each interval is
97.5 % (Bonferroni), and the test's claim, "time-exposure or brightest image beats the
still", holds at 95 % overall if either bar is met.

- **T9-A:** RMSE(`timex`) minus RMSE(`still`), upper end of the 97.5 % interval below zero.
- **T9-B:** RMSE(`brt`) minus RMSE(`still`), the same.

Reported beside them, not bars: the 2026 later test; each camera; `all3` minus `still`;
every arm against `climatology`; the four WMO sea-state classes of the Currituck work
(H_s up to 0.5, 1.25 and 2.5 m, closed on the right; balanced accuracy and macro-F1) read
from the H_s estimates; the validation-to-test change of each arm (does one input carry
over between years better?). The Currituck still model's 0.73 balanced accuracy on 2025
(a fine-tuned ConvNeXt on one camera) is context, not a comparator.

## Order and compute

Package on the desktop CPU (about 93,000 images, niced), shipped through R2. On the
server, one GPU: features of about 93,000 images (minutes), then 12 small heads; about an
hour with the busy GPUs, peak about 15 GB of GPU memory. Launched under the project's GPU
rule (at most 7 of the project's containers, a GPU with at least 40 GB free that keeps
20 GB after the peak, at most two GPUs across tests 3 and 9 from this session).
