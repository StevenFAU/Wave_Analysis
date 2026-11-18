# Shore-camera video vs a single image (WebCOOS, CDIP labels)

Status: **design fixed 2026-11-18, before any clip was scored.** Config
`configs/experiments/webcoos_video_v1.yaml`; code `scripts/webcoos_video_clips.py`
(clips, frames, package) and `scripts/train_video_probe.py` (features, heads, scores).

## Question

Does motion help? Kim et al. 2023 report 78.3 % from one snapshot and 93.2 % from 180 s
of video on one fixed camera, but on a random split of 30 one-hour sessions (test frames
from the same hour as training frames). Kamagata et al. 2026 use V-JEPA + SlowFast on
1.6-s windows of shore video and reach H_s RMSE 0.586 m, R² 0.107, Pearson 0.436 on five
held-out videos (Oak Island E/W, Xishawan), with labels that cannot be traced to an
instrument. Neither isolates the motion: no published camera work compares video and a
single frame **of the same clips, through the same encoder, at unseen days**, against an
instrument. This test does.

## Data

- **Clips:** the WebCOOS video sample (`scripts/webcoos_video_sample.sh`, approved by
  WebCOOS): Currituck (Hampton Inn), Jennette's Pier (south), Oak Island east and west,
  one clip per 30-min CDIP record in daylight. Test days first, then six sample weeks
  2024-04 to 2026-01.
- **Labels:** CDIP buoys (433 Currituck, 243 Jennette, 200 Oak Island): H_s the mean of
  the records centred within 30 min of the clip's midpoint (ADR 0010), T_p from the
  nearest record. The buoys lie 20-30 km offshore in 13-21 m of water, so the label is the
  incident sea, not the breakers the camera sees.
- **Frames:** 16 s around the midpoint at 4 fps (64 frames; V-JEPA 2's pretraining rate
  and length), 256 px high, aspect kept; Currituck's burned-in "Hampton Inn" label blacked
  out first. Clips with the sun below the horizon at their midpoint are left out.
- **Why whole frames, not surf-zone crops:** the cameras do not hold a fixed view. A
  monthly montage of stills (2022-2026) shows Jennette's pointing changing from month to
  month and Oak Island east replaced in mid-2024 (2048 x 1536 to 2560 x 1440); Currituck's
  framing holds but its lens blurs from 2025. A fixed crop box would cut different scenes
  on different days. Each frame is fed as two squares (its left and right edge), which
  together cover it without stretching (V-JEPA 2's position code needs squares).

## Arms (identical pixels)

| Arm | Encoder (frozen) | Input |
|---|---|---|
| `video` | V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`, Apache-2.0/MIT) | the 64 frames in order |
| `still` | the same | the middle frame repeated 64 times: same encoder, same token count, no motion |
| `shuffled` | the same | the 64 frames in a fixed random order: appearance kept, motion destroyed |
| `image` | DINOv3 ViT-L/16 (`vit_large_patch16_dinov3.lvd1689m`, DINOv3 licence) | the middle frame |

Tokens of four layers (50, 67, 83 and 100 % depth) are average-pooled (V-JEPA 8 x 4 x 4
per tile, DINOv3 4 x 4) and cached. Backbone choice and the controls follow the
backbone review (`reports/Vision backbones for wave cameras.md`): V-JEPA 2 is the only
family pretrained on a span that holds a wave period, and the probing literature
recommends the shuffled and single-frame controls.

## Heads and folds

- **Primary:** an attentive probe (one self-attention block, a learned query, d = 256);
  beside it an MLP on each layer's mean token. Targets log H_s and log T_p, standardised
  on each training fold; Huber loss; AdamW, 60 epochs, one-cycle schedule.
- **Folds:** 5 folds of whole local dates (all cameras of a date together, so one storm
  never sits on both sides). Inside each training fold one date picks the epoch.
  5 seeds, predictions averaged in log space. Every clip is predicted once, out of fold.

## Bars (fixed before scoring)

- **M1:** `video` minus `still`, RMSE of H_s, 95 % interval below zero (paired bootstrap
  over local dates, 2000 resamples).
- **M2:** the same for T_p.

Reported beside them, not bars: `video` minus `shuffled` (if the shuffled clips do as
well, the gain is appearance, not motion), `video` minus `image` (DINOv3), each camera
separately, Kamagata's measures (RMSE, R², Pearson of H_s; RMSE of T_p) on the Oak Island
clips, and Kim's five classes of mean wave height (0.631 H_s, edges 0.75-3.25 m).

**When:** the first scored run once at least 15 local dates are framed (the test days
and the first sample weeks); it is interim until the sample download finishes, and the
final run uses the same code and config.

## Results

None yet.
