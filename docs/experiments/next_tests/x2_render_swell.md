# X2 · Render the swell limit: is swell visible from buoy height at all?

Fixed 2026-11-21, before any run. Shared rules: [README](README.md). Status: not
started. Best begun after [B1](b1_hours_before.md) and [B2](b2_swell_cue.md) report.

## Question

"The camera misses swell" is an observation. An explanation needs a case where
everything is known. If seas are simulated with and without swell and viewed from a
camera riding the surface at buoy height, how much does the swell change one snapshot,
compared with how much the picture changes anyway from one moment of the same wind sea
to the next? And does a sequence of snapshots do better? This links Yang's simulation
work to the real-data result, and it tells B1 what to expect.

## Method

**Stage 1 (CPU, no learning).**

- Build sea surfaces by adding wave components drawn from a two-part spectrum: a wind
  sea (JONSWAP shape, significant height 0.5, 1 or 2 m) and a narrow swell (height 0,
  0.5, 1, 2 or 3 m; period 10, 14 or 18 s). Fifty random realisations per combination.
- Place a camera on the surface. It heaves and tilts with the surface under it, as a
  buoy does. Take the camera height and the field of view of the six views from NDBC's
  published description of its buoy cameras; cite the source in the run folder before
  rendering anything. If no figure is published, use 3, 4 and 5 m and report all three.
- Render each view by tracing rays to the surface with a simple shading model: sky
  reflection by the Fresnel law over a uniform sky, plus a constant water colour. Size
  the views like the real ones.
- **Measure.** For each swell setting, the detectability index d′ between "this swell"
  and "no swell" using the project's frozen image features and a linear classifier
  trained on half the realisations. Do it for one snapshot and for a sequence of 12
  snapshots one hour apart with the swell unchanged and the wind sea redrawn.

**Stage 2 (GPU, 10 to 30 hours; only if stage 1 shows d′ above 1 somewhere).** Train
the project's network on renders to regress swell height, find its ceiling in
simulation, then apply the render-trained features to real strips as a probe.

## Decision rules (fixed in advance; there is no pass or fail)

1. If one snapshot gives d′ below 1 for a 1 m, 14 s swell under a 1 m wind sea, while
   the 12-snapshot sequence gives d′ above 2: one image cannot show ordinary swell and
   time context can. That supports B1's design and explains the real-data result.
2. If one snapshot already gives d′ above 2: the information is in the image, and the
   real model's weakness is data coverage or training, not optics.
3. If neither gives d′ above 1: swell height is not recoverable from these views, and
   the paper says the division of labour (camera for wind sea, model for swell) is
   set by geometry.

State plainly what the simple renderer leaves out: foam, glitter, haze, lens droplets
and the buoy's own structure. A result here bounds what geometry allows. It does not
predict real accuracy.

## Controls

Before any swell result, check the renderer against the real strips on two statistics
of the wind sea alone: the horizon's roll spread against measured wind speed, and the
image brightness spread. If the renders do not reproduce the real trend with wind, fix
the renderer first.

## Cost and where

Stage 1 on the desktop CPU (`nice`). Stage 2 on the shared server.

## Build

`src/wave_analysis/physics/render.py` (surface synthesis, camera, ray tracing) with
unit tests on a flat sea and on a single sine wave whose slope is known;
`scripts/render_swell_study.py`. Outputs to `runs/next-x2-<stamp>/`.
