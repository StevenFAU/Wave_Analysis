# B2 · Find the swell cue: which part of the view carries the swell skill?

Fixed 2026-11-21, before any run. Shared rules: [README](README.md). Status: not started.

## Question

On 38,334 strips the buoy's wind alone predicts the swell band with 0.883 m error.
Adding the whole image lowers that to 0.704 m. Adding only a resized crop of the water
gives 0.886 m ([earlier test](../hpc_buoy_followup.md)). So the image helps, the crop
does not, and the cue is unknown. Two explanations are open. The cue may sit in a part
of the view the crop removed or blurred. Or the image may be standing in for light,
season and region, which go with swell climate, and not showing the swell at all.

## Data

- The same five seed-0 fold checkpoints, the same rows (three or more usable water
  views) and the same target as the earlier test: the 0.030 to 0.095 Hz band as an
  equivalent height in metres. The NOAA swell partition (`swell_label_m`) is a
  secondary target.
- Horizon lines from `scripts/buoycam_horizon.py`.

## Arms

Every image arm is a ridge (penalty 10, standardised on the training buoys) on the
features of the masked images plus the wind features (speed, gust, both squares, their
product). Features are extracted under the same mask for training and test rows. A
mask replaces pixels with the view's mean colour. It does not crop or resize, so scale
and position stay as the network learned them.

| Arm | What the network sees |
|---|---|
| whole | Everything (cached; reference 0.704 m) |
| water | Only pixels below the horizon line |
| sky | Only pixels above it |
| band | Only a band of ±12% of the view height around the horizon |
| no structure | Everything except the buoy's own fixed structure. The mask per camera comes from at least 200 daylight strips: pixels whose grey-level spread over time is below that camera's 10th percentile. Commit the builder and ten example masks before any masked score |
| crop | The earlier resized water crop (reference 0.886 m) |

No-pixel arms, same ridge:

| Arm | Inputs |
|---|---|
| wind | The wind features (reference 0.883 m) |
| wind + light | Wind, sun elevation and azimuth, hour and day of year (as sine and cosine), and the six views' mean and spread of brightness |
| wind + geometry | Wind and the horizon roll and pitch per view |

## Decision rules

Define the **retained share** of an arm as (wind error − arm error) ÷ (wind error −
whole error), with the paired group interval.

1. **Light and season proxy.** If wind + light retains 50% or more, the swell skill is
   in large part a proxy. The report and paper then say so, and swell claims are cut to
   what the masked arms add beyond wind + light.
2. **Localised cue.** If exactly one of water, sky or band retains 80% or more, with
   its interval's lower end at 50% or more, while the other two retain under 50%, the
   cue is in that region. The next build reads that region at full resolution.
3. **Resampling, not information.** If water retains 80% or more while crop retains
   about none, the earlier null came from resizing.
4. Otherwise the cue is spread over the view.

Whichever rule applies is the result. There is no pass or fail.

## Controls

The no-structure arm guards against the network reading swell from how the hull sits.
Report every arm by fold: fold 0 is the West Coast, where swell dominates.

## Cost and where

Five masks, five folds, 6 to 10 minutes each on one GPU of the shared server: 3 to 4
GPU-hours. The package and checkpoints are on the server.

## Build

Add `--mask {water,sky,band,nostructure}` to `scripts/measurement_buoy_probe.py`
(beside its water crop), `scripts/buoycam_static_mask.py`, unit tests that a mask
never moves or resizes pixels, and a scorer that prints the retained shares. Outputs to
`runs/next-b2-<stamp>/`.
