# B6 · Neighbour buoys: does a camera improve the forecast somewhere else?

Fixed 2026-11-21, before any run. Shared rules: [README](README.md). Status: not
started. Runs after [B4](b4_camera_as_observation.md), whose error models it reuses.

## Question

A camera corrects the forecast at its own buoy. Forecast errors are spread over
distance, because a forecast that is wrong about the wind is wrong over a region. Does
a camera at one buoy therefore improve the forecast at a buoy 50 to 300 km away that
has no camera of its own in the calculation? A yes means a camera network observes
more sea than its own hulls, which no other group can test: nobody else holds hourly
images from many buoys at once.

## Data

- The nested fusion rows (README rule 3) and the inner calibration predictions.
- **Neighbours of a buoy**: other buoys within 300 km that have a strip within 30
  minutes. Measured on the package: 44 of the 79 buoys have at least one neighbour
  within 300 km, 34 within 200 km, 12 within 100 km. They form 8 station groups of two
  or more buoys; the West Coast group has 18. Neighbours always sit in the target's
  own fold, so no network has seen either.

## Method

The **innovation** at a neighbour j is its camera's wind-wave height minus its
forecast's, in squared heights, as in B4.

1. **Describe (no bar).** The correlation between the forecast's error at the target
   and the innovation at a neighbour, in three distance bins (0 to 100, 100 to 200, 200
   to 300 km), with group intervals.
2. **Update the target without its own camera.** Target analysis = its forecast + the
   sum over neighbours of w(distance) · K_j · innovation_j, with w = a · exp(−distance
   ÷ L). K_j comes from B4. The two numbers a and L are fitted per outer fold on
   calibration rows only (the multi-buoy groups of the other folds, predicted by the
   inner runs).

## Arms, at target buoys with at least one neighbour

Forecast (F); F + neighbours; F + own camera (B4's update); F + own camera +
neighbours.

## Pass mark

- **Primary.** Wind-wave RMSE, (F + neighbours) minus F: upper end below zero, with
  intervals over the whole station groups that have two or more buoys, **and** a gain
  in at least five of those eight groups.
- **Secondary.** (F + own + neighbours) minus (F + own): upper end below zero.
- This is a small sample of groups and the West Coast dominates it. A pass is a
  finding until W3 repeats it.

## Controls

Replace each neighbour with a buoy more than 1,000 km away at the same hour: the gain
must vanish. Report the gain against distance; it should fall with distance.

## What follows

Met: propose, in a new bar, a model over the features of neighbouring cameras (not
only their predictions), and a map product of the wind sea between buoys. Not met:
report the correlation-against-distance figure as the result.

## Cost and where

Desktop CPU, minutes.

## Build

`scripts/sensing_neighbour_update.py`, with a unit test that a target's own strip is
never among its neighbours. Outputs to `runs/next-b6-<stamp>/`.
