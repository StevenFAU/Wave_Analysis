# S2 · Shore video trained end to end: is motion useless, or only frozen?

Fixed 2026-11-21, before any run. Shared rules: [README](README.md). Status: not
started. Branch from `claude/video-whole-clip-491c90`, which holds the video code.

## Question

In the final shore-video run, shuffling the frames changed nothing, and the gain over a
still came from seeing more frames. That run used a frozen video encoder. On buoy
strips, fine-tuning cut error by about a fifth against frozen features. So "motion adds
nothing" may be a fact about waves at these cameras or a fact about frozen encoders.
Training the encoder on ordered and on shuffled clips settles which.

## Data

The 126,099 windows of 3,685 clips (four cameras, 46 local dates) already on the
server, with the five folds over whole local dates of `webcoos_video_v1.yaml`. Labels:
CDIP significant height and peak period. The storm set stays with its own bars.

## Arms

| Arm | Encoder | Frames |
|---|---|---|
| F frozen | The final run's frozen video encoder and clip read-out (reference: 30.5 cm) | Ordered |
| O ordered | The same video encoder, fine-tuned: head only for 2 epochs, then the last four blocks at one tenth the learning rate | Ordered |
| U shuffled | As O, trained and tested on frames shuffled within each window | Shuffled |
| I image | The image encoder fine-tuned on the same windows' frames, mean over frames | No order |

Three seeds per arm when all folds run. Clip prediction is the mean over a clip's
windows, as in the final run.

## Pass marks

Total height, intervals over whole local dates.

- **Primary, motion.** O minus U: upper end below zero. Met means frame order carries
  information once the encoder can adapt.
- **Secondary, fine-tuning.** O minus F: upper end below zero and a gain of at least
  1 cm.
- Reported: peak period for the same arms beside the brightness spectrum (2.59 s); O
  minus I; per camera.

## Stop rule (fixed, because this is the costliest test)

1. **Pilot:** fold 0, seed 0, arms O and U only. Budget 12 GPU-hours.
2. Continue to all folds and seeds only if, on fold 0, O beats U by 1 cm or more, or O
   beats F by 1 cm or more.
3. Hard cap for the whole test: 150 GPU-hours. If the pilot fails rule 2, the result
   is reported as "no gain from order or from fine-tuning in a one-fold pilot", which
   is weaker than a five-fold null and must be worded so.

## Controls

Arm U is the control. Also confirm that O's training loss falls below F's read-out
loss, so a null is not a failure to train.

## Cost and where

Shared server, one GPU, lowest priority: start only when the storm run has finished
and no wave-1 GPU job is waiting.

## Build

A fine-tuning mode in `scripts/train_video_probe.py` (unfrozen blocks, shuffle flag),
unit test that shuffling permutes frames within a window only. Outputs to
`runs/next-s2-<stamp>/`.
