# B8 · A general vision-language model as a baseline

Fixed 2026-11-21, before any run. Shared rules: [README](README.md). Status: not started.

## Question

Reviewers will ask what an off-the-shelf model that was never trained on buoy data
makes of a strip. The answer shows what training on instrument truth adds. The
project's source review found no published test of such a model on sea state.

## Data

The 6,183 strips of the 12 sealed test buoys (`split_station` = `test` in the heights
package), where the trained model scores 0.440 m for total height.
Nothing is tuned on them (README rule 7): the prompt is fixed on the station split's
validation buoys and then run once.

## Model

Open weights, run on the shared server. Primary: **Qwen3-VL-8B-Instruct** (Apache-2.0,
ungated). Optional second: InternVL3.5-8B (Apache-2.0). Confirm each licence on its
model card before downloading. No paid API without Steven's yes.

## Arms

| Arm | What the model gets |
|---|---|
| Z zero-shot | The strip, a fixed instruction, and a request for significant wave height in metres and one of the four sea-state classes. No examples |
| K few-shot | The same, with eight labelled strips from training buoys in the prompt, two per class, fixed once |
| Ours | The trained six-view model (three seeds) |
| ERA5 | The corrected ERA5 line |
| Guess | The training buoys' median height and commonest class |

An answer that cannot be parsed counts as the Guess value, and the share of such
answers is reported.

## What is reported

Total-height RMSE and class macro-F1 for each arm on the same strips, with whole-buoy
intervals for Ours minus Z and Ours minus K. This is a baseline row. There is no pass
mark, with one rule fixed in advance: **if Z or K comes within 25% of our RMSE (0.55 m
or better), the paper must discuss it** and a fine-tuned arm is proposed in a new bar.

## Controls

Run arm Z a second time with the six views in shuffled order. Record the exact prompt,
the model revision hash and the decoding settings (greedy) in the run folder.

## Cost and where

Measure throughput on 200 strips first. Estimated 2 to 6 GPU-hours on one GPU with 40
GB free. The weights (about 16 to 18 GB) must be downloaded on the server itself; the
upload link from the desktop is too slow.

## Build

`scripts/buoycam_vlm_baseline.py` (prompt file, inference, parser, scorer), a parser
unit test, the prompt under `configs/experiments/`. Outputs to `runs/next-b8-<stamp>/`.
