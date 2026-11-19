# Test 3 · Paper architectures on unseen buoy stations

**Question.** Does the project camera model outperform each reference architecture using the same strips and labels? Use the sealed station split first, with seeds 0, 1 and 2, one prediction per strip and no hyperparameter choices on its 12 test stations. Five grouped geographic folds are a secondary comparison. Frozen reference baselines are DINOv3-gentle predictions from the matching split and seeds. Height RMSE of 0.440 m sealed and 0.532 m out of fold gives context. Recompute each metric on the exact common sample IDs. Averaging six views does not create six independent labels.

| Comparison | Inputs and target | Primary paired difference |
|---|---|---|
| Choi buoy classification | ImageNet VGG16, global pooled image features and a two-class head. Split each strip into views, take the lower half, resize to 224 × 224, keep per-view mean RGB in [30,190]. Train on qualifying views with one total weight per strip. Average eligible per-view probabilities at inference. Calm Hs <1.52 m, rough Hs >3.05 m; omit the middle. Apply the project's fixed 2.285 m decision threshold to DINOv3 height predictions. | Accuracy(project) − accuracy(VGG16) on common strips. Also show each view position. |
| Kim snapshot | Four-block grayscale snapshot CNN, one view at a time, aggregated as above on available views. Target the published five half-metre classes; buoy strips only have the project's frozen 0.631 × Hs proxy for individual mean height. Report that limitation. The shore study uses measured individual heights. | Accuracy(project) − accuracy(snapshot CNN) with identical range and bins. |
| Yang | Use the repository's ResNet18 + attitude-MLP + transformer on the ordered six views; predict Hs in metres. Preserve camera geometry. Use detected horizon roll and relative pitch; if pitch is a row displacement, document the projection assumption and run the proxy sensitivity separately. Set unavailable yaw to a constant zero in every arm. Keep the same valid strips/views for the no-attitude control. | RMSE(project) − RMSE(Yang), plus Yang with versus without attitude. |

The Choi class named `Choi2020Regressor` implements the paper's different **ship** task, which uses 16 successive 1 Hz frames. The six headings in a strip cannot supply that sequence. Use the published still-image classification task for the primary comparison. An extension using synchronized video and radar would require its own dataset. Inspect the original Kamagata visual branch implementation before running its shore comparison.

Use training splits for fitting and validation only for epoch selection. Use three initialization seeds; average predictions before calculating accuracy or RMSE. Obtain 95% paired percentile intervals using 2,000 resamples of whole sealed buoys, seed 0; use station groups for the five-fold analysis. Success for each classification row requires a positive lower endpoint; success for Yang requires a negative upper endpoint. Record every row including null results. Report the single-view and geometrically matched project controls beside the operational six-view baseline to expose any advantage from input coverage. Sealed results reused for earlier decisions are retrospective. A fresh archive window is required for confirmation. [Paper comparisons](../literature/benchmark_targets.md)

**First sealed run settings.** Use ten AdamW epochs, learning rate 0.0001, weight decay 0.01, eight strips per batch and validation accuracy (classification) or validation RMSE (height) to select one epoch. VGG16 uses ImageNet initialization and global average pooling. Kim uses the published convolution widths and grayscale, with the project's view-quality mask. For Yang the package provides roll in degrees and a relative horizon row fraction. Use atan(pitch_frac) as a normalized pinhole proxy (one image height focal length), fix missing yaw at zero, and report the assumption with any result. Require all six valid viewpoints for the ordered Yang/control pair to avoid supplying missing entries to the temporal transformer. Crop/resize to 224 square; use full views for Kim/Yang. These are project-side adaptations to the available images and labels.

**Review before the Kim and Yang runs (2026-11-19, fixed before any Kim or Yang run).**

- *Choi reproduced.* The registered scorer on the three local Choi runs gives the recorded result exactly (3,185 strips, 12 buoys, 99.22 % against 94.66 %, +4.55 points [2.08, 8.61]). The paper package (`ndbc_buoycam_heights_v2cv`) has the same 41,091 strips, labels, images and station partitions as the package the DINOv3 runs used (`ndbc_buoycam_heights_v2`): the same 6,183 test and 6,311 validation strips. Both arms choose their epoch on the same validation buoys and see no test label in training. Choi covers 3,185 of the 3,190 calm or rough test strips; the other 5 have no view inside its brightness window.
- *What DINOv3 has that VGG16 does not.* Whole views at 270 × 480 with a transformer across the six views; training on all heights (28,597 strips against 15,896) with wind-wave and swell heights as extra targets; colour jitter. VGG16 sees lower halves at 224 × 224. The input-geometry and single-view controls above remain the way to separate these. DINOv3-gentle was itself chosen among recipes with these 12 buoys among the evidence, so its sealed numbers are not independent of them; the five grouped folds give the clean comparison.
- *Kim, corrected to the paper.* The paper's Fig. 5 and Table 3: four blocks of 3 × 3 convolution (16, 32, 64, 64 filters), batch normalisation and ReLU; 4 × 4 max pooling with stride 4 after blocks 1–3 and none after block 4; the last feature map flattened into one fully connected layer and softmax. The repository's class had 2 × 2 pooling after all four blocks, global average pooling and an extra 64-unit layer; `Kim2023SnapshotCNN` now follows the paper. The paper does not state its input size; at 224 × 224 the layer reads 64 maps of 3 × 3. The training recipe stays the registered one for every arm. The paper's own (Adam, 8 epochs, mini-batches of 5, learning rate not stated) is recorded, not adopted.
- *Yang, as specified.* The repository's model follows every part the paper states: ResNet-18 with global pooling per frame; sine and cosine of roll, pitch and yaw through a two-layer MLP ending in a sigmoid; sinusoidal positions; multi-head self-attention; mean over time; a two-layer regression head; MSE. The paper does not state the width, heads, depth, fusion operator or initialisation; they are fixed here at 256, 8 heads, 2 layers, dropout 0.1, concatenation followed by a linear projection, and ImageNet weights. Roll and pitch are each view's horizon angle and row with that camera's median removed (no labels involved), so they come from the same pixels: this tests the paper's architecture with motion measured in the image, not by an IMU.
- *Yang with against without attitude.* RMSE(with attitude) − RMSE(without) on identical strips, three-seed means, 2,000 resamples of whole sealed buoys (seed 0). Attitude helps when the upper endpoint is below zero. `score_sensing_papers.py --paper <yang runs> --control <no-attitude runs>`.
- Classification runs from here on also save each view's probabilities (`view<k>_prob_<class>`), for the per-view-position report.

**Choi's two controls (fixed 2026-11-19, before their runs).**

- *Geometry and protocol matched.* Arm `choi-dinov3` puts the project's backbone (DINOv3 ConvNeXt-Tiny, `convnext_tiny.dinov3_lvd1689m`, pooled, ImageNet normalisation) in place of VGG16 inside Choi's pipeline and changes nothing else: the same lower halves at 224 × 224, brightness window, calm or rough strips, two-class head, loss, view averaging and recipe; seeds 0, 1 and 2. Two paired differences on identical strips, whole-buoy intervals as above: accuracy(DINOv3 in Choi's pipeline) − accuracy(VGG16) is the backbone alone (`--paper <choi-dinov3> --control <choi>`); accuracy(six-view DINOv3-gentle) − accuracy(DINOv3 in Choi's pipeline) is what the project's input and design add beyond the backbone (`--paper <choi-dinov3> --reference <gentle>`).
- *Single view.* Each view position on its own: the accuracy of that view's probabilities alone (three-seed mean) for VGG16 and for DINOv3 in Choi's pipeline, on the strips where that view passes the brightness window, beside the averaged six-view accuracy (`--views`). VGG16's per-view probabilities come from reloading each seed's chosen epoch (`scripts/predict_sensing_papers.py`); the reload must reproduce the run's test classes on at least 99.9 % of strips.
- Both are descriptive controls without a pass mark: they say where the +4.55 points come from.

**Kim with its own training options (sensitivity, fixed 2026-11-19 before any Kim test score).** Kim's network starts from random weights, and the shared recipe's AdamW at 0.0001 may train it less than its authors did. The first seed's validation error stayed at the majority-class level for three epochs. A second Kim triplet therefore uses the paper's Table 3: Adam (0.9, 0.999), 8 epochs, mini-batches of 5 snapshots (here one strip, 5.8 usable views on average); the learning rate 0.001 and L2 penalty 0.0001 are MATLAB's defaults, since the paper states neither (`configs/experiments/sensing_paper_strips_kim_recipe.yaml`, tag `paper-kim-recipe`). It is scored like the primary Kim triplet. The primary row stays the shared recipe; the sensitivity is reported beside it, and the bar counts as met only if both lower endpoints are positive.

### GPU queue

The sealed strip order is Choi VGG16, Kim snapshot, Yang and its no-attitude control. Each arm uses seeds 0, 1 and 2. Submit one seed at a time and fetch its predictions and settings before aggregating. The deployment runner expects the existing heights package locally for checksum comparison. Set `GPU_HOST` to the authorized SSH alias. Use a current shared-server utilization check to select the device. Torchvision will require the official VGG16 and ResNet18 ImageNet weights in the server torch cache or network access during initialization. Use the same cloned weights for the three seeds.

```bash
export PKG=ndbc_buoycam_heights_v2cv CONFIG=configs/experiments/sensing_paper_strips.yaml
export TRAIN_SCRIPT=scripts/train_sensing_papers.py
deploy/dgx/run.sh check
deploy/dgx/run.sh push
deploy/dgx/run.sh setup
# First run once GPU is selected at check-in
GPU="$SLOT" TAG=paper-choi deploy/dgx/run.sh train station --arm choi --seed 0
# Kim and Yang (seeds 0, 1, 2 each), then the scores
TAG=paper-kim deploy/dgx/run.sh train station --arm kim --seed 0 --num-workers 8
TAG=paper-yang deploy/dgx/run.sh train station --arm yang --seed 0 --num-workers 8
TAG=paper-yang-no-attitude deploy/dgx/run.sh train station --arm yang-no-attitude --seed 0 --num-workers 8
REF=$(echo ../Wave_Analysis/runs/buoy-station-20261118t1655-seed{0,1,2}-heights-dinov3-gentle)
python scripts/score_sensing_papers.py --paper runs/buoy-station-*-paper-kim --reference $REF
python scripts/score_sensing_papers.py --paper runs/buoy-station-*-paper-yang --reference $REF
python scripts/score_sensing_papers.py --paper runs/buoy-station-*-paper-yang --control runs/buoy-station-*-paper-yang-no-attitude
# Choi's controls: per-view reload of each VGG16 seed, then the DINOv3 backbone in Choi's pipeline
TRAIN_SCRIPT=scripts/predict_sensing_papers.py TAG=paper-choi-views deploy/dgx/run.sh train station --seed 0 --run /data/runs/<seed-0 choi run>
TAG=paper-choi-dinov3 deploy/dgx/run.sh train station --arm choi-dinov3 --seed 0 --num-workers 8
python scripts/score_sensing_papers.py --paper runs/buoy-station-*-paper-choi-dinov3 --control runs/buoy-station-*-paper-choi-views --views
python scripts/score_sensing_papers.py --paper runs/buoy-station-*-paper-choi-dinov3 --reference $REF
# Kim with its own training options
CONFIG=configs/experiments/sensing_paper_strips_kim_recipe.yaml TAG=paper-kim-recipe deploy/dgx/run.sh train station --arm kim --seed 0 --num-workers 8
python scripts/score_sensing_papers.py --paper runs/buoy-station-*-paper-kim-recipe --reference $REF
```

Use the matching seed's existing sealed DINOv3-gentle station predictions for the comparison. Scores for a completed triplet use the common test-strip intersection across all seeds and models. `predictions.csv` contains probabilities for classification and metres for height. A qualitative check of predictions on the test split during training would compromise the comparison.
