"""Self-supervised pretraining on the project's own unlabelled images.

* :mod:`wave_analysis.pretraining.buoy_views` - the usable views of a buoy-camera
  package as a dataset for DINOv3's trainer (no label is read).
* :mod:`wave_analysis.pretraining.dinov3_weights` - DINOv3 weights between timm's
  format (the training scripts' backbones) and DINOv3's own (its trainer).

Experiment: ``docs/experiments/domain_pretraining.md``.
"""
