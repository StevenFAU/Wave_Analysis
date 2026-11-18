#!/usr/bin/env python
"""Run DINOv3's trainer (``FINO`` branch) on buoy-camera views; DINOv3's checkout stays unmodified.

    torchrun --standalone --nproc_per_node=1 scripts/dinov3_pretrain.py \\
        --config-file configs/pretraining/dinov3_vitb16_buoycam_fino.yaml --output-dir <dir> \\
        train.dataset_path=BuoyViews:root=<package dir>:split=train \\
        student.resume_from_teacher_chkpt=<scripts/dinov3_weights.py to-dinov3 output>

DINOv3 (``facebookresearch/dinov3``, pinned in ``deploy/dgx/pretrain.sh``) must be
importable. This adds one dataset name, ``BuoyViews``
(:class:`wave_analysis.pretraining.buoy_views.BuoyViews`; keys ``root``, ``split``,
``split_col``, ``with_metadata``), to DINOv3's dataset parser, then hands every
argument to ``dinov3.train.train.main``. DINOv3's own launcher
(``dinov3.run.submit``) needs Slurm; ``torchrun`` replaces it on one node.
Experiment: ``docs/experiments/domain_pretraining.md``.
"""

from __future__ import annotations

from typing import Any

from dinov3.data import loaders

from wave_analysis.pretraining.buoy_views import BuoyViews

_dinov3_parse = loaders._parse_dataset_str


def parse_dataset_str(dataset_str: str) -> tuple[type, dict[str, Any]]:
    """DINOv3's parser, plus ``BuoyViews:root=...:split=...``."""
    name, *tokens = dataset_str.split(":")
    if name != "BuoyViews":
        return _dinov3_parse(dataset_str)
    kwargs: dict[str, Any] = {}
    for token in tokens:
        key, value = token.split("=", 1)
        if key not in ("root", "split", "split_col", "with_metadata"):
            raise ValueError(f"BuoyViews takes root, split, split_col, with_metadata, not {key!r}")
        kwargs[key] = value
    if "with_metadata" in kwargs:
        kwargs["with_metadata"] = kwargs["with_metadata"].lower() in ("true", "1", "yes")
    return BuoyViews, kwargs


loaders._parse_dataset_str = parse_dataset_str

if __name__ == "__main__":
    from dinov3.train import train

    train.main()
