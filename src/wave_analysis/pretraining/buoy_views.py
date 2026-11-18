"""The usable views of a buoy-camera package as unlabelled images for DINOv3's trainer.

Each item is one view of one strip (the package's images have the caption cut),
cropped out of the strip on the fly. Only the columns that place a view are read
from ``labels.csv`` (image path, station, partition, the views' grey statistics):
no wave height, wind or other label enters, so the pretraining is label-free.

The only metadata is the view's **station**, as an index into the sorted station
IDs of the chosen partition, for a FINO guide head (DINOv3's ``FINO`` branch,
``dinov3/train/metadata_utils``). DINOv3's data contract: an item is
``(image, (label, metadata))`` with ``metadata`` a dataclass whose field names
match the guides' names.

Used through ``scripts/dinov3_pretrain.py`` as
``BuoyViews:root=<package dir>:split=train``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets import VisionDataset

from wave_analysis.datasets.buoycam import usable_view_mask, view_columns
from wave_analysis.sources.ndbc.buoycam import N_VIEWS

#: The package's rule for a usable view (``images.min_view_mean`` / ``min_view_std``).
MIN_VIEW_MEAN = 20.0
MIN_VIEW_STD = 4.0


@dataclass
class ViewMetadata:
    """Label-free metadata of one view; field names are guide names."""

    station: int  # index into BuoyViews.stations


def view_table(
    package: Path | str,
    split: str = "train",
    *,
    split_col: str = "split_station",
    min_view_mean: float = MIN_VIEW_MEAN,
    min_view_std: float = MIN_VIEW_STD,
) -> pd.DataFrame:
    """One row per usable view of the strips in partition ``split``: image, view, station_id."""
    cols = ["image", "station_id", split_col]
    cols += view_columns("view_mean", N_VIEWS) + view_columns("view_std", N_VIEWS)
    t = pd.read_csv(Path(package) / "labels.csv", usecols=cols, dtype={"station_id": str})
    t = t[t[split_col] == split].reset_index(drop=True)
    if t.empty:
        raise ValueError(f"no strips with {split_col} == {split!r} in {package}")
    ok = usable_view_mask(t, min_mean=min_view_mean, min_std=min_view_std, n_views=N_VIEWS)
    rows, views = np.nonzero(ok)
    return pd.DataFrame(
        {
            "image": t["image"].to_numpy()[rows],
            "view": views.astype(np.int64),
            "station_id": t["station_id"].to_numpy()[rows],
        }
    )


def crop_view(strip: Image.Image, view: int, n_views: int = N_VIEWS) -> Image.Image:
    """View ``view`` (0-based, left to right) of a strip of ``n_views`` equal views."""
    w, h = strip.size
    step = w // n_views
    return strip.crop((view * step, 0, (view + 1) * step, h))


class BuoyViews(VisionDataset):  # type: ignore[misc]
    """Usable views of one partition of a buoy-camera package: ``(image, (0, ViewMetadata))``.

    ``split``: the partition of ``split_col`` (default the station split's ``train``,
    the 55 training buoys). ``with_metadata=False`` gives ``(image, 0)``.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        *,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        transforms: Callable[..., Any] | None = None,
        with_metadata: bool = True,
        split_col: str = "split_station",
    ) -> None:
        super().__init__(
            root, transforms=transforms, transform=transform, target_transform=target_transform
        )
        table = view_table(root, split, split_col=split_col)
        self.stations = sorted(table["station_id"].unique())
        index = {s: i for i, s in enumerate(self.stations)}
        self.images = table["image"].tolist()
        self.views = table["view"].to_numpy()
        self.station_idx = table["station_id"].map(index).to_numpy(dtype=np.int64)
        self.with_metadata = with_metadata

    def __len__(self) -> int:
        return len(self.images)

    def get_target(self, index: int) -> Any:
        """``(0, ViewMetadata)``, or ``0`` without metadata (DINOv3 ignores the label)."""
        if not self.with_metadata:
            return 0
        return (0, ViewMetadata(station=int(self.station_idx[index])))

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        with Image.open(Path(self.root) / self.images[index]) as im:
            image = crop_view(im.convert("RGB"), int(self.views[index]))
        target = self.get_target(index)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


__all__ = ["MIN_VIEW_MEAN", "MIN_VIEW_STD", "BuoyViews", "ViewMetadata", "crop_view", "view_table"]
