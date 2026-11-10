"""Sea-state classes from a camera's paired stills: labels, splits and a training package.

The first vision task is a four-class sea-state classifier (WMO sea-state
code, WMO-No. 306, code table 3700) trained on one camera's stills and judged
on held-out days and a held-out year against non-vision baselines (ADR 0007).
This module turns a pairing table (``scripts/webcoos_pairs.py``) into

* a class per still from its window-mean label (ADR 0010),
* two leakage-safe splits (whole local days; a held-out year with an embargo),
* baseline predictions that use no image (majority class, monthly
  climatology, a wave hindcast with a linear correction fitted on training
  stills), and
* a compact package of resized stills with a labels table and checksums, the
  only thing that goes to a GPU cluster.

**Classes.** Code table 3700 gives height bands for codes 0-9 and notes that
"the exact bounding height shall be assigned to the lower code figure", so
the bands are closed on the right: smooth (codes 0-2, H <= 0.5 m), slight
(3, 0.5 < H <= 1.25 m), moderate (4, 1.25 < H <= 2.5 m), rough or more (5-9,
H > 2.5 m).

**Labels near a boundary.** A buoy estimates Hs from a 30-minute record with
a sampling error of a few per cent, so a still whose label lies within that
of a class boundary could honestly belong to either class.
:func:`near_boundary` marks them so results can be reported with and
without them.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from wave_analysis.datasets.splits import EXCLUDED, TEST, TRAIN, SplitResult, grouped_split

#: Upper bounds (inclusive) of the first three WMO-based classes [m].
WMO_BOUNDS_M: tuple[float, ...] = (0.5, 1.25, 2.5)
WMO_CLASS_NAMES: tuple[str, ...] = ("smooth", "slight", "moderate", "rough_or_more")
#: Extra test partition of the year split (data after the test year).
TEST_LATER = "test_later"


def sea_state_class(hs_m: ArrayLike, bounds_m: Sequence[float] = WMO_BOUNDS_M) -> NDArray[np.int64]:
    """Class index per height: the number of bounds strictly below it; -1 where not finite.

    With the WMO bounds, 0.5 m is "smooth" and 0.51 m "slight" (bands closed
    on the right, see the module notes).
    """
    v = np.asarray(hs_m, dtype=np.float64)
    out = np.searchsorted(np.asarray(bounds_m, dtype=np.float64), v, side="left").astype(np.int64)
    out[~np.isfinite(v)] = -1
    return out


def near_boundary(
    hs_m: ArrayLike, bounds_m: Sequence[float] = WMO_BOUNDS_M, *, rel_tol: float = 0.04
) -> NDArray[np.bool_]:
    """Whether each height lies within ``rel_tol`` (relative) of a class boundary."""
    v = np.asarray(hs_m, dtype=np.float64)[:, None]
    b = np.asarray(bounds_m, dtype=np.float64)[None, :]
    return np.asarray((np.abs(v - b) <= rel_tol * b).any(axis=1))


def add_calendar(df: pd.DataFrame, *, time_col: str = "time_utc", tz: str) -> pd.DataFrame:
    """Add ``local_day``, ``local_week`` (ISO), ``local_month`` and ``local_year`` columns."""
    local = pd.to_datetime(df[time_col], utc=True).dt.tz_convert(tz)
    out = df.copy()
    out["local_day"] = local.dt.strftime("%Y-%m-%d")
    iso = local.dt.isocalendar()
    out["local_week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    out["local_month"] = local.dt.month.astype(int)
    out["local_year"] = local.dt.year.astype(int)
    return out


def day_split(df: pd.DataFrame, spec: Mapping[str, Any]) -> SplitResult:
    """Whole local days to train/val/test (``protocol: iid_grouped``)."""
    return grouped_split(
        df, spec.get("group_col", "local_day"), spec["fractions"], seed=int(spec.get("seed", 0))
    )


def year_split(
    df: pd.DataFrame, spec: Mapping[str, Any], *, time_col: str = "time_utc"
) -> SplitResult:
    """Train on some years, test on a later one, and keep later years as an extra test.

    ``spec``: ``train_years``, ``test_years``, ``later_years`` (labelled
    :data:`TEST_LATER`), ``val`` (``group_col``, ``fraction``, ``seed``: whole
    groups of the training years held out for model selection) and
    ``embargo`` (stills within this time after the start of the first test
    year are excluded, so a swell spanning New Year cannot link the parts).
    Stills of other years are excluded.
    """
    year = df["local_year"].astype(int)
    labels = pd.Series(EXCLUDED, index=df.index, name="split", dtype=object)
    train_years = [int(y) for y in spec["train_years"]]
    test_years = [int(y) for y in spec["test_years"]]
    later = [int(y) for y in spec.get("later_years", [])]
    in_train = year.isin(train_years)
    labels[in_train] = TRAIN
    labels[year.isin(test_years)] = TEST
    labels[year.isin(later)] = TEST_LATER
    v = spec.get("val", {})
    if v:
        frac = float(v["fraction"])
        sub = grouped_split(
            df.loc[in_train], v.get("group_col", "local_week"), (1 - frac, frac, 0.0),
            seed=int(v.get("seed", 0)),
        )  # fmt: skip
        labels.loc[sub.labels.index] = sub.labels.to_numpy()
    emb = pd.Timedelta(spec.get("embargo", "0h"))
    if emb > pd.Timedelta(0):
        t = pd.to_datetime(df[time_col], utc=True)
        first_test = df.loc[year.isin(test_years), time_col]
        if len(first_test):
            start = pd.to_datetime(first_test, utc=True).min().normalize()
            labels[(t >= start) & (t < start + emb) & year.isin(test_years)] = EXCLUDED
    return SplitResult(
        labels,
        "year_holdout",
        {
            "train_years": train_years,
            "test_years": test_years,
            "later_years": later,
            "val": dict(v),
            "embargo": str(emb),
        },
    )


# --------------------------------------------------------------------------- #
# Baselines that use no image (ADR 0007)
# --------------------------------------------------------------------------- #


def majority_baseline(y: NDArray[np.int64], train: NDArray[np.bool_]) -> NDArray[np.int64]:
    """The most frequent training class, for every still."""
    counts = np.bincount(y[train & (y >= 0)])
    return np.full(len(y), int(np.argmax(counts)), dtype=np.int64)


def monthly_baseline(
    y: NDArray[np.int64], month: NDArray[np.int64], train: NDArray[np.bool_]
) -> NDArray[np.int64]:
    """The most frequent training class of each calendar month (seasonal climatology)."""
    overall = int(np.argmax(np.bincount(y[train & (y >= 0)])))
    out = np.full(len(y), overall, dtype=np.int64)
    for m in np.unique(month):
        rows = train & (month == m) & (y >= 0)
        if rows.any():
            out[month == m] = int(np.argmax(np.bincount(y[rows])))
    return out


def nowcast_at(times: pd.Series, series: pd.Series, *, max_gap: str = "3h") -> NDArray[np.float64]:
    """A gridded series linearly interpolated to the still times; NaN across gaps over ``max_gap``."""
    s = series.dropna().sort_index()
    if s.empty:
        return np.full(len(times), np.nan)
    t = _ns(pd.DatetimeIndex(pd.to_datetime(times, utc=True)))
    x = _ns(pd.DatetimeIndex(pd.to_datetime(s.index, utc=True)))
    y = s.to_numpy(dtype=np.float64)
    out = np.interp(t, x, y, left=np.nan, right=np.nan)
    pos = np.clip(np.searchsorted(x, t), 1, len(x) - 1)
    gap = (x[pos] - x[pos - 1]) > pd.Timedelta(max_gap).value
    out[gap] = np.nan
    return np.asarray(out, dtype=np.float64)


def _ns(t: pd.DatetimeIndex) -> NDArray[np.int64]:
    """Nanoseconds since the epoch of UTC times."""
    return np.asarray(t.tz_convert(None).to_numpy().astype("datetime64[ns]").astype(np.int64))


def linear_correction(
    x: NDArray[np.float64], y: NDArray[np.float64], rows: NDArray[np.bool_]
) -> tuple[float, float]:
    """Least-squares ``y = a + b x`` on ``rows`` (finite values only), as in the Waimea benchmark."""
    ok = rows & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan"), float("nan")
    b, a = np.polyfit(x[ok], y[ok], 1)
    return float(a), float(b)


# --------------------------------------------------------------------------- #
# Package
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ResizeSpec:
    """How the stills are stored in a package."""

    long_side_px: int = 900
    jpeg_quality: int = 90


def resize_still(src: str | Path, dest: str | Path, spec: ResizeSpec) -> tuple[str, int, int, int]:
    """Write a resized copy of a still; return ``(sha256, size_bytes, width, height)``.

    JPEG DCT scaling (``draft``) halves the decoding work before the
    Lanczos resize. The copy is written to a temporary name, synced and
    renamed, so an interrupted build never leaves a truncated image.
    """
    from PIL import Image

    from wave_analysis.ingest.durable import atomic_open

    with Image.open(src) as im:
        w, h = im.size
        scale = spec.long_side_px / max(w, h)
        size = (round(w * scale), round(h * scale))
        im.draft("RGB", (size[0] * 2, size[1] * 2))
        out = im.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    dest = Path(dest)
    with atomic_open(dest, "wb") as fh:
        out.save(fh, format="JPEG", quality=spec.jpeg_quality, optimize=True)
    data = dest.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data), size[0], size[1]


def package_path(time_utc: pd.Timestamp, name: str) -> str:
    """Path of a still inside a package: ``images/<YYYY>/<MM>/<DD>/<name>`` (UTC date)."""
    return f"images/{time_utc:%Y/%m/%d}/{name}"


def write_checksums(root: Path, rel_paths: Sequence[str], out: Path) -> Path:
    """``sha256sum``-format list of files (``sha256sum -c`` checks a copy on the cluster)."""
    from wave_analysis.ingest.checksum import sha256_file
    from wave_analysis.ingest.durable import atomic_open

    with atomic_open(out, "w") as fh:
        for rel in sorted(rel_paths):
            fh.write(f"{sha256_file(root / rel)}  {rel}\n")
    return out


def class_table(
    df: pd.DataFrame, class_col: str, names: Sequence[str], by: str | Sequence[str]
) -> pd.DataFrame:
    """Counts of each class per group (columns in class order, plus ``total``)."""
    keys = [by] if isinstance(by, str) else list(by)
    t = df.groupby([*keys, class_col]).size().unstack(class_col, fill_value=0)
    t = t.reindex(columns=range(len(names)), fill_value=0)
    t.columns = list(names)
    t["total"] = t.sum(axis=1)
    return t


def default_workers() -> int:
    """Processes for resizing: the CPUs this process may use, at most 8."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - not Linux
        n = os.cpu_count() or 1
    return max(1, min(8, n))


__all__ = [
    "TEST_LATER",
    "WMO_BOUNDS_M",
    "WMO_CLASS_NAMES",
    "ResizeSpec",
    "add_calendar",
    "class_table",
    "day_split",
    "default_workers",
    "linear_correction",
    "majority_baseline",
    "monthly_baseline",
    "near_boundary",
    "nowcast_at",
    "package_path",
    "resize_still",
    "sea_state_class",
    "write_checksums",
    "year_split",
]
