r"""Temporal and spatial synchronisation of heterogeneous observations.

The central rule (``docs/methodology/synchronization.md``): **a synchronised
sample always carries both original timestamps and their offset**,

.. math::
    \Delta t = t_{\text{target}} - t_{\text{reference}},

together with the spatial separation and a synchronisation-quality flag. Choi
et al. (2020) paired NDBC buoy-camera images (~10 min past the hour) with wave
records (~40 min past the hour), a 30-minute offset. That is tolerable for coarse
sea-state classes but not for regression, spectra, or event detection - and
it is invisible unless recorded per sample.

Two pairing modes are provided:

* :func:`pair_nearest` - nearest reference record within a tolerance.
* :func:`pair_interval_overlap` - pair when acquisition *windows* overlap
  (preferred when both sources report windows, e.g. CDIP ``waveTimeBounds``,
  video clip start/end).

and one physical correction:

* :func:`propagation_lag` - group-velocity travel time for a wave state to
  propagate from an offshore buoy to a nearshore camera.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from wave_analysis.physics.dispersion import group_velocity
from wave_analysis.processing.geospatial import haversine_m
from wave_analysis.qc.flags import QCFlag


def pair_nearest(
    targets: pd.DataFrame,
    references: pd.DataFrame,
    *,
    tolerance: str | pd.Timedelta,
    target_time: str = "time_utc",
    reference_time: str = "time_utc",
    direction: Literal["nearest", "backward", "forward"] = "nearest",
    suspect_offset: str | pd.Timedelta | None = None,
    by: str | None = None,
    suffixes: tuple[str, str] = ("", "_ref"),
) -> pd.DataFrame:
    """Pair each target row with the nearest reference row in time.

    Unmatched targets are **kept** (reference columns NaN, ``sync_qc_flag`` =
    MISSING) so that coverage gaps are visible in downstream statistics.

    Parameters
    ----------
    targets, references
        Frames with UTC ``datetime64[ns, UTC]`` time columns.
    tolerance
        Maximum absolute offset for a match.
    direction
        ``"backward"`` uses only references at or before the target time
        (causal pairing, appropriate for real-time evaluation).
    suspect_offset
        Offsets larger than this (but within ``tolerance``) are flagged SUSPECT.
    by
        Optional key column (e.g. ``station_id``) to pair within groups.

    Returns
    -------
    pandas.DataFrame
        Target rows (original order) with reference columns suffixed, plus
        ``target_time_utc``, ``reference_time_utc``, ``time_offset_s`` and
        ``sync_qc_flag``.
    """
    tol = pd.Timedelta(tolerance)
    t = targets.copy()
    r = references.copy()
    t["_order"] = np.arange(len(t))
    # merge_asof requires identical datetime resolutions (pandas >= 3 may infer us or ns).
    t[target_time] = pd.to_datetime(t[target_time], utc=True).dt.as_unit("ns")
    r[reference_time] = pd.to_datetime(r[reference_time], utc=True).dt.as_unit("ns")
    t["target_time_utc"] = t[target_time]
    r["reference_time_utc"] = r[reference_time]
    t = t.sort_values(target_time)
    r = r.sort_values(reference_time)
    merged = pd.merge_asof(
        t,
        r,
        left_on=target_time,
        right_on="reference_time_utc",
        by=by,
        tolerance=tol,
        direction=direction,
        suffixes=suffixes,
    )
    merged = merged.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    offset = (merged["target_time_utc"] - merged["reference_time_utc"]).dt.total_seconds()
    merged["time_offset_s"] = offset
    flags = np.full(len(merged), int(QCFlag.PASS), dtype=np.int8)
    flags[offset.isna().to_numpy()] = int(QCFlag.MISSING)
    if suspect_offset is not None:
        big = offset.abs() > pd.Timedelta(suspect_offset).total_seconds()
        flags[big.fillna(False).to_numpy()] = int(QCFlag.SUSPECT)
    merged["sync_qc_flag"] = flags
    return merged


def pair_interval_overlap(
    targets: pd.DataFrame,
    references: pd.DataFrame,
    *,
    target_start: str,
    target_end: str,
    reference_start: str,
    reference_end: str,
    min_overlap_fraction: float = 0.5,
) -> pd.DataFrame:
    """Pair records whose acquisition windows overlap.

    The overlap fraction is computed relative to the *target* window (e.g. a
    video clip): a 3-min clip fully inside a 20-min buoy record has fraction
    1.0. Returns one row per (target, reference) pair meeting the threshold,
    with ``overlap_s``, ``overlap_fraction``, and ``center_offset_s``
    (target centre minus reference centre).
    """
    t = targets.reset_index(names="target_index")
    r = references.reset_index(names="reference_index")
    ts, te = t[target_start].to_numpy(), t[target_end].to_numpy()
    rs, re_ = r[reference_start].to_numpy(), r[reference_end].to_numpy()
    lo = np.maximum(ts[:, None], rs[None, :])
    hi = np.minimum(te[:, None], re_[None, :])
    overlap = (hi - lo) / np.timedelta64(1, "s")
    dur = (te - ts) / np.timedelta64(1, "s")
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = overlap / dur[:, None]
    ii, jj = np.nonzero(frac >= min_overlap_fraction)
    out = pd.concat(
        [t.take(ii).reset_index(drop=True), r.take(jj).reset_index(drop=True).add_suffix("_ref")],
        axis=1,
    )
    out["overlap_s"] = overlap[ii, jj]
    out["overlap_fraction"] = frac[ii, jj]
    tc = ts[ii] + (te[ii] - ts[ii]) / 2
    rc = rs[jj] + (re_[jj] - rs[jj]) / 2
    out["center_offset_s"] = (tc - rc) / np.timedelta64(1, "s")
    return out


def propagation_lag(
    distance_m: ArrayLike, peak_period_s: ArrayLike, depth_m: ArrayLike | None = None
) -> np.ndarray:
    """Group-velocity travel time [s] of the peak wave component over ``distance_m``.

    First-order estimate for aligning an offshore reference buoy with a
    nearshore camera: energy at the peak period propagates at :math:`c_g(f_p, h)`.
    Using a single representative depth ignores shoaling-induced slowing, so
    this is an *order-of-magnitude* correction (e.g. 10 km at Tp = 10 s in deep
    water is ~ 21 min) whose uncertainty should be propagated, not trusted to
    the minute.
    """
    d = np.asarray(distance_m, dtype=np.float64)
    f = 1.0 / np.asarray(peak_period_s, dtype=np.float64)
    return np.asarray(d / group_velocity(f, depth_m), dtype=np.float64)


def spatial_separation(
    lat: ArrayLike, lon: ArrayLike, ref_lat: ArrayLike, ref_lon: ArrayLike
) -> np.ndarray:
    """Distance [m] between target and reference positions (recorded on every pair)."""
    return haversine_m(lat, lon, ref_lat, ref_lon)
