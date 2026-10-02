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

Three pairing modes are provided:

* :func:`pair_nearest` - nearest reference record within a tolerance.
* :func:`pair_interval_overlap` - pair when acquisition *windows* overlap
  (preferred when both sources report windows, e.g. CDIP ``waveTimeBounds``,
  video clip start/end).
* :func:`pair_window` - *every* reference record near each target, with its
  offset, so that the rule choosing the label is applied later and can be
  changed (e.g. when the target's clock is only known to some minutes).

:func:`label_from_candidates` then turns those candidates into one label per
target with a named rule (nearest, interpolate, window mean, in-record; ADR
0010), so that results can be compared across rules.

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


def pair_window(
    targets: pd.DataFrame,
    references: pd.DataFrame,
    *,
    window: str | pd.Timedelta,
    target_time: str = "time_utc",
    reference_start: str = "start_utc",
    reference_end: str = "end_utc",
) -> pd.DataFrame:
    """Every reference record whose centre lies within ``window`` of each target time.

    Returns one row per (target, reference) pair with ``target_index`` and
    ``reference_index`` (index labels of the inputs), ``offset_s`` (target time
    minus reference centre, so positive when the target is later),
    ``in_record`` (the target time lies in ``[start, end)``) and ``rank``
    (0 for the nearest centre of each target; ties go to the earlier record).
    Targets without a time or without any reference in the window have no
    rows; count them from ``targets``.
    """

    def ns(s: pd.Series) -> np.ndarray:
        utc = pd.to_datetime(s, utc=True).dt.tz_convert(None)
        return utc.to_numpy(dtype="datetime64[ns]").view(np.int64)

    w = pd.Timedelta(window).value
    s_ns, e_ns = ns(references[reference_start]), ns(references[reference_end])
    if references[[reference_start, reference_end]].isna().any().any():
        raise ValueError("reference records need both start and end times")
    if (e_ns < s_ns).any():
        raise ValueError("a reference record ends before it starts")
    c_ns = s_ns + (e_ns - s_ns) // 2
    order = np.argsort(c_ns, kind="stable")
    c_sorted = c_ns[order]

    has_time = targets[target_time].notna().to_numpy()
    t_ns = np.where(has_time, ns(targets[target_time]), 0)
    lo = np.searchsorted(c_sorted, t_ns - w, side="left")
    hi = np.searchsorted(c_sorted, t_ns + w, side="right")
    n = np.where(has_time, hi - lo, 0)
    ti = np.repeat(np.arange(len(t_ns)), n)
    pos = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + np.repeat(lo, n)
    ri = order[pos]
    offset = t_ns[ti] - c_ns[ri]
    out = pd.DataFrame(
        {
            "target_index": targets.index.to_numpy()[ti],
            "reference_index": references.index.to_numpy()[ri],
            "offset_s": offset / 1e9,
            "in_record": (t_ns[ti] >= s_ns[ri]) & (t_ns[ti] < e_ns[ri]),
        }
    )
    key = np.lexsort((c_ns[ri], np.abs(offset), ti))
    rank = np.empty(len(out), dtype=np.int64)
    first = np.r_[0, np.flatnonzero(np.diff(ti[key])) + 1] if len(out) else np.array([], int)
    within = np.arange(len(out)) - np.repeat(first, np.diff(np.r_[first, len(out)]))
    rank[key] = within
    out["rank"] = rank
    return out


LabelRule = Literal["nearest", "interpolate", "window_mean", "in_record"]
LABEL_RULES: tuple[LabelRule, ...] = ("nearest", "interpolate", "window_mean", "in_record")


def _wrap_deg(x: pd.Series) -> pd.Series:
    """Angles wrapped to [-180, 180)."""
    return (x + 180.0) % 360.0 - 180.0


def _atan2_deg360(sin: np.ndarray, cos: np.ndarray) -> np.ndarray:
    """Direction in [0, 360) from vector components (rounding can give 360.0 itself)."""
    deg = np.rad2deg(np.arctan2(sin, cos)) % 360.0
    return np.where(deg >= 360.0, 0.0, deg)


def label_from_candidates(
    candidates: pd.DataFrame,
    value: str,
    *,
    rule: LabelRule,
    target: str = "target_index",
    max_offset: str | pd.Timedelta = "30min",
    max_gap: str | pd.Timedelta = "40min",
    lag_s: float | str | None = None,
    circular: bool = False,
) -> pd.DataFrame:
    """One label per target from the reference records around it (ADR 0010).

    ``candidates`` is :func:`pair_window` output joined with the reference
    values: one row per (target, record) with ``offset_s`` (target time minus
    record centre), ``in_record`` and the ``value`` column. Records whose value
    is missing are ignored.

    Rules
    -----
    ``nearest``
        The record with the nearest centre, if within ``max_offset``; ties go
        to the earlier record (as :func:`pair_window`'s rank).
    ``interpolate``
        Linear in time between the nearest record centred at or before the
        target and the nearest after it, if their centres are at most
        ``max_gap`` apart. A target exactly at a centre takes that record.
    ``window_mean``
        Mean of every record centred within ``max_offset``. With two records
        30 min apart and independent sampling noise, it halves the noise
        variance of a single record. Meant for statistics of the whole
        spectrum; a peak value (T_p, D_p) can jump between swell systems, and
        the mean of two peaks may match neither (ADR 0010 uses ``nearest``).
    ``in_record``
        The record whose sample contains the target time.

    ``lag_s`` (seconds, or the name of a column such as a per-record
    propagation lag) moves each target time earlier before the rule is
    applied, for a reference upstream of the target. It cannot be combined
    with ``in_record``, whose containment flag refers to the unshifted time.
    ``circular=True`` treats the values as directions in degrees (vector mean
    and interpolation; spread from wrapped deviations).

    Returns
    -------
    pandas.DataFrame
        Indexed by ``target`` (every target present in ``candidates``, in
        order of appearance), with ``label`` (NaN when the rule finds no
        record), ``spread`` (max - min of the records used; NaN with fewer
        than two), ``n_records`` and ``max_abs_offset_s`` (after the lag).
    """
    if rule not in LABEL_RULES:
        raise ValueError(f"unknown label rule {rule!r}; choose one of {LABEL_RULES}")
    if lag_s is not None and rule == "in_record":
        raise ValueError("the in_record rule cannot be combined with a lag")
    cols = [target, "offset_s", value]
    if rule == "in_record":
        cols.append("in_record")
    if isinstance(lag_s, str):
        cols.append(lag_s)
    c = candidates[list(dict.fromkeys(cols))].copy()
    targets = pd.Index(c[target].unique(), name=target)
    c = c[c[value].notna()]
    off = c["offset_s"].astype("float64")
    if lag_s is not None:
        off = off - (c[lag_s].astype("float64") if isinstance(lag_s, str) else float(lag_s))
    c["off"] = off
    c["abs_off"] = off.abs()
    max_off = pd.Timedelta(max_offset).total_seconds()

    def first_by(frame: pd.DataFrame, by: list[str], ascending: list[bool]) -> pd.DataFrame:
        frame = frame.sort_values([target, *by], ascending=[True, *ascending], kind="stable")
        return frame.groupby(target, sort=False).head(1).set_index(target)

    if rule in ("nearest", "in_record"):
        sel = c[c["abs_off"] <= max_off] if rule == "nearest" else c[c["in_record"].astype(bool)]
        best = first_by(sel, ["abs_off", "off"], [True, False])  # tie: earlier record
        res = pd.DataFrame(
            {
                "label": best[value].astype("float64"),
                "spread": np.nan,
                "n_records": 1,
                "max_abs_offset_s": best["abs_off"],
            }
        )
    elif rule == "window_mean":
        sel = c[c["abs_off"] <= max_off]
        v = sel[value].astype("float64")
        g = sel.groupby(target, sort=False)
        n = g.size()
        if circular:
            rad = np.deg2rad(v.to_numpy())
            s = pd.Series(np.sin(rad), index=sel.index).groupby(sel[target], sort=False).mean()
            co = pd.Series(np.cos(rad), index=sel.index).groupby(sel[target], sort=False).mean()
            mean = pd.Series(_atan2_deg360(s.to_numpy(), co.to_numpy()), index=s.index)
            dev = _wrap_deg(v - sel[target].map(mean).astype("float64")).groupby(sel[target])
            spread = dev.max() - dev.min()
        else:
            mean = v.groupby(sel[target], sort=False).mean()
            spread = v.groupby(sel[target], sort=False).max() - v.groupby(sel[target]).min()
        res = pd.DataFrame(
            {
                "label": mean,
                "spread": spread.where(n >= 2),
                "n_records": n,
                "max_abs_offset_s": g["abs_off"].max(),
            }
        )
    else:  # interpolate
        gap = pd.Timedelta(max_gap).total_seconds()
        before = first_by(c[c["off"] >= 0], ["off"], [True])
        after = first_by(c[c["off"] < 0], ["off"], [False])
        j = before[[value, "off"]].join(after[[value, "off"]], lsuffix="_b", rsuffix="_a")
        vb, va = j[f"{value}_b"].astype("float64"), j[f"{value}_a"].astype("float64")
        span = j["off_b"] - j["off_a"]
        w = j["off_b"] / span
        if circular:
            rb, ra, wv = np.deg2rad(vb.to_numpy()), np.deg2rad(va.to_numpy()), w.to_numpy()
            sin = np.sin(rb) + wv * (np.sin(ra) - np.sin(rb))
            cos = np.cos(rb) + wv * (np.cos(ra) - np.cos(rb))
            lab = pd.Series(_atan2_deg360(sin, cos), index=j.index)
            spread = _wrap_deg(va - vb).abs()
        else:
            lab = vb + w * (va - vb)
            spread = (va - vb).abs()
        ok = span <= gap
        exact = j["off_b"] == 0
        res = pd.DataFrame(
            {
                "label": lab.where(ok).mask(exact, vb),
                "spread": spread.where(ok & ~exact),
                "n_records": np.where(exact, 1, np.where(ok, 2, 0)),
                "max_abs_offset_s": pd.concat([j["off_b"], -j["off_a"]], axis=1)
                .max(axis=1)
                .where(ok)
                .mask(exact, 0.0),
            },
            index=j.index,
        )
    out = res.reindex(targets)
    out["n_records"] = out["n_records"].fillna(0).astype("int64")
    out["label"] = out["label"].astype("float64")
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
