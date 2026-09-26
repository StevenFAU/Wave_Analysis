"""IOOS QARTOD real-time QC tests for in-situ surface waves (bulk-parameter subset).

Implements the long-term (LT) time-series tests of the *Manual for Real-Time
Quality Control of In-Situ Surface Wave Data*, Version 2.1 (U.S. IOOS, 2019;
https://doi.org/10.25607/OBP-759), plus the generic gap test. Test numbers
below refer to that manual.

Thresholds are **operator-defined and location-dependent** in QARTOD. This
module therefore never hard-codes them; they are supplied from
``configs/qc/wave_qc.yaml`` and recorded in each :class:`QCTestResult` so the
parameters travel with the flags into provenance.

Implemented
-----------
* Test 9  - time-series gap (applied here to the bulk-parameter series)
* Test 15 - mean and standard deviation
* Test 16 - flat line
* Test 17 - operational frequency range (spectral)
* Test 18 - low-frequency energy (spectral)
* Test 19 - bulk wave parameters max/min/acceptable range
* Test 20 - rate of change
* Test 21 - neighbor check

Not implemented (require raw instrument streams not distributed by NDBC):
Tests 1-8 (ADCP signal diagnostics) and Tests 10-14 (short-term displacement
series). The interface is designed so they can be added for CDIP displacement
data without changes to callers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from wave_analysis.qc.flags import QCFlag, QCTestResult

_REF = "IOOS QARTOD In-Situ Surface Waves Manual v2.1 (2019), doi:10.25607/OBP-759"
FlagArray = NDArray[np.int8]


def _epoch_seconds(times: ArrayLike) -> NDArray[np.float64]:
    """UTC epoch seconds, independent of the datetime resolution pandas chose (ns/us/s)."""
    t = pd.to_datetime(np.asarray(times), utc=True)
    seconds = (t - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)
    return np.asarray(seconds, dtype=np.float64)


def _init_flags(values: NDArray[np.float64]) -> FlagArray:
    flags = np.full(values.shape, int(QCFlag.PASS), dtype=np.int8)
    flags[~np.isfinite(values)] = int(QCFlag.MISSING)
    return flags


def gap_test(
    times: ArrayLike, expected_interval_s: float, *, tolerance: float = 0.5
) -> QCTestResult:
    """Test 9 (time-series gap), applied at the bulk-record level.

    A record is SUSPECT if the interval since the previous record exceeds
    ``expected_interval_s * (1 + tolerance)`` - i.e. it follows a gap. It is
    FAIL if timestamps are non-increasing (duplicate or out-of-order), which
    otherwise silently corrupts rate-of-change tests and window construction.
    """
    t = _epoch_seconds(times)
    dt = np.diff(t)
    flags = np.full(len(t), int(QCFlag.PASS), dtype=np.int8)
    if len(t) > 1:
        flags[1:][dt > expected_interval_s * (1 + tolerance)] = int(QCFlag.SUSPECT)
        flags[1:][dt <= 0] = int(QCFlag.FAIL)
    return QCTestResult(
        "gap",
        "temporal",
        flags,
        {"expected_interval_s": expected_interval_s, "tolerance": tolerance},
        _REF + " Test 9",
    )


def gross_range_test(
    values: ArrayLike,
    fail_span: tuple[float, float],
    suspect_span: tuple[float, float] | None = None,
    *,
    name: str = "gross_range",
) -> QCTestResult:
    """Test 19 (bulk parameter max/min/acceptable range) for one parameter.

    Values outside ``fail_span`` (sensor/physical limits) are FAIL; values
    inside it but outside ``suspect_span`` (local climatological limits) are
    SUSPECT. The manual specifies that a wave-*height* failure should fail all
    bulk parameters of the record; see :func:`propagate_height_failure`.
    """
    v = np.asarray(values, dtype=np.float64)
    flags = _init_flags(v)
    ok = np.isfinite(v)
    lo, hi = fail_span
    flags[ok & ((v < lo) | (v > hi))] = int(QCFlag.FAIL)
    if suspect_span is not None:
        slo, shi = suspect_span
        sus = ok & (flags == int(QCFlag.PASS)) & ((v < slo) | (v > shi))
        flags[sus] = int(QCFlag.SUSPECT)
    return QCTestResult(
        name,
        "physical",
        flags,
        {
            "fail_span": list(fail_span),
            "suspect_span": list(suspect_span) if suspect_span else None,
        },
        _REF + " Test 19",
    )


def propagate_height_failure(height_flags: ArrayLike, other_flags: ArrayLike) -> FlagArray:
    """Apply the Test-19 rule "if wave height fails, no bulk parameters should be released"."""
    h = np.asarray(height_flags, dtype=np.int8)
    o = np.asarray(other_flags, dtype=np.int8).copy()
    o[(h == int(QCFlag.FAIL)) & (o != int(QCFlag.MISSING))] = int(QCFlag.FAIL)
    return o


def flat_line_test(
    values: ArrayLike,
    *,
    suspect_count: int = 3,
    fail_count: int = 5,
    tolerance: float = 1e-6,
) -> QCTestResult:
    """Test 16 (flat line / stuck sensor).

    Following the manual's codable instructions, the present observation is
    SUSPECT when it and the preceding ``suspect_count - 1`` observations are all
    equal within ``tolerance``, and FAIL for ``fail_count``. Values equal to
    zero are exempt from FAIL (manual: "POn != 0"). Missing values break a run.
    """
    v = np.asarray(values, dtype=np.float64)
    flags = _init_flags(v)
    run = 1
    for i in range(1, v.size):
        if np.isfinite(v[i]) and np.isfinite(v[i - 1]) and abs(v[i] - v[i - 1]) <= tolerance:
            run += 1
        else:
            run = 1
        if not np.isfinite(v[i]):
            continue
        if run >= fail_count and v[i] != 0:
            flags[i] = int(QCFlag.FAIL)
        elif run >= suspect_count:
            flags[i] = int(QCFlag.SUSPECT)
    return QCTestResult(
        "flat_line",
        "sensor",
        flags,
        {"suspect_count": suspect_count, "fail_count": fail_count, "tolerance": tolerance},
        _REF + " Test 16",
    )


def rate_of_change_test(
    values: ArrayLike, times: ArrayLike, max_change: float, *, per_seconds: float | None = None
) -> QCTestResult:
    """Test 20 (rate of change) - FAIL when ``|x_n - x_{n-1}|`` exceeds a threshold.

    The manual's example is ``MAXHSDIFF = 2 m`` between successive records.
    If ``per_seconds`` is given, the threshold scales with the actual time
    separation (``max_change * dt / per_seconds``) so irregular sampling does
    not trigger false failures. Applies only to integral parameters (Hs, mean
    period), not to discrete peak period/direction (manual test exception).
    """
    v = np.asarray(values, dtype=np.float64)
    t = _epoch_seconds(times)
    flags = _init_flags(v)
    if v.size > 1:
        diff = np.abs(np.diff(v))
        thresh = np.full(diff.shape, max_change)
        if per_seconds is not None:
            thresh = max_change * np.maximum(np.diff(t), 1.0) / per_seconds
        bad = np.isfinite(diff) & (diff > thresh)
        flags[1:][bad] = int(QCFlag.FAIL)
    return QCTestResult(
        "rate_of_change",
        "temporal",
        flags,
        {"max_change": max_change, "per_seconds": per_seconds},
        _REF + " Test 20",
    )


def mean_std_test(
    values: ArrayLike,
    times: ArrayLike,
    *,
    window: str = "24h",
    n_std: float = 3.0,
    min_periods: int = 6,
) -> QCTestResult:
    """Test 15 (mean and standard deviation) using a trailing time window.

    SUSPECT when a value lies outside ``mean +/- n_std * std`` of the preceding
    ``window`` (exclusive of the value itself, so a single outlier cannot mask
    itself).
    """
    v = np.asarray(values, dtype=np.float64)
    s = pd.Series(v, index=pd.to_datetime(np.asarray(times), utc=True))
    roll = s.rolling(window, closed="left", min_periods=min_periods)
    mu, sd = roll.mean().to_numpy(), roll.std().to_numpy()
    flags = _init_flags(v)
    ok = np.isfinite(v) & np.isfinite(mu) & np.isfinite(sd)
    flags[ok & (np.abs(v - mu) > n_std * sd)] = int(QCFlag.SUSPECT)
    return QCTestResult(
        "mean_std",
        "temporal",
        flags,
        {"window": window, "n_std": n_std, "min_periods": min_periods},
        _REF + " Test 15",
    )


def operational_frequency_range_test(
    frequency: ArrayLike,
    instrument_span: tuple[float, float],
    local_span: tuple[float, float] | None = None,
) -> QCTestResult:
    """Test 17 - flag spectral *bins* outside the instrument (FAIL) or local (SUSPECT) range."""
    f = np.asarray(frequency, dtype=np.float64)
    flags = _init_flags(f)
    flags[(f < instrument_span[0]) | (f > instrument_span[1])] = int(QCFlag.FAIL)
    if local_span is not None:
        sus = (flags == int(QCFlag.PASS)) & ((f < local_span[0]) | (f > local_span[1]))
        flags[sus] = int(QCFlag.SUSPECT)
    return QCTestResult(
        "operational_frequency_range",
        "sensor",
        flags,
        {
            "instrument_span": list(instrument_span),
            "local_span": list(local_span) if local_span else None,
        },
        _REF + " Test 17",
    )


def low_frequency_energy_test(
    frequency: ArrayLike,
    density: ArrayLike,
    bandwidth: ArrayLike,
    *,
    f_cutoff: float,
    energy_span: tuple[float, float],
) -> QCTestResult:
    """Test 18 - SUSPECT spectra whose energy below ``f_cutoff`` is outside ``energy_span`` [m^2].

    Detects spurious low-frequency energy (e.g. hull/mooring response or
    integration drift). The manual cautions that infragravity-dominated
    sites need broad limits.
    """
    f = np.asarray(frequency, dtype=np.float64)
    S = np.atleast_2d(np.asarray(density, dtype=np.float64))
    df = np.asarray(bandwidth, dtype=np.float64)
    low = f < f_cutoff
    E = np.nansum(S[:, low] * df[low], axis=1)
    flags = np.full(S.shape[0], int(QCFlag.PASS), dtype=np.int8)
    flags[np.all(np.isnan(S), axis=1)] = int(QCFlag.MISSING)
    flags[(flags == int(QCFlag.PASS)) & ((energy_span[0] > E) | (energy_span[1] < E))] = int(
        QCFlag.SUSPECT
    )
    return QCTestResult(
        "low_frequency_energy",
        "sensor",
        flags,
        {"f_cutoff": f_cutoff, "energy_span": list(energy_span)},
        _REF + " Test 18",
    )


def neighbor_test(values: ArrayLike, neighbor_values: ArrayLike, max_delta: float) -> QCTestResult:
    """Test 21 - SUSPECT when ``|W1 - W2| > max_delta`` (no FAIL by design; manual §3.3.3).

    The caller is responsible for time-aligning the neighbour series (e.g. with
    :func:`wave_analysis.processing.synchronize.pair_nearest`).
    """
    a = np.asarray(values, dtype=np.float64)
    b = np.asarray(neighbor_values, dtype=np.float64)
    flags = _init_flags(a)
    both = np.isfinite(a) & np.isfinite(b)
    flags[both & (np.abs(a - b) > max_delta)] = int(QCFlag.SUSPECT)
    flags[np.isfinite(a) & ~np.isfinite(b)] = int(QCFlag.NOT_EVALUATED)
    return QCTestResult("neighbor", "physical", flags, {"max_delta": max_delta}, _REF + " Test 21")
