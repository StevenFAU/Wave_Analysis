"""Parsers for NDBC text products (realtime and historical).

The parsers are *transcriptions*: they convert text to typed arrays, replace
documented missing-value sentinels with NaN, and undo documented transport
scaling (r1/r2 x100 in historical files). They do **not** rename variables to
CF names or convert units - that is ``normalize``'s job (``sources/ndbc/source.py``).

Format history handled (all verified on real files, see ``tests/fixtures/ndbc``):

==================  =============================================================
Era                 Header / layout
==================  =============================================================
<= 1998             ``YY MM DD hh WD WSPD ... BAR ...``      2-digit year, no minute
1999 - 2004         ``YYYY MM DD hh WD ...``                 4-digit year, no minute
2005 - 2006         ``YYYY MM DD hh mm WD ...``              minute column added
>= 2007             ``#YY MM DD hh mm WDIR ...`` + ``#yr mo dy hr mn ...`` units row
realtime            as >= 2007, ``MM`` for missing values
==================  =============================================================

Spectral files: historical files carry the frequency grid in the header
(which itself changed, e.g. 0.03-0.40 Hz at 0.01 Hz in 2003 vs the 47-band
0.02-0.485 Hz grid later); realtime files carry ``value (frequency)`` pairs
on every row, and ``data_spec`` adds a separation frequency column.

Missing values (NDBC measdes): realtime ``MM``; historical "a variable number of
9's ... depending on the data type". Sentinels are therefore matched **per
column** (e.g. ``WDIR`` 999 is missing but 99 degrees is a valid direction).
"""

from __future__ import annotations

import gzip
import io
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from wave_analysis.sources.ndbc.products import SCALED_R_PRODUCTS

_DATE_TOKENS = {"YY", "YYYY", "MM", "DD", "hh", "mm", "yr", "mo", "dy", "hr", "mn"}
_COLUMN_ALIASES = {"WD": "WDIR", "BAR": "PRES", "DIR": "DIR", "SPD": "SPD"}

#: Per-column historical missing-value sentinels.
STDMET_SENTINELS: dict[str, float] = {
    "WDIR": 999,
    "WSPD": 99.0,
    "GST": 99.0,
    "WVHT": 99.0,
    "DPD": 99.0,
    "APD": 99.0,
    "MWD": 999,
    "PRES": 9999.0,
    "ATMP": 999.0,
    "WTMP": 999.0,
    "DEWP": 999.0,
    "VIS": 99.0,
    "PTDY": 99.0,
    "TIDE": 99.0,
}
_GENERIC_SENTINELS = (99.0, 999.0, 9999.0, 99999.0)

#: Units assumed when a file has no units row (pre-2007 historical files).
DEFAULT_UNITS_HISTORICAL: dict[str, str] = {
    "WDIR": "degT",
    "WSPD": "m/s",
    "GST": "m/s",
    "WVHT": "m",
    "DPD": "sec",
    "APD": "sec",
    "MWD": "degT",
    "PRES": "hPa",
    "ATMP": "degC",
    "WTMP": "degC",
    "DEWP": "degC",
    "VIS": "mi",
    "PTDY": "hPa",
    "TIDE": "ft",
}

_COMPASS = {
    name: i * 22.5
    for i, name in enumerate(
        [
            "N",
            "NNE",
            "NE",
            "ENE",
            "E",
            "ESE",
            "SE",
            "SSE",
            "S",
            "SSW",
            "SW",
            "WSW",
            "W",
            "WNW",
            "NW",
            "NNW",
        ]
    )
}


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #


def read_text(src: bytes | str | Path) -> str:
    """Read NDBC text from raw bytes or a file path (``str`` or ``Path``), gunzipping if needed.

    Gzip is detected by magic number, not by file extension.
    """
    raw = Path(src).read_bytes() if isinstance(src, str | Path) else src
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("ascii", errors="replace")


def _split_header(text: str) -> tuple[list[str], list[str] | None, list[str]]:
    """Return (header tokens, units tokens or None, data lines)."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty NDBC file")
    header = lines[0].lstrip("#").split()
    units: list[str] | None = None
    start = 1
    if len(lines) > 1 and lines[1].startswith("#"):
        units = lines[1].lstrip("#").split()
        start = 2
    data = [ln for ln in lines[start:] if not ln.startswith("#")]
    return header, units, data


def _n_date_columns(header: list[str]) -> int:
    n = 0
    for tok in header:
        if tok in _DATE_TOKENS:
            n += 1
        else:
            break
    if n not in (4, 5):
        raise ValueError(f"unrecognised NDBC date header: {header[:6]}")
    return n


def _times_from_columns(cols: NDArray[Any]) -> NDArray[np.datetime64]:
    """Build UTC datetime64[ns] from an (n, 4|5) array of Y M D h [m]."""
    c = cols.astype(np.int64)
    year = c[:, 0]
    year = np.where(year < 100, year + 1900, year)  # 2-digit years exist only pre-1999
    minute = c[:, 4] if c.shape[1] >= 5 else np.zeros(len(c), dtype=np.int64)
    ts = pd.to_datetime(
        {"year": year, "month": c[:, 1], "day": c[:, 2], "hour": c[:, 3], "minute": minute}
    )
    return ts.to_numpy(dtype="datetime64[ns]")


def _format_era(header: list[str], units: list[str] | None, n_date: int) -> str:
    first = header[0]
    if units is not None:
        return "yyyy_minute_units"
    if first == "YY":
        return "yy_hour" if n_date == 4 else "yy_minute"
    return "yyyy_hour" if n_date == 4 else "yyyy_minute"


# --------------------------------------------------------------------------- #
# Tabular products (stdmet, ocean, cwind, supl, srad, spec)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ParsedTable:
    """A parsed tabular NDBC product in provider names and units."""

    product: str
    data: pd.DataFrame  # index: time_utc (tz-aware); columns: provider names
    units: dict[str, str]
    format_era: str
    units_inferred: bool
    meta: dict[str, Any] = field(default_factory=dict)


def _apply_sentinels(df: pd.DataFrame, sentinels: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        s = pd.to_numeric(out[col], errors="coerce")
        if col in sentinels:
            s = s.mask(np.isclose(s, sentinels[col]))
        else:
            s = s.mask(s.isin(_GENERIC_SENTINELS))
        out[col] = s.astype("float64")
    return out


def parse_stdmet(
    src: bytes | str | Path, *, product: str = "stdmet", realtime: bool | None = None
) -> ParsedTable:
    """Parse standard-meteorological-style tables (also ocean, cwind, supl, srad).

    ``realtime`` only affects the default-unit assumption for files lacking a
    units row; it is inferred from the presence of ``MM`` tokens when ``None``.
    """
    text = read_text(src)
    header, units_row, data_lines = _split_header(text)
    n_date = _n_date_columns(header)
    names = [_COLUMN_ALIASES.get(h, h) for h in header[n_date:]]
    if realtime is None:
        # Realtime files mark missing data with an "MM" *data token*; the header's
        # month column is also "MM", so only data lines are inspected.
        realtime = any("MM" in ln.split() for ln in data_lines[:500])
    if not data_lines:
        empty = pd.DataFrame(columns=names, index=pd.DatetimeIndex([], tz="UTC", name="time_utc"))
        return ParsedTable(
            product, empty, {}, _format_era(header, units_row, n_date), units_row is None
        )

    raw = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        sep=r"\s+",
        header=None,
        dtype=str,
        na_values=["MM"],
        keep_default_na=False,
        engine="python",
    )
    raw = raw.reindex(columns=range(n_date + len(names)))
    times = _times_from_columns(raw.iloc[:, :n_date].to_numpy())
    values = raw.iloc[:, n_date:].copy()
    values.columns = pd.Index(names)
    values = _apply_sentinels(values, STDMET_SENTINELS)
    values.index = pd.DatetimeIndex(times, name="time_utc").tz_localize("UTC")

    if units_row is not None:
        units = dict(zip(names, units_row[n_date:], strict=False))
        inferred = False
    else:
        units = {n: DEFAULT_UNITS_HISTORICAL.get(n, "unknown") for n in names}
        if realtime and "VIS" in units:
            units["VIS"] = "nmi"
        inferred = True
    return ParsedTable(
        product=product,
        data=values,
        units=units,
        format_era=_format_era(header, units_row, n_date),
        units_inferred=inferred,
        meta={"n_rows": len(values), "realtime": bool(realtime)},
    )


def parse_spectral_summary(src: bytes | str | Path) -> ParsedTable:
    """Parse the realtime ``.spec`` spectral summary.

    Compass-point directions (``SwD``, ``WWD``) are converted to degrees
    (resolution 22.5 deg) in new ``*_deg`` columns, keeping the original strings.
    ``STEEPNESS`` is NDBC's categorical label (SWELL, AVERAGE, STEEP, VERY_STEEP).
    """
    text = read_text(src)
    header, units_row, data_lines = _split_header(text)
    n_date = _n_date_columns(header)
    names = header[n_date:]
    rows = [ln.split() for ln in data_lines]
    times = _times_from_columns(np.array([r[:n_date] for r in rows]))
    df = pd.DataFrame([r[n_date:] for r in rows], columns=names)
    df.index = pd.DatetimeIndex(times, name="time_utc").tz_localize("UTC")
    categorical = {"SwD", "WWD", "STEEPNESS"}
    for c in names:
        if c in categorical:
            df[c] = df[c].where(~df[c].isin(["MM", "N/A"]))
        else:
            df[c] = pd.to_numeric(df[c].replace({"MM": None, "N/A": None}), errors="coerce")
    for c in ("SwD", "WWD"):
        if c in df:
            df[f"{c}_deg"] = df[c].map(_COMPASS).astype("float64")
    units = dict(zip(names, units_row[n_date:], strict=False)) if units_row else {}
    return ParsedTable("spec", df, units, _format_era(header, units_row, n_date), units_row is None)


# --------------------------------------------------------------------------- #
# Spectral products (swden/data_spec, swdir, swdir2, swr1, swr2)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SpectralBlock:
    """Rows of one spectral product sharing a single frequency grid."""

    product: str
    time: NDArray[np.datetime64]
    frequency: NDArray[np.float64]
    values: NDArray[np.float64]
    separation_frequency: NDArray[np.float64] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


_PAIR = re.compile(r"(\S+)\s+\(\s*([0-9.]+)\s*\)")


def _spectral_missing(v: NDArray[np.float64]) -> NDArray[np.float64]:
    out = v.copy()
    out[out >= 998.9] = np.nan
    return out


def _group_by_grid(
    product: str,
    times: list[np.datetime64],
    grids: list[tuple[float, ...]],
    values: list[list[float]],
    seps: list[float] | None,
    meta: dict[str, Any],
) -> list[SpectralBlock]:
    blocks: list[SpectralBlock] = []
    order: dict[tuple[float, ...], list[int]] = {}
    for i, g in enumerate(grids):
        order.setdefault(g, []).append(i)
    for g, idx in order.items():
        vals = _spectral_missing(np.array([values[i] for i in idx], dtype=np.float64))
        if product in SCALED_R_PRODUCTS and not meta.get("realtime", False):
            vals = vals / 100.0
        t = np.array([times[i] for i in idx], dtype="datetime64[ns]")
        sort = np.argsort(t, kind="stable")
        sep = None
        if seps is not None:
            sep = _spectral_missing(np.array([seps[i] for i in idx], dtype=np.float64))[sort]
        blocks.append(SpectralBlock(product, t[sort], np.array(g), vals[sort], sep, dict(meta)))
    blocks.sort(key=lambda b: b.time[0] if b.time.size else np.datetime64("NaT"))
    return blocks


def parse_spectral(
    src: bytes | str | Path, product: str, *, realtime: bool | None = None
) -> list[SpectralBlock]:
    """Parse a spectral product file into one or more :class:`SpectralBlock` s.

    Parameters
    ----------
    product
        One of ``swden``, ``swdir``, ``swdir2``, ``swr1``, ``swr2``.
    realtime
        Whether the file is in realtime format (value/frequency pairs per row).
        Detected from the content when ``None``. Controls r1/r2 un-scaling:
        historical r values are divided by 100 (NDBC measdes).

    Returns
    -------
    list[SpectralBlock]
        Normally one block; more than one if the frequency grid changes within
        the file (e.g. a payload change mid-year).
    """
    text = read_text(src)
    header, _units, data_lines = _split_header(text)
    n_date = _n_date_columns(header)
    if realtime is None:
        realtime = "(" in (data_lines[0] if data_lines else "")
    meta = {"realtime": realtime, "n_date_columns": n_date}

    times: list[np.datetime64] = []
    grids: list[tuple[float, ...]] = []
    values: list[list[float]] = []
    seps: list[float] | None = [] if (realtime and product == "swden") else None

    if realtime:
        date_rows = []
        for ln in data_lines:
            toks = ln.split()
            date_rows.append(toks[:n_date])
            rest = ln.split(None, n_date)[n_date] if len(toks) > n_date else ""
            if seps is not None:
                first, _, rest = rest.partition(" ")
                seps.append(float(first) if first not in ("MM", "") else np.nan)
            pairs = _PAIR.findall(rest)
            grids.append(tuple(round(float(f), 5) for _, f in pairs))
            values.append([float(v) if v != "MM" else np.nan for v, _ in pairs])
        if date_rows:
            times = list(_times_from_columns(np.array(date_rows)))
    else:
        freqs = tuple(round(float(f), 5) for f in header[n_date:])
        rows = [ln.split() for ln in data_lines]
        if rows:
            arr = np.array(
                [r[: n_date + len(freqs)] for r in rows if len(r) >= n_date + len(freqs)]
            )
            times = list(_times_from_columns(arr[:, :n_date]))
            vals = arr[:, n_date:].astype(np.float64)
            values = vals.tolist()
            grids = [freqs] * len(values)
    return _group_by_grid(product, times, grids, values, seps, meta)


# --------------------------------------------------------------------------- #
# ADCP
# --------------------------------------------------------------------------- #


def parse_adcp(src: bytes | str | Path) -> pd.DataFrame:
    """Parse the NDBC ``adcp`` (``a``) current-profile format into long form.

    Rows have a variable number of ``DEPnn DIRnn SPDnn`` triplets. Output
    columns: ``time_utc, bin, depth_m, direction_degT, speed_cm_s``.
    """
    text = read_text(src)
    header, _units, data_lines = _split_header(text)
    n_date = _n_date_columns(header)
    records: list[tuple[Any, ...]] = []
    date_rows: list[list[str]] = []
    triples: list[list[str]] = []
    for ln in data_lines:
        toks = ln.split()
        date_rows.append(toks[:n_date])
        triples.append(toks[n_date:])
    if not date_rows:
        return pd.DataFrame(columns=["time_utc", "bin", "depth_m", "direction_degT", "speed_cm_s"])
    times = _times_from_columns(np.array(date_rows))
    for t, vals in zip(times, triples, strict=True):
        for b in range(len(vals) // 3):
            d, di, s = (float(x) for x in vals[3 * b : 3 * b + 3])
            records.append((t, b + 1, d, di, s))
    df = pd.DataFrame(
        records, columns=["time_utc", "bin", "depth_m", "direction_degT", "speed_cm_s"]
    )
    for c in ("depth_m", "direction_degT", "speed_cm_s"):
        df[c] = df[c].mask(df[c].isin(_GENERIC_SENTINELS))
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    return df


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def assemble_spectrum(
    blocks: dict[str, SpectralBlock], *, meta: dict[str, Any] | None = None
) -> Any:
    """Combine per-product spectral blocks into a :class:`FrequencySpectrum`.

    ``swden`` is required and defines the time axis; directional products are
    aligned to it by exact timestamp (rows absent from a directional file
    become NaN - they are *not* dropped). All blocks must share the ``swden``
    frequency grid; mismatches raise rather than being silently interpolated.
    """
    from wave_analysis.processing.spectra import FrequencySpectrum

    if "swden" not in blocks:
        raise ValueError("assemble_spectrum requires an 'swden' block")
    den = blocks["swden"]
    kw: dict[str, Any] = {}
    for prod, name in (("swdir", "alpha1"), ("swdir2", "alpha2"), ("swr1", "r1"), ("swr2", "r2")):
        b = blocks.get(prod)
        if b is None:
            continue
        if b.frequency.shape != den.frequency.shape or not np.allclose(b.frequency, den.frequency):
            raise ValueError(f"{prod} frequency grid differs from swden; regrid explicitly")
        s = pd.DataFrame(b.values, index=pd.DatetimeIndex(b.time))
        s = s[~s.index.duplicated(keep="first")]
        kw[name] = s.reindex(pd.DatetimeIndex(den.time)).to_numpy(dtype=np.float64)
    return FrequencySpectrum(
        time=den.time,
        frequency=den.frequency,
        density=den.values,
        separation_frequency=den.separation_frequency,
        meta={**(meta or {}), "time_reference": "nominal", "products": sorted(blocks)},
        **kw,
    )


def iter_realtime_or_historical(paths: Iterable[Path]) -> Iterable[tuple[Path, bool]]:
    """Yield ``(path, is_realtime)`` using the file-name convention (``.txt.gz`` = historical)."""
    for p in paths:
        yield p, not p.name.endswith(".txt.gz")
