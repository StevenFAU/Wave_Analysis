"""NDBC parsers on real (trimmed) upstream files spanning every format era."""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest

from wave_analysis.processing.spectra import bulk_parameters
from wave_analysis.sources.ndbc import parsers as P
from wave_analysis.sources.ndbc.parsers import assemble_spectrum


def _hist(ndbc_fixtures, name):
    return ndbc_fixtures / "historical" / f"{name}.txt.gz"


@pytest.mark.parametrize(
    ("name", "era", "first_time", "wdir", "wvht"),
    [
        ("41010h1995", "yy_hour", "1995-01-01T00:00Z", 150.0, 1.80),
        ("41010h2003", "yyyy_hour", "2003-01-01T00:00Z", 156.0, 2.29),
        ("41010h2006", "yyyy_minute", "2006-01-01T00:00Z", 227.0, 0.92),
    ],
)
def test_stdmet_legacy_formats(ndbc_fixtures, name, era, first_time, wdir, wvht):
    t = P.parse_stdmet(_hist(ndbc_fixtures, name))
    assert t.format_era == era
    assert t.units_inferred
    assert t.data.index[0] == pd.Timestamp(first_time)
    assert t.data["WDIR"].iloc[0] == wdir  # 'WD' header aliased to WDIR
    assert t.data["WVHT"].iloc[0] == pytest.approx(wvht)
    assert "PRES" in t.data.columns  # 'BAR' aliased to PRES
    assert t.units["VIS"] == "mi"
    # 999 MWD and 999.0 DEWP are missing sentinels in these files
    assert t.data["MWD"].isna().iloc[0]


def test_stdmet_modern_historical(ndbc_fixtures):
    t = P.parse_stdmet(_hist(ndbc_fixtures, "41010h2023"))
    assert t.format_era == "yyyy_minute_units"
    assert not t.units_inferred
    assert t.units["WVHT"] == "m" and t.units["VIS"] == "mi"
    assert str(t.data.index.tz) == "UTC"
    assert t.data.index.is_monotonic_increasing
    assert t.data["WVHT"].notna().sum() > 50
    # No sentinel value may survive parsing.
    for col, sentinel in P.STDMET_SENTINELS.items():
        if col in t.data:
            assert not np.isclose(t.data[col], sentinel).any(), col


def test_stdmet_realtime(ndbc_fixtures):
    t = P.parse_stdmet(ndbc_fixtures / "realtime" / "41010.txt")
    assert t.meta["realtime"]
    assert t.units["VIS"] == "nmi"
    assert t.data["WVHT"].isna().iloc[0]  # 'MM' -> NaN
    assert t.data["PTDY"].notna().any()  # '+0.5' style signed values parse


def test_wdir_99_is_valid_not_missing():
    text = (
        "#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS  TIDE\n"
        "#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC   mi    ft\n"
        "2023 01 01 00 00  99  5.0  6.0  1.00  8.00  6.00  99 1015.0  20.0  22.0  15.0 99.0 99.00\n"
    )
    t = P.parse_stdmet(gzip.compress(text.encode()))
    assert t.data["WDIR"].iloc[0] == 99.0
    assert t.data["MWD"].iloc[0] == 99.0
    assert np.isnan(t.data["VIS"].iloc[0])
    assert np.isnan(t.data["TIDE"].iloc[0])


def test_spectral_summary(ndbc_fixtures):
    t = P.parse_spectral_summary(ndbc_fixtures / "realtime" / "41010.spec")
    row = t.data.iloc[0]
    assert row["SwD"] == "NNE" and row["SwD_deg"] == 22.5
    assert row["STEEPNESS"] in {"SWELL", "AVERAGE", "STEEP", "VERY_STEEP"}


def test_historical_spectral_and_r_scaling(ndbc_fixtures):
    den = P.parse_spectral(_hist(ndbc_fixtures, "41010w2023"), "swden")
    r1 = P.parse_spectral(_hist(ndbc_fixtures, "41010j2023"), "swr1")
    assert len(den) == 1 and den[0].frequency.size == 47
    assert den[0].frequency[0] == pytest.approx(0.02)
    assert den[0].frequency[-1] == pytest.approx(0.485)
    vals = r1[0].values[np.isfinite(r1[0].values)]
    assert vals.max() <= 1.0 and vals.min() >= 0.0  # un-scaled from hundredths
    # First row of the raw file: r1 at 0.02 Hz is "12" -> 0.12
    assert r1[0].values[0, 0] == pytest.approx(0.12)


def test_historical_2003_grid_without_minutes(ndbc_fixtures):
    b = P.parse_spectral(_hist(ndbc_fixtures, "41010w2003"), "swden")[0]
    np.testing.assert_allclose(b.frequency[:3], [0.03, 0.04, 0.05])
    assert b.time[0] == np.datetime64("2003-01-01T00:00")


def test_realtime_spectral_pairs_and_separation(ndbc_fixtures):
    den = P.parse_spectral(ndbc_fixtures / "realtime" / "41010.data_spec", "swden")[0]
    r1 = P.parse_spectral(ndbc_fixtures / "realtime" / "41010.swr1", "swr1")[0]
    assert den.separation_frequency is not None
    assert np.all((den.separation_frequency > 0) & (den.separation_frequency < 1))
    assert np.all(np.diff(den.time.astype("int64")) > 0)  # sorted ascending
    vals = r1.values[np.isfinite(r1.values)]
    assert vals.min() >= 0 and vals.max() <= 1  # realtime r is NOT scaled
    assert np.isnan(r1.values[0, 0])  # 999.00 -> NaN


def test_grid_change_is_split_into_blocks():
    text = (
        "#YY  MM DD hh mm Sep_Freq  < spec_1 (freq_1) spec_2 (freq_2) ... >\n"
        "2026 01 01 00 40 0.100 1.0 (0.050) 2.0 (0.100)\n"
        "2026 01 01 01 40 0.100 1.0 (0.050) 2.0 (0.100) 3.0 (0.150)\n"
    )
    blocks = P.parse_spectral(text.encode(), "swden", realtime=True)
    assert [b.frequency.size for b in blocks] == [2, 3]


def test_hm0_reproduces_reported_wvht(ndbc_fixtures):
    """Key validation: 4*sqrt(m0) of the NDBC spectrum reproduces stdmet WVHT to rounding.

    Full-year result (scripts/validate_ndbc_hm0.py): mean diff 0.0002 m, SD 0.005 m, n = 4281.
    """
    den = P.parse_spectral(_hist(ndbc_fixtures, "41010w2023"), "swden")[0]
    spec = assemble_spectrum({"swden": den})
    hm0 = bulk_parameters(spec).set_index("time_utc")["Hm0"]
    wvht = P.parse_stdmet(_hist(ndbc_fixtures, "41010h2023")).data["WVHT"].dropna()
    joined = pd.concat([hm0, wvht], axis=1, join="inner").dropna()
    assert len(joined) >= 20
    diff = joined["Hm0"] - joined["WVHT"]
    assert abs(diff.mean()) < 0.01
    assert diff.abs().max() <= 0.02


def test_assemble_full_directional(ndbc_fixtures):
    codes = {"swden": "w", "swdir": "d", "swdir2": "i", "swr1": "j", "swr2": "k"}
    blocks = {
        p: P.parse_spectral(_hist(ndbc_fixtures, f"41010{c}2023"), p)[0] for p, c in codes.items()
    }
    spec = assemble_spectrum(blocks)
    assert spec.is_directional
    b = bulk_parameters(spec)
    assert b["Dm"].between(0, 360).all()
    assert b["spread_m"].dropna().between(0, 81.1).all()  # max sqrt(2) rad


def test_assemble_rejects_mismatched_grid(ndbc_fixtures):
    den = P.parse_spectral(_hist(ndbc_fixtures, "41010w2023"), "swden")[0]
    bad = P.SpectralBlock("swr1", den.time, den.frequency[:-1], den.values[:, :-1])
    with pytest.raises(ValueError, match="frequency grid"):
        assemble_spectrum({"swden": den, "swr1": bad})
    with pytest.raises(ValueError, match="swden"):
        assemble_spectrum({"swr1": bad})


def test_adcp_long_format(ndbc_fixtures):
    df = P.parse_adcp(_hist(ndbc_fixtures, "41001a2010"))
    assert list(df.columns) == ["time_utc", "bin", "depth_m", "direction_degT", "speed_cm_s"]
    assert df["bin"].min() == 1
    assert df.iloc[0]["speed_cm_s"] == 47.0


def test_ocean_product_generic_sentinels(ndbc_fixtures):
    t = P.parse_stdmet(_hist(ndbc_fixtures, "32st1o2010"), product="ocean")
    assert t.data["OTMP"].iloc[0] == pytest.approx(19.1)
    assert t.data["COND"].isna().all()  # 999.00
    assert t.data["TURB"].isna().iloc[0]  # 99


def test_accepts_str_path_and_raw_bytes(ndbc_fixtures):
    path = _hist(ndbc_fixtures, "41010h2006")
    a = P.parse_stdmet(str(path)).data
    b = P.parse_stdmet(path.read_bytes()).data
    pd.testing.assert_frame_equal(a, b)


def test_bad_header_raises():
    with pytest.raises(ValueError):
        P.parse_stdmet(b"FOO BAR BAZ\n1 2 3\n")
    with pytest.raises(ValueError):
        P.parse_stdmet(b"")
