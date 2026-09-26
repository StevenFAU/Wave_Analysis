#!/usr/bin/env python
"""Reproduce the check that NDBC WVHT equals 4*sqrt(m0) of the published spectrum.

Downloads one station-year of ``stdmet`` and ``swden`` (default 41010, 2023),
integrates each spectrum with midpoint bin widths, and compares with WVHT at
identical timestamps.

Result recorded in docs/datasets/ndbc.md (2026-09-25): n = 4281,
mean(Hm0 - WVHT) = +0.0002 m, SD = 0.005 m, max |diff| = 0.053 m. So WVHT is
the spectral Hm0 to within reporting precision, even though NDBC's glossary
describes WVHT as the average of the highest one-third of waves.

    uv run python scripts/validate_ndbc_hm0.py [station] [year]
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

from wave_analysis.ingest.downloader import Downloader
from wave_analysis.processing.spectra import bulk_parameters
from wave_analysis.sources.ndbc import NDBCSource, assemble_spectrum


def main(station: str = "41010", year: int = 2023) -> None:
    with tempfile.TemporaryDirectory() as tmp, Downloader() as dl:
        src = NDBCSource(dl)
        entries = src.fetch(station, ["stdmet", "swden"], dest_root=Path(tmp), years=[year])
        paths = {e.product: Path(e.local_path) for e in entries if e.local_path}
        wvht = src.parse(paths["stdmet"], "stdmet").data["WVHT"].dropna()
        blocks = src.parse(paths["swden"], "swden")
        hm0 = pd.concat(
            [
                bulk_parameters(assemble_spectrum({"swden": b})).set_index("time_utc")["Hm0"]
                for b in blocks
            ]
        )
    j = pd.concat([hm0, wvht], axis=1, join="inner").dropna()
    d = j["Hm0"] - j["WVHT"]
    print(
        f"{station} {year}: n={len(d)} mean={d.mean():+.4f} m sd={d.std():.4f} m max|d|={d.abs().max():.3f} m"
    )


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["41010"]), *(int(y) for y in sys.argv[2:3]))
