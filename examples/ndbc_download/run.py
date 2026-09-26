"""NDBC download → parse → normalize → derive → QC walkthrough (network required)."""

import sys
import tempfile
from pathlib import Path

import pandas as pd

from wave_analysis.ingest.downloader import Downloader
from wave_analysis.processing.spectra import bulk_parameters
from wave_analysis.qc.flags import QCFlag
from wave_analysis.schemas.observation import validate_observation_frame
from wave_analysis.sources.ndbc import SPECTRAL_SET, NDBCSource, assemble_spectrum


def main(station: str = "41010", year: int = 2023) -> None:
    with tempfile.TemporaryDirectory() as tmp, Downloader() as dl:
        src = NDBCSource(dl)
        entries = src.fetch(station, ["stdmet", *SPECTRAL_SET], dest_root=Path(tmp), years=[year])
        for e in entries:
            print(
                f"{e.product:7s} {e.status.value:10s} {e.size_bytes or 0:>9,d} B  sha256={str(e.sha256)[:12]}…"
            )
        files = {e.product: Path(e.local_path) for e in entries if e.local_path}

        obs = src.normalize(
            src.parse(files["stdmet"], "stdmet"),
            station_id=station,
            source_file=files["stdmet"].name,
        )
        validate_observation_frame(obs)
        print(
            f"\n{len(obs):,} standardized observations; variables: {sorted(obs['variable'].unique())}"
        )

        blocks = {p: src.parse(files[p], p)[0] for p in SPECTRAL_SET if p in files}
        spec = assemble_spectrum(blocks)
        bulk = bulk_parameters(spec)
        print(
            f"\n{spec.n_time:,} directional spectra on {spec.frequency.size} bins "
            f"({spec.frequency[0]:.4f}–{spec.frequency[-1]:.3f} Hz)"
        )
        print(bulk[["Hm0", "Tp_smooth", "Tm02", "Dm", "spread_m"]].describe().round(2).to_string())

        for r in src.validate(obs):
            counts = {k: v for k, v in r.counts().items() if v}
            print(f"QC {r.test:15s} {counts}")
        passed = pd.Series(next(r for r in src.validate(obs) if r.test == "gross_range_hs").flags)
        print(f"\nH_s gross-range PASS rate: {(passed == QCFlag.PASS).mean():.1%}")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["41010"]), *(int(y) for y in sys.argv[2:3]))
