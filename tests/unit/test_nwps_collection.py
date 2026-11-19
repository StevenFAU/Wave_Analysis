"""Guard the NWPS file-selection and rerun behavior with a mock forecast office."""

import datetime

import httpx

from wave_analysis.ingest.downloader import Downloader
from wave_analysis.ingest.manifest import read_manifest
from wave_analysis.sources import nwps


def test_cycle_sources_and_rerun(tmp_path, monkeypatch):
    root = "https://test.invalid/"
    base = root + "er.20261115/mhx/"
    f = "nwps.t00z."
    payloads = {
        base: '<a href="00/">',
        base + "00/CG0/": f'<a href="{f}cg0_trkng.mhx.grib2">',
        base
        + "00/CG1/": f'<a href="{f}spc2d_41025_cg1.mhx.txt"><a href="{f}cg1.mhx.grib2"><a href="{f}ripprob.txt">',
        base + "00/CG0/" + f + "cg0_trkng.mhx.grib2": "GRIB",
        base + "00/CG1/" + f + "spc2d_41025_cg1.mhx.txt": "SWAN",
    }
    calls = []

    def handle(req):
        calls.append(str(req.url))
        return httpx.Response(200, text=payloads[str(req.url)])

    monkeypatch.setattr(
        nwps,
        "Downloader",
        lambda **kw: Downloader(min_interval_s=0, transport=httpx.MockTransport(handle)),
    )
    monkeypatch.setenv("WAVE_ANALYSIS_DATA", str(tmp_path))
    for _ in range(2):
        nwps.collect({"base": root, "offices": {"er": ["mhx"]}}, datetime.date(2026, 11, 15), 1)
    entries = list(read_manifest(tmp_path / "raw/nwps/_manifests/nwps.csv"))
    assert len(entries) == 2 and all(
        r.sha256 and r.local_path.startswith("data/raw/nwps/") for r in entries
    )
    assert calls.count(base + "00/CG0/" + f + "cg0_trkng.mhx.grib2") == 1
