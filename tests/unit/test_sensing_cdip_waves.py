"""Test complete upcross waves and two-row QC on a synthetic OPeNDAP subset."""

import importlib
from pathlib import Path


def test_per_wave_targets(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    m = importlib.import_module("sensing_cdip_waves")
    n = 50
    bounds = "\n".join(f"[{i}], {i * 1000}, {(i + 1) * 1000}" for i in range(n))
    payload = f"zcrossTimeBounds[{n}][2]\n{bounds}\n\n"
    for name, values in [
        ("zcrossZPeak", [1, -1] * (n // 2)),
        ("zcrossHeight", [3, 1] * (n // 2)),
        ("zcrossPeriod", [2] * n),
        ("zcrossFlagPrimary", [1, 2] + [1] * (n - 2)),
    ]:
        payload += name + f"[{n}]\n" + ", ".join(map(str, values)) + "\n\n"
    r = m.targets(payload, 0, 50)
    assert r["good_wave_count"] == 24 and r["incl_unevaluated_wave_count"] == 25
    assert r["good_mean_height_m"] == 3 and r["good_tz_s"] == 2
