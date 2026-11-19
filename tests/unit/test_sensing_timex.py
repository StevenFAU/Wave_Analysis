"""Test 9's package helpers (split by date, mask and resize) and probe script end to end (CPU)."""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]


def _mod(name: str):
    sys.path.insert(0, str(ROOT / "scripts"))
    return importlib.import_module(name)


def test_split_is_by_date_only():
    m = _mod("sensing_timex_package")
    days = pd.date_range("2024-03-01", "2026-06-30", freq="D").strftime("%Y-%m-%d")
    d = pd.DataFrame({"local_day": days})
    cfg = {"split": {"train_year": 2024, "val_week_share": 0.15, "seed": 0, "test_year": 2025,
                     "embargo_days": 3, "later_year": 2026}}  # fmt: skip
    s = m.assign_split(d, cfg)
    assert set(s) == {"train", "val", "test", "later", ""}  # "": the embargo days
    assert (
        s[d["local_day"].str.startswith("2025-01-0") & (d["local_day"] <= "2025-01-03")] == ""
    ).all()
    # whole weeks: a validation week never shares a week with a training day
    wk = pd.to_datetime(d["local_day"]).dt.isocalendar()
    w = wk["year"].astype(str) + wk["week"].astype(str)
    assert not set(w[s == "val"]) & set(w[s == "train"])
    assert s.equals(m.assign_split(d, cfg))  # deterministic


def test_resize_masks_currituck_caption(tmp_path):
    m = _mod("sensing_timex_package")
    src = tmp_path / "a.jpg"
    Image.fromarray(np.full((1520, 2688, 3), 200, np.uint8)).save(src)
    dest, sha = m._resize(
        (str(src), str(tmp_path / "o" / "b.jpg"), "still", 256, 90, [0, 170, 560, 170])
    )
    im = np.asarray(Image.open(dest))
    assert im.shape == (256, 452, 3) and len(sha) == 64
    assert im[-5, 5].max() < 20 and im[5, 5].min() > 180  # lower-left box black, rest untouched


def test_probe_end_to_end(tmp_path, monkeypatch):
    m = _mod("sensing_timex_probe")
    rng = np.random.default_rng(0)
    rows = []
    for split, year in [("train", 2024), ("val", 2024), ("test", 2025), ("later", 2026)]:
        for i in range(24):
            rows.append({"camera": "cam", "stamp": f"{year}-{split}-{i}", "local_day": f"{year}-06-{i % 8 + 1:02d}",
                         "split": split, "Hs_label_m": float(np.exp(rng.normal(0, 0.4)))})  # fmt: skip
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    pd.DataFrame(rows).to_csv(pkg / "triples.csv", index=False)
    (pkg / "SHA256SUMS").write_text("x  triples.csv\n")
    n = len(rows)
    fake = {k: rng.standard_normal((n, 2, 4, 16, 16)).astype(np.float16) for k in m.KINDS}
    monkeypatch.setattr(m, "extract", lambda d, cfg, args, cache: fake)
    rc = m.main(["--config", str(ROOT / "configs/experiments/sensing_timex_v1.yaml"), "--package", str(pkg),
                 "--out", str(tmp_path / "out"), "--feature-cache", str(tmp_path / "cache"),
                 "--epochs", "1", "--device", "cpu", "--num-workers", "0"])  # fmt: skip
    assert rc == 0
    res = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert set(res["bars"]) == {
        "T9-A (timex - still, H_s RMSE, 2025)",
        "T9-B (brt - still, H_s RMSE, 2025)",
    }
    p = pd.read_csv(tmp_path / "out" / "predictions.csv")
    assert {"pred_still", "pred_timex", "pred_brt", "pred_all3", "pred_climatology"} <= set(
        p.columns
    )
    assert m.classes(np.array([0.5, 0.51, 1.25, 2.5, 2.6]), [0.5, 1.25, 2.5]).tolist() == [
        0,
        1,
        1,
        2,
        3,
    ]
