"""Test 1's nested calibration (scripts/sensing_nested_splits.py, scripts/sensing_nested_fusion.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wave_analysis.ingest.checksum import sha256_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import sensing_nested_fusion as nf  # noqa: E402
import sensing_nested_splits as ns  # noqa: E402


def _strips(n_groups: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        for i in range(20 + (7 * g) % 41):  # uneven group sizes, as the buoys' are
            rows.append({"station_group": f"g{g}", "station_id": f"{g}{i % 2}", "cv_fold": g % 5,
                         "swell_label_m": rng.uniform(0.5, 2), "wind_wave_label_m": rng.uniform(0.2, 1)})  # fmt: skip
    return pd.DataFrame(rows)


def test_pair_roles_hold_out_both_folds_and_validate_inside_the_pool(monkeypatch):
    import buoycam_cv_folds

    monkeypatch.setattr(buoycam_cv_folds, "N_DRAWS", 50)
    monkeypatch.setattr(buoycam_cv_folds, "VAL_MIN_BUOYS", 2)
    t = _strips()
    role, summary = ns.pair_roles(t)
    fold = t.groupby("station_group")["cv_fold"].first()
    assert len(summary) == 10
    for a, b in ns.PAIRS:
        name = ns.pair_name(a, b)
        for g, f in fold.items():
            assert (role[g][name] == "test") == (f in (a, b))
        assert any(role[g][name] == "val" for g in fold.index)


def test_calibration_takes_each_other_fold_from_its_pair_with_k():
    rows = [(ns.pair_name(a, b), str(f)) for a, b in ns.PAIRS for f in (a, b)]
    inner = pd.DataFrame(rows, columns=["pair", "cv_fold"])
    cal = nf.calibration(inner, "2")
    assert sorted(cal["cv_fold"]) == ["0", "1", "3", "4"]  # every other fold once
    assert all("2" in p[4:] for p in cal["pair"])  # each from a model that held out fold 2


def test_variance_weights_favour_the_smaller_residual():
    rng = np.random.default_rng(0)
    y = rng.uniform(0.3, 1.5, 400)
    cal = pd.DataFrame({"true_wind_wave_m": y, "cmp_hs": rng.uniform(1, 3, 400),
                        "cam_ww": y * 1.01, "cmp_ww": y * 1.3})  # fmt: skip
    cuts = np.quantile(cal["cmp_hs"], [0.25, 0.5, 0.75])
    vw = nf.VarWeight().fit(cal, cuts)
    assert all(v["camera_weight"] > 0.9 for v in vw.describe().values())
    test = cal.assign(cmp_hs=2.0)
    hs, ww = vw.predict(test.assign(cmp_hs=np.maximum(test["cmp_ww"], 2.0)))
    assert np.all(np.abs(ww - y) < np.abs(test["cmp_ww"] - y))
    assert np.all(hs >= ww)


def test_nested_builder_protects_existing_and_overlapping_paths(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    marker = src / "original"
    marker.write_text("keep")
    alias = tmp_path / "alias"
    alias.symlink_to(src, target_is_directory=True)
    existing = tmp_path / "other"
    existing.mkdir()
    for dest in (src, src / "child", tmp_path, alias / "child", existing):
        with pytest.raises(SystemExit):
            ns.main(["--src", str(src), "--out", str(dest)])
    assert marker.read_text() == "keep"
    assert existing.is_dir()
    assert not (src / "child").exists()


def test_nested_builder_publishes_only_verified_packages(tmp_path, monkeypatch):
    import buoycam_cv_folds

    monkeypatch.setattr(buoycam_cv_folds, "N_DRAWS", 50)
    monkeypatch.setattr(buoycam_cv_folds, "VAL_MIN_BUOYS", 2)
    src = tmp_path / "source"
    src.mkdir()
    strips = _strips()
    strips.to_csv(src / "labels.csv", index=False)
    folds = strips.groupby("station_group", as_index=False).agg(
        stations=("station_id", "nunique"),
        strips=("station_id", "size"),
        cv_fold=("cv_fold", "first"),
    )
    folds.assign(swell_share=0.5).to_csv(src / "folds.csv", index=False)
    sums = "".join(f"{sha256_file(src / name)}  {name}\n" for name in ("labels.csv", "folds.csv"))
    manifest = src / "SHA256SUMS"
    manifest.write_text(sums.replace(sha256_file(src / "folds.csv"), "0" * 64))
    dest = tmp_path / "nested"
    args = ["--src", str(src), "--out", str(dest)]
    with pytest.raises(ValueError, match="checksum mismatch"):
        ns.main(args)
    assert list(tmp_path.iterdir()) == [src]
    manifest.write_text(sums)
    assert ns.main(args) == 0
    assert (dest / "folds.csv").samefile(src / "folds.csv")
    assert not (src / "labels.csv").samefile(dest / "labels.csv")
    for line in (dest / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert sha256_file(dest / name) == digest
