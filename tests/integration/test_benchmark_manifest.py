"""Run the benchmark CLI on retained and filtered station holdouts."""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("test_groups", "bad_rows", "missing"),
    [([], [], "test"), (["C"], ["C"], "test"), (["C"], ["B"], "val"), (["C"], [], None)],
)
def test_required_benchmark_partitions(tmp_path, test_groups, bad_rows, missing):
    cfg = yaml.safe_load((ROOT / "configs/experiments/baseline_hs.yaml").read_text())
    cfg["split"].update(test_groups=test_groups, val_groups=["B"])
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    samples = tmp_path / "strips.parquet"
    rows = pd.DataFrame({"sample_id": [1, 2, 3], "station_id": ["A", "B", "C"]})
    rows.assign(
        Hs_m=1.0,
        provider_qc_hs=1,
        time_offset_s=0.0,
        time_utc="2026-10-01T00:00Z",
        sync_qc_flag=rows.station_id.isin(bad_rows).map({True: 9, False: 1}),
    ).to_parquet(samples)
    result = subprocess.run(
        [sys.executable, ROOT / "scripts/build_ml_manifest.py", config, samples],
        env={**os.environ, "WAVE_ANALYSIS_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    if missing:
        assert result.returncode != 0
        assert f"no eligible rows for {missing}" in result.stderr
        assert not (tmp_path / "data").exists()
    else:
        assert result.returncode == 0, result.stderr
        manifest = next((tmp_path / "data").rglob("*.parquet"))
        assert pd.read_parquet(manifest)["split"].tolist() == ["train", "val", "test"]
