"""The offline end-to-end example must keep working (it exercises every v0.1 stage)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

EXAMPLE = (
    Path(__file__).resolve().parents[2] / "examples" / "synchronized_sample" / "pipeline_demo.py"
)


def test_pipeline_demo_runs():
    spec = importlib.util.spec_from_file_location("pipeline_demo", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.run(verbose=False)
    assert out["n_manifest"] > 50
    assert out["leakage_ok"] == 1.0
    assert 0 <= out["clim_mae"] < 2.0
