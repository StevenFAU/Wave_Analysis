"""A held-station triplet must use matching strips and all three seeds."""

import importlib
import json
from pathlib import Path

import pandas as pd
import pytest


def test_three_seed_comparison(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    score = importlib.import_module("score_sensing_papers")
    pp = []
    rr = []
    for seed in range(3):
        p, r = tmp_path / f"p{seed}", tmp_path / f"r{seed}"
        p.mkdir()
        r.mkdir()
        meta = {"seed": seed, "split": "station"}
        (p / "settings.json").write_text(
            json.dumps({**meta, "arm": "choi", "labels_sha256": "test"})
        )
        (r / "metrics.json").write_text(
            json.dumps(
                {
                    "run": {
                        **meta,
                        "aux": "none",
                        "config": "ndbc_buoycam_heights_v2",
                        "settings": {"backbone": "timm:convnext_tiny.dinov3_lvd1689m"},
                    }
                }
            )
        )
        d = pd.DataFrame({"sample_id": ["a", "b"], "station_id": ["101", "201"]})
        d.assign(target=[0, 1], prob_0=[0.8, 0.7], prob_1=[0.2, 0.3]).to_csv(
            p / "predictions.csv", index=False
        )
        d.assign(partition="test", true_hs_m=[1, 4], pred_hs_m=[1, 3]).to_csv(
            r / "predictions.csv", index=False
        )
        pp.append(p)
        rr.append(r)
    result = score.compare(pp, rr)
    assert result["rows"] == 2 and result["delta"] == 0.5
    with pytest.raises(ValueError):
        score.compare([pp[0]] * 3, rr)
