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


def test_attitude_contrast_needs_identical_strips(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    score = importlib.import_module("score_sensing_papers")
    runs = {"yang": [], "yang-no-attitude": []}
    for arm, preds in (("yang", [1.0, 3.0]), ("yang-no-attitude", [2.0, 3.0])):
        for seed in range(3):
            p = tmp_path / f"{arm}{seed}"
            p.mkdir()
            (p / "settings.json").write_text(
                json.dumps({"seed": seed, "split": "station", "arm": arm, "labels_sha256": "x"})
            )
            pd.DataFrame(
                {"sample_id": ["a", "b"], "station_id": ["101", "201"], "target": [1.0, 3.0]}
            ).assign(pred_hs_m=preds).to_csv(p / "predictions.csv", index=False)
            runs[arm].append(p)
    result = score.compare_arms(runs["yang"], runs["yang-no-attitude"])
    assert result["rows"] == 2 and result["yang"] == 0.0
    assert abs(result["yang-no-attitude"] - 0.5**0.5) < 1e-12
    with pytest.raises(ValueError):
        score.compare_arms(runs["yang-no-attitude"], runs["yang"])


def test_backbone_control_and_view_positions(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    score = importlib.import_module("score_sensing_papers")
    nan = float("nan")
    runs = {"choi-dinov3": [], "choi": []}
    # two strips, both calm (class 0); view 1 unusable on the second strip
    for arm, p1 in (("choi-dinov3", [0.9, 0.8]), ("choi", [0.4, 0.8])):
        for seed in range(3):
            p = tmp_path / f"{arm}{seed}"
            p.mkdir()
            (p / "settings.json").write_text(
                json.dumps({"seed": seed, "split": "station", "arm": arm, "labels_sha256": "x"})
            )
            frame = pd.DataFrame(
                {"sample_id": ["a", "b"], "station_id": ["101", "201"], "target": [0, 0]}
            ).assign(prob_0=p1, prob_1=[1 - x for x in p1])
            for v in range(6):
                usable = [1.0, nan if v == 1 else 1.0]
                frame[f"view{v}_prob_0"] = [x * u for x, u in zip(p1, usable, strict=True)]
                frame[f"view{v}_prob_1"] = [(1 - x) * u for x, u in zip(p1, usable, strict=True)]
            frame.to_csv(p / "predictions.csv", index=False)
            runs[arm].append(p)
    result = score.compare_arms(runs["choi-dinov3"], runs["choi"], views=True)
    assert result["choi-dinov3"] == 1.0 and result["choi"] == 0.5 and result["delta"] == 0.5
    assert [v["rows"] for v in result["views"]] == [2, 1, 2, 2, 2, 2]
    with pytest.raises(ValueError):
        score.compare_arms(runs["choi"], runs["choi-dinov3"])


def test_out_of_fold_comparison_resamples_station_groups(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    score = importlib.import_module("score_sensing_papers")
    paper, ref = [], []
    for fold in range(5):
        rows = pd.DataFrame(
            {
                "sample_id": [f"s{fold}a", f"s{fold}b"],
                "station_id": [f"{fold}01", f"{fold}02"],
                "station_group": [f"g{fold}", f"g{fold}"],
            }
        )
        for seed in range(3):
            p, r = tmp_path / f"p{fold}{seed}", tmp_path / f"r{fold}{seed}"
            p.mkdir()
            r.mkdir()
            meta = {"seed": seed, "split": f"fold{fold}"}
            (p / "settings.json").write_text(
                json.dumps({**meta, "arm": "yang", "labels_sha256": "x"})
            )
            (r / "metrics.json").write_text(
                json.dumps(
                    {
                        "run": {
                            **meta,
                            "aux": "none",
                            "config": "ndbc_buoycam_heights_v2cv",
                            "settings": {"backbone": "timm:convnext_tiny.dinov3_lvd1689m"},
                        }
                    }
                )
            )
            rows.assign(target=[1.0, 2.0], pred_hs_m=[2.0, 2.0]).to_csv(
                p / "predictions.csv", index=False
            )
            rows.drop(columns="station_group").assign(
                partition="test", true_hs_m=[1.0, 2.0], pred_hs_m=[1.0, 2.0]
            ).to_csv(r / "predictions.csv", index=False)
            paper.append(p)
            ref.append(r)
    result = score.compare(paper, ref)
    assert result["rows"] == 10 and result["station_groups"] == 5
    assert result["project"] == 0.0 and abs(result["paper"] - 0.5**0.5) < 1e-12
    with pytest.raises(ValueError):
        score.compare(paper[:12], ref)
