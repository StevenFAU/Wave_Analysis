"""Historical wind-wave weighting tests fixed in docs/experiments/wind_wave_reliability.md."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import sensing_nested_fusion as nested
from wave_analysis.config import repo_root
from wave_analysis.evaluation.regression import rmse
from wave_analysis.evaluation.uncertainty import paired_block_bootstrap_ci
from wave_analysis.ingest.checksum import sha256_file

INPUTS = ["forecast", "wind", "disagreement"]
TRANSFER_ARMS = [
    "forecast",
    "camera",
    "fusion",
    "varweight",
    "constant",
    "two_inputs",
    "reliability",
]
LOCAL_ARMS = [*TRANSFER_ARMS, "local_constant", "local_reliability"]
WIND = "true_wind_wave_m"


def inputs(root):
    runs = root / "runs"
    labels = pd.read_csv(
        root / "data/processed/packages/ndbc_buoycam_heights_v2cvn/labels.csv",
        dtype={"station_id": str},
        low_memory=False,
    )
    oof = pd.read_csv(
        runs / "buoycam-cv-stage1-20261119/oof_dinov3-gentle.csv", dtype={"station_id": str}
    )
    inner_cam = nested.inner_camera(sorted(runs.glob("buoy-pair*-seed*-nested-gentle")), oof)
    inner = nested.complete(
        nested.with_forecast(
            nested.inner_features(inner_cam, labels),
            pd.read_parquet(runs / "forecast-baseline-inner-v1.parquet"),
            "model_wind",
            by_pair=True,
        )
    ).rename(columns={"cmp_ww": "forecast", "cam_ww": "camera", WIND: "observed"})
    path = runs / "forecast-fusion-nested-v1/rows.parquet"
    outer = pd.read_parquet(
        path,
        filters=[
            ("baseline", "==", "model_wind"),
            ("arrangement", "==", "nested"),
            ("target", "==", WIND),
        ],
    )
    outer = outer.merge(
        labels[["sample_id", "station_id", "wind_speed_mps", "time_utc", "wind_wave_label_m"]],
        on=["sample_id", "station_id"],
        how="left",
        validate="1:1",
    ).rename(columns={"wind_speed_mps": "wind"})
    for frame in (inner, outer):
        frame["cv_fold"] = frame.cv_fold.astype(str)
        frame["disagreement"] = frame.camera - frame.forecast
        if not np.isfinite(frame[[*INPUTS, "observed"]].to_numpy()).all():
            raise ValueError("missing forecast, camera, wind or observed height")
    outer["time_utc"] = pd.to_datetime(outer.time_utc, utc=True, errors="raise")
    if (len(outer), outer.station_group.nunique(), outer.station_id.nunique()) != (34723, 41, 71):
        raise ValueError("unexpected historical scoring population")
    if outer.time_utc.isna().any() or not np.allclose(
        outer.observed, outer.wind_wave_label_m, atol=1e-5
    ):
        raise ValueError("invalid label join")
    return inner, outer, sha256_file(path)


def bin_ids(df, cuts):
    ids = np.zeros(len(df), dtype=int)
    for name, edges in cuts.items():
        ids = 3 * ids + np.searchsorted(edges, df[name], side="right")
    return ids


def fit_weights(cal, features):
    d = cal.disagreement.to_numpy()
    residual = (cal.observed - cal.forecast).to_numpy()
    denom = d @ d
    constant = float(np.clip(d @ residual / denom, 0, 1)) if denom else 0.0
    mean_d2 = float(denom / len(cal))
    cuts = {c: np.quantile(cal[c], [1 / 3, 2 / 3]).tolist() for c in features}
    ids = bin_ids(cal, cuts)
    cells = 3 ** len(features)
    counts = np.bincount(ids, minlength=cells)
    cross = np.bincount(ids, weights=d * residual, minlength=cells)
    square = np.bincount(ids, weights=d * d, minlength=cells)
    penalty = 100 * mean_d2
    weights = np.full(cells, constant)
    np.divide(cross + penalty * constant, square + penalty, out=weights, where=square + penalty > 0)
    return {
        "cuts": cuts,
        "weights": weights.clip(0, 1).tolist(),
        "counts": counts.tolist(),
        "constant": constant,
        "mean_d2": mean_d2,
        "rows": len(cal),
    }


def transfer(inner, outer):
    frames, fits = [], {}
    for k, holdout in outer.groupby("cv_fold"):
        test = holdout.copy()
        cal = nested.calibration(inner, k)
        expected = outer.loc[outer.cv_fold != k, "sample_id"]
        if (
            cal.sample_id.duplicated().any()
            or set(cal.sample_id) != set(expected)
            or set(cal.station_group) & set(test.station_group)
        ):
            raise ValueError(f"invalid calibration population for fold {k}")
        fits[k] = {}
        for arm, cols in (("two_inputs", INPUTS[:2]), ("reliability", INPUTS)):
            fit = fit_weights(cal, cols)
            fits[k][arm] = fit
            weight = np.asarray(fit["weights"])[bin_ids(test, fit["cuts"])]
            test["weight_" + arm] = weight
            test[arm] = test.forecast + weight * test.disagreement
        test["weight_constant"] = fit["constant"]
        test["constant"] = test.forecast + test.weight_constant * test.disagreement
        test["inner_mean_d2"] = fit["mean_d2"]
        frames.append(test)
    return pd.concat(frames, ignore_index=True), fits


def local_calibration(rows):
    frames, coverage = [], []
    for station, site in rows.groupby("station_id"):
        start = site.time_utc.min()
        early = site[site.time_utc < start + pd.Timedelta(days=7)]
        later = site[site.time_utc >= start + pd.Timedelta(days=9)].copy()
        record = {
            "station_id": station,
            "start_utc": start,
            "early": len(early),
            "later": len(later),
            "embargo": len(site) - len(early) - len(later),
            "eligible": len(early) >= 24 and len(later) > 0,
        }
        coverage.append(record)
        if not record["eligible"]:
            continue
        if (
            not (site.cv_fold == site.cv_fold.iloc[0]).all()
            or not (site.inner_mean_d2 == site.inner_mean_d2.iloc[0]).all()
        ):
            raise ValueError(f"ambiguous fold for station {station}")
        d = early.disagreement.to_numpy()
        denom = d @ d + 50 * early.inner_mean_d2.iloc[0]
        for arm in ("constant", "reliability"):
            offset = float(d @ (early.observed - early[arm]) / denom) if denom else 0.0
            record[arm + "_offset"] = offset
            weight = (later["weight_" + arm] + offset).clip(0, 1)
            later["local_" + arm] = later.forecast + weight * later.disagreement
        frames.append(later)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(coverage)


def score(population, df, arms, candidate, comparators):
    scores, contrasts = [], []
    for scope, parts in (
        ("overall", [("all", df)]),
        ("fold", df.groupby("cv_fold")),
        ("group", df.groupby("station_group")),
    ):
        for name, part in parts:
            scores.append(
                {
                    "population": population,
                    "scope": scope,
                    "unit": name,
                    "strips": len(part),
                    "buoys": part.station_id.nunique(),
                    "groups": part.station_group.nunique(),
                    **{arm: rmse(part.observed, part[arm]) for arm in arms},
                }
            )
    for scope, part in (("all", df), ("excluding_41040", df[df.station_group != "g41040"])):
        by_group = (
            part[arms].sub(part.observed, axis=0).pow(2).groupby(part.station_group).mean().pow(0.5)
        )
        for other in comparators:
            for weighting in ("strip", "equal_group"):
                if weighting == "strip":
                    y, a, b, groups = (
                        part.observed,
                        part[candidate],
                        part[other],
                        part.station_group,
                    )
                else:
                    # RMSE of group RMSEs against zero is sqrt(mean(group MSE)).
                    y, a, b, groups = (
                        np.zeros(len(by_group)),
                        by_group[candidate],
                        by_group[other],
                        by_group.index,
                    )
                diff, lo, hi = paired_block_bootstrap_ci(
                    y, a, b, rmse, groups=groups, n_boot=2000, seed=0
                )
                contrasts.append(
                    {
                        "population": population,
                        "scope": scope,
                        "weighting": weighting,
                        "candidate": candidate,
                        "comparator": other,
                        "delta_m": diff,
                        "lo_m": lo,
                        "hi_m": hi,
                        "improved_groups": int((by_group[candidate] < by_group[other]).sum()),
                        "groups": len(by_group),
                    }
                )
    return pd.DataFrame(scores), pd.DataFrame(contrasts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = repo_root()
    parser.add_argument("--out", type=Path, default=root / "runs/wind-wave-reliability-v1")
    out = parser.parse_args().out
    if out.exists():
        raise FileExistsError(out)
    inner, outer, digest = inputs(root)
    rows, fits = transfer(inner, outer)
    later, coverage = local_calibration(rows)
    fit_scores, fit_contrasts = score(
        "unseen",
        rows,
        TRANSFER_ARMS,
        "reliability",
        ["forecast", "fusion", "varweight", "constant", "two_inputs"],
    )
    local_scores, local_contrasts = score(
        "later",
        later,
        LOCAL_ARMS,
        "local_reliability",
        ["reliability", "local_constant", "forecast", "fusion", "varweight"],
    )
    scores = pd.concat([fit_scores, local_scores], ignore_index=True)
    contrasts = pd.concat([fit_contrasts, local_contrasts], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as directory:
        stage = Path(directory) / "result"
        stage.mkdir()
        scores.to_csv(stage / "scores.csv", index=False)
        contrasts.to_csv(stage / "contrasts.csv", index=False)
        coverage.to_csv(stage / "station-calibration.csv", index=False)
        later_preds = later[["sample_id", "local_constant", "local_reliability"]]
        rows.merge(later_preds, on="sample_id", how="left", validate="1:1").to_parquet(
            stage / "rows.parquet", index=False
        )
        (stage / "fits.json").write_text(
            json.dumps({"historical_rows_sha256": digest, "folds": fits}, indent=2) + "\n"
        )
        stage.rename(out)
    print(scores[scores.scope == "overall"].round(4).to_string(index=False))
    print(contrasts.query("scope=='all' and weighting=='strip'").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
