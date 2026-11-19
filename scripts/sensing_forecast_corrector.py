"""Fit the registered no-camera arms on the five geographic outer folds."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wave_analysis.datasets.sea_state import nowcast_at


def prepare(labels, matched, era, depths):
    shared = ["sample_id", "station_id", "station_group", "cv_fold", "time_utc"]
    d = labels.merge(matched, on=shared, validate="1:1")
    if len(d) != len(labels):
        raise ValueError("GFS matching table must cover the complete package")
    d = d.merge(depths[["station_id", "depth_m"]], on="station_id", how="left", validate="m:1")
    times = pd.to_datetime(d.time_utc, utc=True)
    era_cols = [c for c in era if c.startswith("era5_")]
    by_station = era.set_index("time_utc").groupby("station_id")
    for col in era_cols:
        # Interpolate directional components to avoid averaging across north.
        direction = col.endswith("_deg")
        for suffix in ["_sin", "_cos"] if direction else [""]:
            key = col + suffix
            d[key] = np.nan
            for station, subset in by_station:
                rows = d.station_id.eq(str(station))
                if not rows.any():
                    continue
                v = subset[col].dropna()
                if direction:
                    angle = np.deg2rad(v)
                    v = np.sin(angle) if suffix == "_sin" else np.cos(angle)
                d.loc[rows, key] = nowcast_at(times[rows], v, max_gap="3h")
    cols = [c for c in d if c.startswith("era5_") and not c.endswith("_deg")]
    cols += ["depth_m", "latitude", "longitude", "hs_m", "systems", "extra_systems"]
    for i in range(1, 7):
        prefix = f"system{i}_"
        present = d[prefix + "present"].eq(True)
        deg = np.deg2rad(d[prefix + "direction_deg"])
        for suffix, values in (("sin", np.sin(deg)), ("cos", np.cos(deg))):
            d[prefix + suffix] = values
        for suffix in ["present", "wind", "hs_m", "tp_s", "sin", "cos"]:
            col = prefix + suffix
            d[col] = d[col].where(present, 0).astype(float)
            cols.append(col)
    return d, cols


def rmse(y, pred):
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def evaluate(d, cols):
    wind = ["wind_speed_mps", "wind_gust_mps"]
    valid = d.bulletin_present & np.isfinite(d[cols + wind]).all(axis=1)
    # This historical arm standardizes the logs and predicts a log height.
    old = np.log(np.maximum(d[["era5_hs_m", "era5_windsea_m", "era5_swell_m"]], 0.05))
    old[wind] = np.log(np.maximum(d[wind], 0.5))
    outputs = []
    for k in range(5):
        roles = d[f"split_fold{k}"]
        if d.groupby("station_group")[f"split_fold{k}"].nunique().max() != 1:
            raise ValueError("geographic group crosses roles")
        for target in ["Hs_label_m", "wind_wave_label_m"]:
            eligible = valid & np.isfinite(d[target])
            train, val, test = (eligible & roles.eq(role) for role in ["train", "val", "test"])
            if not all(r.any() for r in [train, val, test]):
                raise ValueError(f"empty role in fold {k}")
            y = d[target]
            out = d.loc[test, ["sample_id", "station_id", "station_group", "cv_fold"]].copy()
            out["target"], out["observed"] = target, y[test]
            base = make_pipeline(StandardScaler(), Ridge(alpha=1))
            base.fit(old[train], np.log(np.maximum(y[train], 0.05)))
            out["era5_wind"] = np.exp(base.predict(old[test]))
            for name, features in [("model", cols), ("model_wind", cols + wind)]:
                x = d[features]
                candidates = {
                    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
                    "boost": HistGradientBoostingRegressor(
                        max_iter=200,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        early_stopping=False,
                        random_state=0,
                    ),
                }
                validation = {}
                for arm, model in candidates.items():
                    model.fit(x[train], y[train])
                    validation[arm] = rmse(y[val], model.predict(x[val]))
                chosen = min(validation, key=validation.get)
                out[name] = candidates[chosen].predict(x[test])
                out[name + "_arm"] = chosen
                out[name + "_val_rmse"] = validation[chosen]
            outputs.append(out)
    result = pd.concat(outputs, ignore_index=True)
    if result.duplicated(["target", "sample_id"]).any():
        raise ValueError("duplicate held predictions")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ["labels", "matched", "era", "depths", "out"]:
        p.add_argument("--" + n, required=True, type=Path)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    labels, depths = (
        pd.read_csv(path, dtype={"station_id": str}, low_memory=False)
        for path in (a.labels, a.depths)
    )
    d, cols = prepare(labels, pd.read_parquet(a.matched), pd.read_parquet(a.era), depths)
    predictions = evaluate(d, cols)
    predictions.to_parquet(a.out, index=False)
    for target, g in predictions.groupby("target"):
        print(
            target,
            len(g),
            g.station_id.nunique(),
            g.station_group.nunique(),
            {c: round(rmse(g.observed, g[c]), 4) for c in ("era5_wind", "model", "model_wind")},
        )


if __name__ == "__main__":
    main()
