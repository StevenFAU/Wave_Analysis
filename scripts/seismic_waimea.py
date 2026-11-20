"""Evaluate registered Waimea seismic-only and monthly climatology arms."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def evaluate(manifest, hourly):
    hours = pd.concat((pd.read_csv(path) for path in hourly), ignore_index=True)
    hours["hour"] = pd.to_datetime(hours.utc, utc=True)
    d = manifest.assign(
        hour=pd.to_datetime(manifest.time_utc, utc=True).dt.floor("h") - pd.Timedelta(hours=1)
    ).merge(hours, on="hour", how="left", validate="m:1")
    print("manifest rows by seismic status:", d.status.fillna("missing").value_counts().to_dict())
    cols = [
        "primary_velocity_variance",
        "secondary_velocity_variance",
        "primary_peak_hz",
        "secondary_peak_hz",
        "wave_period_proxy_s",
    ]
    targets = ["Hs_label_m", "Tp_label_s"]
    use = (
        d.status.eq("ok")
        & np.isfinite(d[cols + targets]).all(axis=1)
        & (d[targets] > 0).all(axis=1)
    )
    d = d.loc[use].copy()
    d[cols[:2]] = np.log(d[cols[:2]])
    predictions = []
    for split in ["day", "week", "year"]:
        train, test = (d[d["split_" + split].eq(role)] for role in ["train", "test"])
        if set(train.hst_day) & set(test.hst_day):
            raise ValueError("shared dates across roles")
        x = train[cols].to_numpy()
        mu, sd = x.mean(axis=0), x.std(axis=0)
        sd[sd == 0] = 1
        x = (x - mu) / sd
        for target in targets:
            y = train[target].to_numpy()
            clim = test.hst_month.map(train.groupby("hst_month")[target].mean()).fillna(y.mean())
            coef = np.linalg.solve(x.T @ x + 10 * np.eye(len(cols)), x.T @ (y - y.mean()))
            pred = y.mean() + ((test[cols].to_numpy() - mu) / sd) @ coef
            predictions.append(
                test[["sample_id", "hst_day", "hour", "peak_records_disagree"]].assign(
                    split=split,
                    target=target,
                    observed=test[target],
                    climatology=clim,
                    seismic=pred,
                )
            )
    return pd.concat(predictions, ignore_index=True)


def summarize(predictions):
    rows = []
    for (split, target), d in predictions.groupby(["split", "target"]):
        for name, g in [("all", d), ("agree", d[d.peak_records_disagree.eq(False)])]:
            errors = g[["climatology", "seismic"]].sub(g.observed, axis=0).pow(2)
            by_day = errors.groupby(g.hst_day)
            sums, n = by_day.sum().to_numpy(), by_day.size().to_numpy()
            ix = np.random.default_rng(0).integers(len(n), size=(2000, len(n)))
            vals = np.sqrt(sums[ix].sum(axis=1) / n[ix].sum(axis=1)[:, None])
            lo, hi = np.quantile(vals[:, 1] - vals[:, 0], [0.025, 0.975])
            c, s = np.sqrt(errors.mean())
            rows.append(
                {
                    "split": split,
                    "target": target,
                    "cohort": name,
                    "rows": len(g),
                    "days": len(n),
                    "climatology_rmse": c,
                    "seismic_rmse": s,
                    "lower": lo,
                    "upper": hi,
                }
            )
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--hourly", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    files = sorted(a.hourly.glob("*.csv"))
    if not files or {f.stem for f in files} != {f.stem for f in a.raw.glob("*.mseed")}:
        raise ValueError("incomplete set of processed days")
    d = evaluate(pd.read_parquet(a.manifest), files)
    res = summarize(d)
    a.out.mkdir(exist_ok=False, parents=True)
    d.to_parquet(a.out / "predictions.parquet", index=False)
    res.to_csv(a.out / "scores.csv", index=False)
    print(res[res.split.eq("year")].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
