"""Test 2 (per-wave labels): score the shore video's fixed out-of-fold estimates against CDIP
zero-crossing statistics on the development dates.

Bar and arms: ``docs/experiments/sensing_per_wave_bars.md``, "Test 2 bar". No network is
trained here. The interim video run's out-of-fold H_s and T_p estimates (fixed before
any clip was scored, ``docs/experiments/webcoos_video.md``) are carried to the new
targets by one ratio per training fold, computed from training-fold labels only (never
from predictions); the period head is the registered pixel-spectrum ridge refitted on the
new target in the same folds.

    python scripts/sensing_shore_test2.py --labels runs/cdip-development-v1.csv \\
        --interim <interim run>/predictions_whole_clip.csv --out runs/<name>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wave_analysis.evaluation.uncertainty import paired_block_bootstrap_ci

KIM_EDGES = [0.75, 1.25, 1.75, 2.25, 2.75, 3.25]  # mean individual wave height (m)
TIERS = ("good", "incl_unevaluated")  # primary first
HEIGHTS = {"mean_height": "mean_height_m", "h_third": "h_third_m", "h_tenth": "h_tenth_m"}
HEIGHT_ARMS = {  # arm: the interim run's H_s column it carries
    "clip": "pred_hs_clip",  # V-JEPA 2 whole-clip read-out (energy mean of all windows)
    "window": "pred_hs_window",  # the same probe, midpoint 16-s window only
    "snapshot": "pred_hs_window_image",  # DINOv3 probe, one frame (the midpoint window's middle)
}
SPECTRUM = ["log_T_pix", "log_T_m01", "log_T_m02", "frac_25_16s", "frac_16_12s", "frac_12_9s",
            "frac_9_6s", "frac_6_3s", "log10_band_power", "rate_share"]  # fmt: skip
RIDGE_ALPHA = 1.0
BOOT = {"n_boot": 2000, "alpha": 0.05, "seed": 0}
MIN_DATES, MIN_CLASS_DATES = (
    8,
    3,
)  # coverage rule: >= 8 dates in range, two classes on >= 3 dates each


def assessable(true_class: np.ndarray, days: np.ndarray) -> bool:
    """The coverage rule of the class endpoints (docs/experiments/sensing_per_wave_bars.md)."""
    per_class = pd.Series(days).groupby(np.asarray(true_class)).nunique()
    return bool(len(set(days)) >= MIN_DATES and (per_class >= MIN_CLASS_DATES).sum() >= 2)


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def within_one(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p)) <= 1))


def kim_class(h: np.ndarray) -> np.ndarray:
    """Class 0-4 (clipped at the ends): for predictions; truth is restricted to the range first."""
    return np.clip(np.digitize(np.asarray(h, float), KIM_EDGES[1:-1]), 0, 4)


def in_kim_range(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, float)
    return (h >= KIM_EDGES[0]) & (h < KIM_EDGES[-1])


def ridge_oof(
    x: np.ndarray, y: np.ndarray, fold: np.ndarray, alpha: float = RIDGE_ALPHA
) -> np.ndarray:
    """The video run's ridge recipe: inputs standardised per training fold, missing = training
    mean, intercept not penalised (``scripts/train_video_probe.py``, ``ridge_oof``)."""
    pred = np.full(len(y), np.nan)
    for f in sorted(set(fold)):
        te, tr = fold == f, fold != f
        mu = np.nanmean(x[tr], axis=0)
        sd = np.nanstd(x[tr], axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        z = np.nan_to_num((x - mu) / sd, nan=0.0)
        ym = y[tr].mean()
        beta = np.linalg.solve(z[tr].T @ z[tr] + alpha * np.eye(z.shape[1]), z[tr].T @ (y[tr] - ym))
        pred[te] = z[te] @ beta + ym
    return pred


def per_fold(fold: np.ndarray, fn) -> np.ndarray:  # type: ignore[no-untyped-def]
    """``fn(train_mask, test_mask)`` returns the test fold's values; assembled out of fold."""
    out = np.full(len(fold), np.nan)
    for f in sorted(set(fold)):
        te, tr = fold == f, fold != f
        out[te] = fn(tr, te)
    return out


def arms(d: pd.DataFrame, tier: str) -> pd.DataFrame:
    """Out-of-fold estimates of every test 2 target, one column ``pred_<target>_<arm>``."""
    fold = d["fold"].to_numpy()
    hs = d["Hs_label_m"].to_numpy(float)
    out = d.copy()
    for t, col in HEIGHTS.items():
        y = d[f"{tier}_{col}"].to_numpy(float)
        ratio = per_fold(
            fold, lambda tr, te, y=y: np.full(te.sum(), np.exp(np.mean(np.log(y[tr] / hs[tr]))))
        )
        out[f"ratio_{t}"] = ratio
        for arm, src in HEIGHT_ARMS.items():
            out[f"pred_{t}_{arm}"] = d[src].to_numpy(float) * ratio
        out[f"pred_{t}_climatology"] = per_fold(
            fold, lambda tr, te, y=y: np.full(te.sum(), np.exp(np.mean(np.log(y[tr]))))
        )
    tz = d[f"{tier}_tz_s"].to_numpy(float)
    tp = d["Tp_label_s"].to_numpy(float)
    out["pred_tz_climatology"] = per_fold(
        fold, lambda tr, te: np.full(te.sum(), np.exp(np.mean(np.log(tz[tr]))))
    )
    out["pred_tz_spectrum"] = np.exp(ridge_oof(d[SPECTRUM].to_numpy(float), np.log(tz), fold))
    tz_ratio = per_fold(
        fold, lambda tr, te: np.full(te.sum(), np.exp(np.mean(np.log(tz[tr] / tp[tr]))))
    )
    out["pred_tz_clip"] = d["pred_tp_clip"].to_numpy(float) * tz_ratio
    # Kim's classes: truth only inside 0.75-3.25 m; the majority class of the training fold's
    # in-range clips (ties to the lower class)
    hm = d[f"{tier}_mean_height_m"].to_numpy(float)
    inside = in_kim_range(hm)
    out["kim_in_range"] = inside
    out["kim_true"] = np.where(inside, kim_class(hm), -1)

    def majority(tr: np.ndarray, te: np.ndarray) -> np.ndarray:
        c = np.bincount(kim_class(hm[tr & inside]), minlength=5)
        return np.full(te.sum(), int(np.argmax(c)) if c.sum() else -1)

    out["kim_majority"] = per_fold(fold, majority).astype(int)
    for arm in HEIGHT_ARMS:
        out[f"kim_{arm}"] = kim_class(out[f"pred_mean_height_{arm}"].to_numpy(float))
    return out


def diff(y: np.ndarray, a: np.ndarray, b: np.ndarray, metric, groups: np.ndarray) -> dict:  # type: ignore[no-untyped-def]
    point, lo, hi = paired_block_bootstrap_ci(y, a, b, metric, groups=groups, **BOOT)
    return {"diff": point, "lo": lo, "hi": hi}


def score(o: pd.DataFrame, tier: str) -> dict[str, Any]:
    g = o["local_day"].to_numpy()
    res: dict[str, Any] = {
        "tier": tier,
        "n_clips": len(o),
        "n_dates": int(o["local_day"].nunique()),
        "clips_by_camera": o["camera"].value_counts().sort_index().to_dict(),
        "regression": {},
        "diffs": {},
    }
    for t, col in {**HEIGHTS, "tz": "tz_s"}.items():
        y = o[f"{tier}_{col}"].to_numpy(float)
        names = [*HEIGHT_ARMS, "climatology"] if t != "tz" else ["spectrum", "clip", "climatology"]
        res["regression"][t] = {a: rmse(y, o[f"pred_{t}_{a}"].to_numpy(float)) for a in names}
        pairs = (
            [("clip", "snapshot"), ("clip", "window"), ("clip", "climatology"), ("snapshot", "climatology")]
            if t != "tz" else [("spectrum", "climatology"), ("clip", "climatology"), ("spectrum", "clip")]
        )  # fmt: skip
        for a, b in pairs:
            res["diffs"][f"{t}: {a} - {b} (RMSE)"] = diff(
                y, o[f"pred_{t}_{a}"].to_numpy(float), o[f"pred_{t}_{b}"].to_numpy(float), rmse, g
            )
    k = o[o["kim_in_range"]]
    yk, gk = k["kim_true"].to_numpy(), k["local_day"].to_numpy()
    cls = {f"{KIM_EDGES[c]:.2f}-{KIM_EDGES[c + 1]:.2f}": int((yk == c).sum()) for c in range(5)}
    kim: dict[str, Any] = {
        "n_clips": len(k),
        "n_dates": int(k["local_day"].nunique()),
        "classes_present": len(set(yk)),
        "clips_by_class": cls,
        "dates_by_class": {
            n: int(k.loc[yk == c, "local_day"].nunique()) for c, n in enumerate(cls)
        },
        "accuracy": {},
        "within_one": {},
    }
    kim["assessable"] = assessable(yk, gk)
    for a in ["majority", *HEIGHT_ARMS]:
        p = k[f"kim_{a}"].to_numpy()
        kim["accuracy"][a] = accuracy(yk, p)
        kim["within_one"][a] = within_one(yk, p)
    for a, b in [
        ("clip", "majority"),
        ("clip", "snapshot"),
        ("snapshot", "majority"),
        ("clip", "window"),
    ]:
        kim[f"{a} - {b}"] = diff(
            yk, k[f"kim_{a}"].to_numpy(), k[f"kim_{b}"].to_numpy(), accuracy, gk
        )
    kim["by_camera"] = {
        cam: {
            "n": len(c),
            **{a: accuracy(c["kim_true"], c[f"kim_{a}"]) for a in ["majority", "clip", "snapshot"]},
        }
        for cam, c in k.groupby("camera")
    }
    res["kim"] = kim
    res["bars"] = {
        "T2-A (class accuracy, clip - majority)": _met(
            kim["clip - majority"], higher=True, ok=kim["assessable"]
        ),
        "T2-B (class accuracy, clip - snapshot)": _met(
            kim["clip - snapshot"], higher=True, ok=kim["assessable"]
        ),
        "T2-C (T_z RMSE, spectrum - climatology)": _met(
            res["diffs"]["tz: spectrum - climatology (RMSE)"], higher=False
        ),
    }
    res["wave_count"] = {
        "median": float(o[f"{tier}_wave_count"].median()),
        "min": int(o[f"{tier}_wave_count"].min()),
        "median_count_x_tz_s": float((o[f"{tier}_wave_count"] * o[f"{tier}_tz_s"]).median()),
    }
    return res


def _met(d: dict, *, higher: bool, ok: bool = True) -> str:
    if not ok:
        return "unassessable (coverage)"
    return "met" if (d["lo"] > 0 if higher else d["hi"] < 0) else "not met"


def report(all_res: dict) -> str:
    lines = ["# Test 2 (per-wave labels), development dates", ""]
    for tier, r in all_res["tiers"].items():
        k = r["kim"]
        lines += [f"## Tier `{tier}`: {r['n_clips']} clips, {r['n_dates']} dates", ""]
        lines += ["| Bar | Difference [95 %] | Verdict |", "|---|---|---|"]
        src = {
            "T2-A (class accuracy, clip - majority)": k["clip - majority"],
            "T2-B (class accuracy, clip - snapshot)": k["clip - snapshot"],
            "T2-C (T_z RMSE, spectrum - climatology)": r["diffs"][
                "tz: spectrum - climatology (RMSE)"
            ],
        }
        for bar, verdict in r["bars"].items():
            d = src[bar]
            lines.append(
                f"| {bar} | {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] | {verdict} |"
            )
        acc = ", ".join(f"{a} {v:.1%}" for a, v in k["accuracy"].items())
        lines += ["", f"Kim range: {k['n_clips']} clips on {k['n_dates']} dates, classes {k['clips_by_class']}.",
                  f"Accuracy: {acc}.", ""]  # fmt: skip
        lines += ["| Target | RMSE by arm |", "|---|---|"]
        for t, v in r["regression"].items():
            lines.append(f"| {t} | " + ", ".join(f"{a} {x:.3f}" for a, x in v.items()) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--interim", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    labels = pd.read_csv(a.labels, dtype={"station": str, "cdip_station": str})
    it = pd.read_csv(a.interim)
    lab_cols = ["clip_id"] + [c for c in labels.columns if c.startswith(TIERS)]
    d0 = it.merge(labels[lab_cols], on="clip_id", how="left", validate="1:1")
    all_res: dict[str, Any] = {
        "labels_sha256": hashlib.sha256(a.labels.read_bytes()).hexdigest(),
        "interim_sha256": hashlib.sha256(a.interim.read_bytes()).hexdigest(),
        "interim_clips": len(it),
        "tiers": {},
    }
    preds = []
    for tier in TIERS:
        d = d0[d0[f"{tier}_mean_height_m"].notna()].reset_index(drop=True)
        o = arms(d, tier)
        all_res["tiers"][tier] = score(o, tier)
        preds.append(o.assign(tier=tier))
    pd.concat(preds).to_csv(a.out / "predictions.csv", index=False)
    (a.out / "scores.json").write_text(json.dumps(all_res, indent=2, default=float) + "\n")
    (a.out / "report.md").write_text(report(all_res))
    print(report(all_res))


if __name__ == "__main__":
    main()
