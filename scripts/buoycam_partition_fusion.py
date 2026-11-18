#!/usr/bin/env python
"""Stage 2: partition-aware fusion. The camera's wind sea plus ERA5's swell, against ERA5 + buoy wind.

    python scripts/buoycam_partition_fusion.py --oof runs/<stage-1 oof>/oof_<arm>.csv --out runs/<name>

Choices fixed on 2026-11-18, before any fusion was fitted or scored (``docs/experiments/
ndbc_buoycam_sea_state.md``, "Stage 2: partition-aware fusion"). Rank 3 of the
architecture review. NDBC's partition adds as energies on every camera buoy record
(WVHT² = SwH² + WWH² within the 0.1 m rounding, ``scripts/buoycam_partition_check.py``),
so a total height can be built from a swell part and a wind-sea part.

* **Inputs per strip:** the stage-1 camera's out-of-fold heights (``pred_*_m``) and the
  spread of its log wind-wave height (``sigma_wind_wave_log``); ERA5's total, wind-sea
  and swell heights at the buoy, as interpolated (``era5_*_raw_m``, no correction); the
  buoy's wind speed and gust; and five **gate features** that say how well the camera
  can see: the camera's spread, the sun's elevation, the number of usable views, and
  the mean grey and mean contrast (standard deviation) of the usable views.
* **Out of fold, nested:** every combiner predicting fold k is fitted on the other four
  folds' strips (whose camera heights are themselves out of fold), so no strip is
  predicted by anything fitted on its buoy or its group.
* **The bar's comparator, ERA5 + wind:** Test 3's ridge (standardised features, alpha
  1.0) of log H_s, and separately of log wind-wave height, on the logs of ERA5's three
  heights and the buoy's wind speed and gust. Its two outputs are T (total) and W
  (wind sea).
* **Main candidate, partition swap with a gate:** the comparator's own total with its
  wind-sea energy swapped for the camera's,
  H_s² = T² + k g (b C_ww² - W²), wind-wave height² = g b C_ww² + (1 - g) W²,
  where C_ww is the camera's wind-wave height, g = sigmoid(w0 + w · gate features)
  falls back to the comparator where the camera cannot see (dusk, haze, glare), and
  k (0 to 2) is how much of the wind-sea correction reaches the total: k = 1 is the full
  swap, (T² - W²) + C², if T carries all of W's error. g = 0 or k = 0 gives the
  comparator's total, so the bar asks only whether the camera's wind sea adds
  information. b > 0, k and the gate are fitted by least squares of log H_s and log
  wind-wave height together (L-BFGS-B; 1e-3 times the squared gate weights added), with
  T and W from the comparator fitted on the same strips.
* **Beside them, not part of the bar:** the task's partition rule on raw ERA5 with nothing
  fitted (H_s² = swh² - shww² + C_ww², wind sea = C_ww); the full swap (k = 1) with the
  gate, and with the gate held open (g = 1); a partition fusion on raw ERA5 with fitted scales and the same gate,
  H_s² = a (swh² - shww²)₊ + g b C_ww² + (1 - g) c shww²; Test 3's full ridge with the
  camera; gradient boosting on every input above (:class:`HistBoost`, least squares,
  300 trees of depth 3, learning rate 0.05, 32 quantile bins per input, at least 20
  strips per leaf; written here because scikit-learn is not a project dependency);
  ERA5 with a line on each fold's training strips; the camera alone. TabPFN is not
  run: its weights need a non-commercial licence to be accepted.
* **Bar S2:** the gated swap must lower the out-of-fold RMSE of H_s **and** of the
  wind-wave height against ERA5 + wind, each 95 % interval of the difference below
  zero (paired bootstrap over whole buoys). Beside it: the same over station groups.
* Scored per target on the strips where every input and that target exist.

Writes ``<out>/fusion.json``, ``report.md``, ``fusion_oof.csv`` (every strip's fused
heights) and ``models.pkl`` (each fold's fitted combiners, for the strips archived
after the package: ``--apply``).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from wave_analysis.config import data_dir
from wave_analysis.evaluation.regression import rmse
from wave_analysis.evaluation.uncertainty import paired_block_bootstrap_ci

FLOOR = 0.05
U_MIN = 0.5
ALPHA = 1.0
GATE_L2 = 1e-3
TARGETS = {"hs": "true_hs_m", "wind_wave": "true_wind_wave_m"}
GATE = ["cam_sigma_ww", "sun_elevation_deg", "usable_views", "view_grey", "view_contrast"]
BAR = "partition swap on ERA5 + wind, gated"
COMPARATOR = "ERA5 + wind (the bar)"


def lg(x: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(np.asarray(x, dtype=float), FLOOR))


def features(d: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """The inputs of every combiner, one row per strip of ``d`` (an out-of-fold table)."""
    lab = labels.set_index("sample_id").loc[d["sample_id"]].reset_index()
    mean_cols = [f"view_mean_{k}" for k in range(6)]
    std_cols = [f"view_std_{k}" for k in range(6)]
    vm, vs = lab[mean_cols].to_numpy(float), lab[std_cols].to_numpy(float)
    usable = (vm >= 20) & (vs >= 4)
    with np.errstate(invalid="ignore"):
        grey = np.where(usable, vm, np.nan)
        contrast = np.where(usable, vs, np.nan)
    f = pd.DataFrame({
        "sample_id": d["sample_id"], "station_id": d["station_id"], "station_group": d["station_group"],
        "cv_fold": d["cv_fold"].astype(str), "time_utc": d["time_utc"],
        "true_hs_m": d["true_hs_m"], "true_wind_wave_m": d["true_wind_wave_m"],
        "era5_hs": d["era5_hs_raw_m"], "era5_ww": d["era5_wind_wave_raw_m"], "era5_sw": d["era5_swell_raw_m"],
        "era5_line_hs": d["era5_hs_m"], "era5_line_ww": d["era5_wind_wave_m"],
        "cam_hs": d["pred_hs_m"], "cam_ww": d["pred_wind_wave_m"], "cam_sw": d["pred_swell_m"],
        "cam_sigma_hs": d["sigma_hs_log"], "cam_sigma_ww": d["sigma_wind_wave_log"],
        "wind": lab["wind_speed_mps"].to_numpy(float), "gust": lab["wind_gust_mps"].to_numpy(float),
        "sun_elevation_deg": lab["sun_elevation_deg"].to_numpy(float),
        "usable_views": usable.sum(axis=1).astype(float),
        "view_grey": np.nanmean(grey, axis=1), "view_contrast": np.nanmean(contrast, axis=1),
        "illumination": lab["illumination"].to_numpy(),
    })  # fmt: skip
    return f


def x_era5_wind(f: pd.DataFrame) -> np.ndarray:
    return np.column_stack([lg(f["era5_hs"]), lg(f["era5_ww"]), lg(f["era5_sw"]),
                            np.log(np.maximum(f["wind"], U_MIN)), np.log(np.maximum(f["gust"], U_MIN))])  # fmt: skip


def x_camera(f: pd.DataFrame) -> np.ndarray:
    return np.column_stack([lg(f["cam_hs"]), lg(f["cam_ww"]), lg(f["cam_sw"])])


def x_all(f: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [x_era5_wind(f), x_camera(f), f["cam_sigma_hs"], f[GATE].to_numpy(float)]
    )


# ----------------------------------------------------------------------------- combiners


class Ridge:
    """Ridge on standardised features (Test 3's)."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> Ridge:
        self.mu, self.sd = x.mean(0), x.std(0)
        self.sd[self.sd == 0] = 1.0
        z = (x - self.mu) / self.sd
        self.ym = y.mean()
        self.coef = np.linalg.solve(z.T @ z + ALPHA * np.eye(z.shape[1]), z.T @ (y - self.ym))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(((x - self.mu) / self.sd) @ self.coef + self.ym)


class PartitionFusion:
    """Wind-sea energy swapped for the camera's behind a gate (see the module text).

    ``base="comparator"``: on the ERA5 + wind ridge's total T and wind sea W (columns
    ``cmp_hs``, ``cmp_ww``), H_s² = T² + k g (b C² - W²), with k fitted (``transfer``) or 1.
    ``base="raw"``: on raw ERA5, H_s² = a (swh² - shww²)₊ + g b C² + (1 - g) c shww².
    ``gate=False`` holds g = 1. The wind sea² is g b C² + (1 - g) W² (raw: c shww²).
    """

    def __init__(self, gate: bool = True, base: str = "comparator", transfer: bool = False) -> None:
        self.gate, self.base, self.transfer = gate, base, transfer and base == "comparator"

    @property
    def n_scale(self) -> int:
        return (2 if self.transfer else 1) if self.base == "comparator" else 3

    def _parts(self, f: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.base == "comparator":
            t, w = f["cmp_hs"].to_numpy(float), f["cmp_ww"].to_numpy(float)
        else:
            t, w = f["era5_hs"].to_numpy(float), f["era5_ww"].to_numpy(float)
        return t**2, f["cam_ww"].to_numpy(float) ** 2, w**2, f[GATE].to_numpy(float)

    def _scales(self, theta: np.ndarray) -> dict[str, Any]:
        n = self.n_scale
        if self.base == "comparator":
            k = 2 * float(expit(theta[1])) if self.transfer else 1.0
            return {"a": 1.0, "b": float(np.exp(theta[0])), "c": 1.0, "k": k, "gate": theta[n:]}
        a, b, c = np.exp(theta[:3])
        return {"a": float(a), "b": float(b), "c": float(c), "k": 1.0, "gate": theta[n:]}

    def _predict(self, theta: np.ndarray, parts: tuple) -> tuple[np.ndarray, np.ndarray]:
        t2, cam, era, g = parts
        sc = self._scales(theta)
        gt = sc["gate"]
        gate = expit(gt[0] + ((g - self.gmu) / self.gsd) @ gt[1:]) if self.gate else 1.0
        ww2 = gate * sc["b"] * cam + (1 - gate) * sc["c"] * era
        if self.base == "comparator":
            hs2 = t2 + sc["k"] * gate * (sc["b"] * cam - era)
        else:
            hs2 = sc["a"] * np.maximum(t2 - era, 0.0) + ww2
        return np.sqrt(np.maximum(hs2, FLOOR**2)), np.sqrt(ww2)

    def fit(self, f: pd.DataFrame) -> PartitionFusion:
        parts = self._parts(f)
        self.gmu, self.gsd = parts[3].mean(0), parts[3].std(0)
        self.gsd[self.gsd == 0] = 1.0
        yh, yw = lg(f["true_hs_m"]), lg(f["true_wind_wave_m"])
        n = self.n_scale

        def loss(theta: np.ndarray) -> float:
            hs, ww = self._predict(theta, parts)
            w = theta[n + 1 :]
            pen = GATE_L2 * float(w @ w) if self.gate else 0.0
            return float(np.mean((lg(hs) - yh) ** 2) + np.mean((lg(ww) - yw) ** 2)) + pen

        theta0 = np.zeros(n + (1 + len(GATE) if self.gate else 0))
        if self.gate:
            theta0[n] = 2.0  # start with the camera mostly trusted (k starts at 1)
        res = minimize(loss, theta0, method="L-BFGS-B")
        self.theta, self.converged = res.x, bool(res.success)
        return self

    def predict(self, f: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self._predict(self.theta, self._parts(f))

    def gate_values(self, f: pd.DataFrame) -> np.ndarray:
        gt = self._scales(self.theta)["gate"]
        g = f[GATE].to_numpy(float)
        return np.asarray(expit(gt[0] + ((g - self.gmu) / self.gsd) @ gt[1:]))

    def describe(self) -> dict[str, Any]:
        sc = self._scales(self.theta)
        out: dict[str, Any] = {x: round(sc[x], 3) for x in ("a", "b", "c", "k")}
        out["converged"] = self.converged
        if self.gate:
            out["gate_bias"] = round(float(sc["gate"][0]), 3)
            out["gate_weights"] = dict(zip(GATE, np.round(sc["gate"][1:], 3).tolist(), strict=True))
        return out


class HistBoost:
    """Least-squares gradient boosting with histogram splits: deterministic, numpy only."""

    def __init__(self, n_trees: int = 300, depth: int = 3, lr: float = 0.05, bins: int = 32,
                 min_leaf: int = 20, l2: float = 1.0) -> None:  # fmt: skip
        self.n_trees, self.depth, self.lr, self.bins = n_trees, depth, lr, bins
        self.min_leaf, self.l2 = min_leaf, l2

    def _bin(self, x: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.searchsorted(e, x[:, j], side="right") for j, e in enumerate(self.edges)], 1
        )

    def _tree(self, xb: np.ndarray, r: np.ndarray) -> list:
        """One tree on residuals ``r``: nodes as [feature, bin threshold, left, right] or [leaf value]."""
        nodes: list = []

        def grow(idx: np.ndarray, level: int) -> int:
            me = len(nodes)
            nodes.append(None)
            best = None
            if level < self.depth and idx.size >= 2 * self.min_leaf:
                rs, n = r[idx].sum(), idx.size
                for j in range(xb.shape[1]):
                    cnt = np.bincount(xb[idx, j], minlength=self.bins + 1)
                    sm = np.bincount(xb[idx, j], weights=r[idx], minlength=self.bins + 1)
                    cl, sl = np.cumsum(cnt)[:-1], np.cumsum(sm)[:-1]
                    cr, sr = n - cl, rs - sl
                    ok = (cl >= self.min_leaf) & (cr >= self.min_leaf)
                    if not ok.any():
                        continue
                    gain = np.where(ok, sl**2 / (cl + self.l2) + sr**2 / (cr + self.l2), -np.inf)
                    b = int(np.argmax(gain))
                    if best is None or gain[b] > best[0]:
                        best = (gain[b], j, b)
            if best is None:
                nodes[me] = [float(r[idx].sum() / (idx.size + self.l2))]
                return me
            _, j, b = best
            go_left = xb[idx, j] <= b
            left = grow(idx[go_left], level + 1)
            right = grow(idx[~go_left], level + 1)
            nodes[me] = [j, b, left, right]
            return me

        grow(np.arange(xb.shape[0]), 0)
        return nodes

    @staticmethod
    def _apply(nodes: list, xb: np.ndarray) -> np.ndarray:
        out = np.empty(xb.shape[0])
        stack = [(0, np.arange(xb.shape[0]))]
        while stack:
            k, idx = stack.pop()
            node = nodes[k]
            if len(node) == 1:
                out[idx] = node[0]
                continue
            j, b, left, right = node
            m = xb[idx, j] <= b
            stack += [(left, idx[m]), (right, idx[~m])]
        return out

    def fit(self, x: np.ndarray, y: np.ndarray) -> HistBoost:
        qs = np.linspace(0, 1, self.bins + 1)[1:-1]
        self.edges = [np.unique(np.quantile(x[:, j], qs)) for j in range(x.shape[1])]
        xb = self._bin(x)
        self.base = float(y.mean())
        pred = np.full(y.size, self.base)
        self.trees = []
        for _ in range(self.n_trees):
            t = self._tree(xb, y - pred)
            self.trees.append(t)
            pred += self.lr * self._apply(t, xb)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        xb = self._bin(x)
        return self.base + self.lr * np.sum([self._apply(t, xb) for t in self.trees], axis=0)


def with_comparator(m: dict[str, Any], f: pd.DataFrame) -> pd.DataFrame:
    """``f`` with the ERA5 + wind ridge's total and wind sea (``cmp_hs``, ``cmp_ww``, metres)."""
    xe = x_era5_wind(f)
    return f.assign(
        cmp_hs=np.exp(m["era5_wind_hs"].predict(xe)), cmp_ww=np.exp(m["era5_wind_ww"].predict(xe))
    )


def fit_fold(fit: pd.DataFrame) -> dict[str, Any]:
    """Every combiner, fitted on one fold's fitting strips (all inputs and both targets present)."""
    yh, yw = lg(fit["true_hs_m"]), lg(fit["true_wind_wave_m"])
    m: dict[str, Any] = {}
    for name, xf in (
        ("era5_wind", x_era5_wind),
        ("cam_era5_wind", lambda f: np.column_stack([x_camera(f), x_era5_wind(f)])),
    ):
        m[f"{name}_hs"] = Ridge().fit(xf(fit), yh)
        m[f"{name}_ww"] = Ridge().fit(xf(fit), yw)
    fc = with_comparator(m, fit)
    m["swap_gated"] = PartitionFusion(True, "comparator", transfer=True).fit(fc)
    m["swap_full_gated"] = PartitionFusion(True, "comparator").fit(fc)
    m["swap_open"] = PartitionFusion(False, "comparator").fit(fc)
    m["raw_gated"] = PartitionFusion(True, "raw").fit(fc)
    m["hgb_hs"] = HistBoost().fit(x_all(fit), yh)
    m["hgb_ww"] = HistBoost().fit(x_all(fit), yw)
    return m  # fmt: skip


def predict_fold(m: dict[str, Any], f: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Every candidate's H_s and wind-wave height (metres) for the strips of ``f``."""
    out: dict[str, dict[str, np.ndarray]] = {}
    fc = with_comparator(m, f)
    for key, name in (
        ("swap_gated", BAR),
        ("swap_full_gated", "full swap (k = 1) on ERA5 + wind, gated"),
        ("swap_open", "full swap (k = 1) on ERA5 + wind, gate open"),
        ("raw_gated", "partition fusion on raw ERA5, scaled, gated"),
    ):
        hs, ww = m[key].predict(fc)
        out[name] = {"hs": hs, "wind_wave": ww}
    e_hs, e_ww = f["era5_hs"].to_numpy(float), f["era5_ww"].to_numpy(float)
    cam_ww = f["cam_ww"].to_numpy(float)
    out["partition rule on raw ERA5 (nothing fitted)"] = {
        "hs": np.sqrt(np.maximum(e_hs**2 - e_ww**2, 0) + cam_ww**2),
        "wind_wave": cam_ww,
    }
    out[COMPARATOR] = {"hs": fc["cmp_hs"].to_numpy(), "wind_wave": fc["cmp_ww"].to_numpy()}
    xc = np.column_stack([x_camera(f), x_era5_wind(f)])
    out["camera + ERA5 + wind ridge (Test 3)"] = {
        "hs": np.exp(m["cam_era5_wind_hs"].predict(xc)),
        "wind_wave": np.exp(m["cam_era5_wind_ww"].predict(xc)),
    }
    xa = x_all(f)
    out["gradient boosting, all inputs"] = {
        "hs": np.exp(m["hgb_hs"].predict(xa)),
        "wind_wave": np.exp(m["hgb_ww"].predict(xa)),
    }
    out["ERA5 alone (line)"] = {
        "hs": f["era5_line_hs"].to_numpy(float),
        "wind_wave": f["era5_line_ww"].to_numpy(float),
    }
    out["camera alone"] = {"hs": f["cam_hs"].to_numpy(float), "wind_wave": cam_ww}
    return out  # fmt: skip


def inputs_ok(f: pd.DataFrame) -> np.ndarray:
    cols = ["era5_hs", "era5_ww", "era5_sw", "era5_line_hs", "era5_line_ww", "cam_hs", "cam_ww",
            "cam_sw", "cam_sigma_hs", "wind", "gust", *GATE]  # fmt: skip
    return np.asarray(np.isfinite(f[cols].to_numpy(float)).all(axis=1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--oof", type=Path, required=True, help="oof_<arm>.csv of scripts/buoycam_oof.py"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--labels",
        type=Path,
        default=data_dir("processed") / "packages/ndbc_buoycam_heights_v2cv/labels.csv",
    )
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    kw = {"n_boot": args.n_boot, "alpha": 0.05, "seed": 0}
    d = pd.read_csv(args.oof, dtype={"station_id": str, "cv_fold": str})
    labels = pd.read_csv(args.labels, dtype={"station_id": str}, low_memory=False)
    f = features(d, labels)
    ok = inputs_ok(f)
    both = ok & f["true_hs_m"].notna().to_numpy() & f["true_wind_wave_m"].notna().to_numpy()
    folds = sorted(f["cv_fold"].unique())
    preds: dict[str, dict[str, np.ndarray]] = {}
    models, gate = {}, np.full(len(f), np.nan)
    for k in folds:
        test = (f["cv_fold"] == k).to_numpy() & ok
        m = fit_fold(f[both & (f["cv_fold"] != k).to_numpy()])
        models[k] = m
        for name, pr in predict_fold(m, f[test]).items():
            for h, v in pr.items():
                preds.setdefault(name, {}).setdefault(h, np.full(len(f), np.nan))[test] = v
        gate[test] = m["swap_gated"].gate_values(with_comparator(m, f[test]))

    res: dict[str, Any] = {"oof": str(args.oof), "n_strips": len(f), "n_with_inputs": int(ok.sum()),
                           "targets": {}, "gate": {}, "fits": {}}  # fmt: skip
    for tname, tcol in TARGETS.items():
        y = f[tcol].to_numpy(float)
        sel = ok & np.isfinite(y)
        g, grp = f.loc[sel, "station_id"].to_numpy(), f.loc[sel, "station_group"].to_numpy()
        r: dict[str, Any] = {
            "n": int(sel.sum()),
            "buoys": len(set(g)),
            "rmse_m": {},
            "minus_bar_comparator": {},
        }
        for name, pr in preds.items():
            r["rmse_m"][name] = rmse(y[sel], pr[tname][sel])
        base = preds[COMPARATOR][tname][sel]
        for name, pr in preds.items():
            if name == COMPARATOR:
                continue
            r["minus_bar_comparator"][name] = {
                "by_buoy": list(paired_block_bootstrap_ci(y[sel], pr[tname][sel], base, rmse, groups=g, **kw)),
                "by_group": list(paired_block_bootstrap_ci(y[sel], pr[tname][sel], base, rmse, groups=grp, **kw)),
            }  # fmt: skip
        r["bar_side_met"] = bool(r["minus_bar_comparator"][BAR]["by_buoy"][2] < 0)
        res["targets"][tname] = r
    res["bar_met"] = all(res["targets"][t]["bar_side_met"] for t in TARGETS)
    sel = ok & np.isfinite(gate)
    gf = f[sel].assign(gate=gate[sel])
    res["gate"] = {
        "mean": float(gf["gate"].mean()),
        "by_illumination": gf.groupby("illumination")["gate"].agg(["mean", "size"]).round(3).to_dict("index"),
        "by_camera_spread_quintile": gf.groupby(pd.qcut(gf["cam_sigma_ww"].rank(method="first"), 5, labels=False))["gate"].mean().round(3).to_dict(),
        "share_below_0.5": float((gf["gate"] < 0.5).mean()),
    }  # fmt: skip
    for k, m in models.items():
        res["fits"][k] = {
            key: m[key].describe()
            for key in ("swap_gated", "swap_full_gated", "swap_open", "raw_gated")
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fusion.json").write_text(json.dumps(res, indent=1, default=str))
    out = f[["sample_id", "station_id", "cv_fold", "true_hs_m", "true_wind_wave_m"]].copy()
    out["gate"] = gate
    for name, pr in preds.items():
        slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
        for h, v in pr.items():
            out[f"{slug}__{h}_m"] = v
    out.to_csv(args.out / "fusion_oof.csv", index=False, float_format="%.4f")
    with (args.out / "models.pkl").open("wb") as fh:
        pickle.dump(models, fh)

    lines = [f"# Stage 2: partition-aware fusion, out of fold ({args.oof})\n"]
    for tname, r in res["targets"].items():
        lines.append(f"## {tname} ({r['n']:,} strips, {r['buoys']} buoys)\n")
        for name, v in sorted(r["rmse_m"].items(), key=lambda kv: kv[1]):
            dd = r["minus_bar_comparator"].get(name)
            lines.append(f"- {name}: RMSE {v:.3f} m" + (
                f"; minus ERA5 + wind {dd['by_buoy'][0]:+.3f} [{dd['by_buoy'][1]:+.3f}, {dd['by_buoy'][2]:+.3f}] "
                f"(by group [{dd['by_group'][1]:+.3f}, {dd['by_group'][2]:+.3f}])" if dd else ""))  # fmt: skip
        lines.append("")
    gt = res["gate"]
    lines.append(f"Gate: mean {gt['mean']:.2f}; below 0.5 on {gt['share_below_0.5']:.1%} of strips; by light "
                 + ", ".join(f"{k} {v['mean']:.2f}" for k, v in gt["by_illumination"].items())
                 + "; by camera-spread quintile " + ", ".join(f"{v:.2f}" for v in gt["by_camera_spread_quintile"].values()))  # fmt: skip
    lines.append(f"\n**Bar S2 {'met' if res['bar_met'] else 'not met'}.**")
    text = "\n".join(lines) + "\n"
    (args.out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
