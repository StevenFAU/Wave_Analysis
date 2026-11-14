"""Does the camera add to the hindcast? Combine a run's class probabilities with WIS or ERA5.

Reads a finished year-split run of ``scripts/train_sea_state.py`` and the
package's hindcast heights, and writes ``<run>/fusion/report.md`` and
``fusion.json``. Choices fixed before any test partition was scored
(2026-11-14, ``docs/experiments/currituck_sea_state.md``):

* Combiner: multinomial logistic regression on the camera's four class
  log-probabilities (clipped at 1e-6) and log Hs of the hindcast,
  standardized; inverse class-frequency weights; L2 penalty 1e-3 on the
  coefficients; no tuning.
* Fitted on the run's **val** partition (15 % of the weeks of 2022-2024),
  stills where the hindcast exists. The test partitions are scored once.
* Compared, on the same stills, with (a) the same regression on the hindcast
  alone, fitted on the same val stills, which isolates what the camera adds
  from what refitting adds; (b) the package's hindcast baseline (linear
  correction fitted on the training stills); (c) the camera alone.
* Gains with 95 % intervals from a paired bootstrap over whole days.

Known optimism: the camera's epoch was chosen on the same val partition, so
its val probabilities are slightly better than on unseen data, and the
combiner may trust the camera a little too much. That works against the
fusion on the test partitions, not for it.

    python scripts/sea_state_fusion.py --run runs/<year-split run>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wave_analysis.config import load_yaml
from wave_analysis.evaluation.classification import classification_report
from wave_analysis.evaluation.sea_state import format_line, report
from wave_analysis.evaluation.uncertainty import paired_block_bootstrap_ci
from wave_analysis.models.fusion import fit_logistic, inverse_frequency_weights

L2 = 1e-3
P_FLOOR = 1e-6
GAINS = ("balanced_accuracy", "f1_macro")
#: Hindcast -> (height column, package baseline column, test partitions it covers).
HINDCASTS = {
    "wis": ("wis_hs_m", "wis_year", ("test",)),
    "era5": ("era5_hs_m", "era5_year", ("test", "test_later")),
}


def _metric(name: str, k: int) -> Any:
    def f(a: np.ndarray, b: np.ndarray) -> float:
        rep = classification_report(a.astype(np.int64), b.astype(np.int64), n_classes=k)
        return float(rep[name])  # type: ignore[arg-type]

    return f


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="run directory of a year-split run")
    args = ap.parse_args(argv)
    run = Path(args.run)
    info = json.loads((run / "metrics.json").read_text())["run"]
    if info["split"] != "year":
        raise SystemExit("the combiner is fitted on the year split's val weeks; give a year run")
    prov = load_yaml(run / "provenance.yaml")
    cfg = prov["pipeline"]["config"]
    names = list(cfg["task"]["classes"]["names"])
    k = len(names)
    boot = cfg["evaluation"]["bootstrap"]
    labels = [f for f in prov["input_files"] if str(f).endswith("labels.csv")]
    if not labels:
        raise SystemExit(f"no package labels.csv among the inputs of {run / 'provenance.yaml'}")
    package = Path(str(labels[0])).parent

    pred = pd.read_csv(run / "predictions.csv")
    base = pd.read_csv(package / "baselines.csv")
    d = pred.merge(base, on="sample_id", how="left", validate="one_to_one")
    cam = np.log(np.clip(d[[f"p_{n}" for n in names]].to_numpy(), P_FLOOR, None))

    out_dir = run / "fusion"
    out_dir.mkdir(exist_ok=True)
    results: dict[str, Any] = {
        "run": str(run),
        "choices": {
            "l2": L2,
            "p_floor": P_FLOOR,
            "fit_on": "val",
            "features": "log p_camera, log Hs",
        },
    }
    lines: list[str] = []
    for hc, (hs_col, base_col, parts) in HINDCASTS.items():
        hs = d[hs_col].to_numpy(dtype=np.float64)
        ok = np.isfinite(hs) & (hs > 0)
        log_hs = np.where(ok, np.log(np.where(ok, hs, 1.0)), np.nan)[:, None]
        x_both = np.hstack([cam, log_hs])
        fit = (d["partition"] == "val").to_numpy() & ok
        y = d["true"].to_numpy(dtype=np.int64)
        w = inverse_frequency_weights(y[fit], k)
        both = fit_logistic(x_both[fit], y[fit], k, sample_weight=w, l2=L2)
        alone = fit_logistic(log_hs[fit], y[fit], k, sample_weight=w, l2=L2)
        res_h: dict[str, Any] = {
            "fit_stills": int(fit.sum()),
            "fit_days": int(d.loc[fit, "local_day"].nunique()),
        }
        lines.append(
            f"## {hc}: combiners fitted on {res_h['fit_stills']:,} val stills "
            f"({res_h['fit_days']} days)\n"
        )
        for part in parts:
            m = (d["partition"] == part).to_numpy() & ok
            t = d[m]
            preds = {
                "camera + hindcast": both.predict(x_both[m]),
                "hindcast alone, refitted on val": alone.predict(log_hs[m]),
                f"hindcast baseline ({base_col})": t[base_col].to_numpy(dtype=np.int64),
                "camera alone": t["pred"].to_numpy(dtype=np.int64),
            }
            g = t["local_day"].to_numpy()
            reps = {
                name: report(
                    t["true"],
                    p,
                    groups=g,
                    names=names,
                    near_boundary=t["near_boundary"],
                    n_boot=int(boot["n_boot"]),
                    alpha=float(boot["alpha"]),
                    seed=int(boot["seed"]),
                )
                for name, p in preds.items()
            }
            gains: dict[str, Any] = {}
            fused = preds["camera + hindcast"]
            for other in list(preds)[1:]:
                for metric in GAINS:
                    v, lo, hi = paired_block_bootstrap_ci(
                        t["true"].to_numpy(),
                        fused,
                        preds[other],
                        _metric(metric, k),
                        groups=g,
                        n_boot=int(boot["n_boot"]),
                        alpha=float(boot["alpha"]),
                        seed=int(boot["seed"]),
                    )
                    gains[f"{metric} vs {other}"] = {"value": v, "lo": lo, "hi": hi}
            res_h[part] = {"metrics": reps, "gains_of_camera_plus_hindcast": gains}
            lines.append(f"### {part} (n={len(t):,}, {t['local_day'].nunique()} days)\n\n```")
            lines += [format_line(name, r) for name, r in reps.items()]
            lines.append("")
            lines += [
                f"gain in {key}: {g_['value']:+.3f} [{g_['lo']:+.3f}, {g_['hi']:+.3f}]"
                for key, g_ in gains.items()
            ]
            lines.append("```\n")
            lines.append(
                "confusion, camera + hindcast (rows true): "
                + " / ".join(str(r) for r in reps["camera + hindcast"]["confusion_matrix"])
                + "\n"
            )
        results[hc] = res_h
    (out_dir / "fusion.json").write_text(json.dumps(results, indent=1, default=str))
    text = f"# Camera + hindcast, {run.name}\n\n" + "\n".join(lines)
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
