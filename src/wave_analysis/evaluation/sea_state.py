"""Reporting a sea-state classifier: ordinal metrics with day-bootstrap intervals and a confusion matrix.

What is reported, and why (see ``docs/experiments/currituck_sea_state.md``):

* **Balanced accuracy and macro-F1 lead**, not accuracy: one class can hold
  most stills (always answering "slight" scores about 62 % at Currituck).
* **Quadratic-weighted kappa and within-one accuracy**, because the classes
  are ordered: calling a moderate sea slight is a smaller error than calling
  it smooth.
* **Counts beside every rate**: a class seen on a few dozen days gives a
  noisy row.
* **Intervals from a bootstrap over whole days**: stills of one day share
  their sea state, so they are not independent samples.
* **The same metrics without stills near a class boundary**, where the
  buoy's own sampling error can put the label in either class.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from html import escape
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from wave_analysis.evaluation.classification import classification_report, confusion_matrix
from wave_analysis.evaluation.uncertainty import block_bootstrap_ci

#: Metrics given with intervals, in report order.
HEADLINE = (
    "balanced_accuracy",
    "f1_macro",
    "quadratic_weighted_kappa",
    "within_one_accuracy",
    "accuracy",
)


def _metric(name: str, k: int) -> Callable[[np.ndarray, np.ndarray], float]:
    def f(a: np.ndarray, b: np.ndarray) -> float:
        rep = classification_report(a.astype(np.int64), b.astype(np.int64), n_classes=k)
        return float(rep[name])  # type: ignore[arg-type]

    return f


def report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    groups: ArrayLike,
    names: Sequence[str],
    near_boundary: ArrayLike | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Metrics of one prediction set on one partition (rows with a true class >= 0).

    Returns ``n``, ``n_days``, ``support`` (stills and days per class),
    ``confusion_matrix``, the per-class table, each :data:`HEADLINE` metric as
    ``{"value", "lo", "hi"}``, and ``without_near_boundary`` (the same
    metrics without the stills ``near_boundary`` marks, no intervals).
    """
    a = np.asarray(y_true, dtype=np.int64)
    b = np.asarray(y_pred, dtype=np.int64)
    g = np.asarray(groups)
    nb = np.zeros(len(a), bool) if near_boundary is None else np.asarray(near_boundary, bool)
    keep = (a >= 0) & (b >= 0)
    a, b, g, nb = a[keep], b[keep], g[keep], nb[keep]
    k = len(names)
    out: dict[str, Any] = {"n": int(a.size), "n_days": int(np.unique(g).size)}
    if a.size == 0:
        return out
    rep = classification_report(a, b, n_classes=k, class_names=list(names))
    out["confusion_matrix"] = rep["confusion_matrix"]
    out["per_class"] = rep["per_class"]
    out["support"] = {
        names[i]: {"stills": int((a == i).sum()), "days": int(np.unique(g[a == i]).size)}
        for i in range(k)
    }
    for name in HEADLINE:
        value, lo, hi = block_bootstrap_ci(
            a, b, _metric(name, k), groups=g, n_boot=n_boot, alpha=alpha, seed=seed
        )
        out[name] = {"value": value, "lo": lo, "hi": hi}
    if nb.any() and (~nb).any():
        sub = classification_report(a[~nb], b[~nb], n_classes=k)
        without: dict[str, Any] = {"n": int((~nb).sum())}
        for name in HEADLINE:
            without[name] = float(sub[name])  # type: ignore[arg-type]
        out["without_near_boundary"] = without
    return out


#: What the bootstrap groups are called in text, by grouping column.
GROUP_UNITS = {"local_day": "days", "station_id": "buoys", "station_day": "buoy-days"}


def format_line(name: str, rep: dict[str, Any], *, groups: str = "days") -> str:
    """One text line: the headline metrics with their intervals.

    ``groups`` names the bootstrap's groups (``rep["n_days"]`` counts them):
    days at Currituck, buoys or buoy-days for the buoy cameras (:data:`GROUP_UNITS`).
    """
    if not rep.get("n"):
        return f"{name}: no stills"
    parts = [
        f"{m} {rep[m]['value']:.3f} [{rep[m]['lo']:.3f}, {rep[m]['hi']:.3f}]" for m in HEADLINE[:4]
    ]
    return f"{name}: n={rep['n']:,} ({rep['n_days']} {groups}); " + "; ".join(parts)


def confusion_svg(cm: Sequence[Sequence[int]], names: Sequence[str], *, title: str = "") -> str:
    """A confusion matrix as a self-contained SVG: counts, and the share of each true class.

    Rows are the true classes (with their totals), columns the predictions.
    Cell shading is the row share, so rare classes are as readable as common
    ones.
    """
    m = np.asarray(cm, dtype=np.int64)
    k = m.shape[0]
    rows = m.sum(axis=1, keepdims=True)
    share = np.divide(m, rows, out=np.zeros(m.shape, float), where=rows > 0)
    cell, left, top = 92, 150, 70
    width, height = left + cell * k + 110, top + cell * k + 60
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif" font-size="13">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{left}" y="22" font-size="15" font-weight="bold">{escape(title)}</text>',
        f'<text x="{left + cell * k / 2}" y="{top - 30}" text-anchor="middle">predicted</text>',
        f'<text x="18" y="{top + cell * k / 2}" text-anchor="middle" '
        f'transform="rotate(-90 18 {top + cell * k / 2})">true</text>',
    ]
    for j, nm in enumerate(names):
        out.append(
            f'<text x="{left + cell * j + cell / 2}" y="{top - 10}" text-anchor="middle">'
            f"{escape(nm)}</text>"
        )
    for i, nm in enumerate(names):
        y0 = top + cell * i
        out.append(
            f'<text x="{left - 8}" y="{y0 + cell / 2 + 4}" text-anchor="end">{escape(nm)}</text>'
        )
        for j in range(k):
            x0 = left + cell * j
            s = float(share[i, j])
            shade = round(255 - 200 * s)
            fill = f"rgb({shade},{shade + (255 - shade) // 3},255)"
            ink = "white" if s > 0.55 else "black"
            out += [
                f'<rect x="{x0}" y="{y0}" width="{cell}" height="{cell}" fill="{fill}" '
                f'stroke="#888" stroke-width="0.5"/>',
                f'<text x="{x0 + cell / 2}" y="{y0 + cell / 2 - 2}" text-anchor="middle" '
                f'fill="{ink}" font-weight="bold">{int(m[i, j]):,}</text>',
                f'<text x="{x0 + cell / 2}" y="{y0 + cell / 2 + 16}" text-anchor="middle" '
                f'fill="{ink}">{100 * s:.0f} %</text>',
            ]
        out.append(
            f'<text x="{left + cell * k + 10}" y="{y0 + cell / 2 + 4}">n = {int(rows[i, 0]):,}</text>'
        )
    out.append(
        f'<text x="{left}" y="{height - 20}" fill="#555">cell: stills (share of the true class)</text>'
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


__all__ = ["HEADLINE", "confusion_matrix", "confusion_svg", "format_line", "report"]
