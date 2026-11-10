"""Sea-state classes, splits, baselines and reporting (scripts/sea_state_package.py, train_sea_state.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wave_analysis.datasets.sea_state import (
    TEST_LATER,
    add_calendar,
    linear_correction,
    majority_baseline,
    monthly_baseline,
    near_boundary,
    nowcast_at,
    package_path,
    sea_state_class,
    year_split,
)
from wave_analysis.datasets.splits import EXCLUDED, TEST, TRAIN, VAL
from wave_analysis.evaluation.sea_state import confusion_svg, report

NAMES = ["smooth", "slight", "moderate", "rough_or_more"]


def test_wmo_bands_are_closed_on_the_right() -> None:
    # Code table 3700: "the exact bounding height shall be assigned to the lower code figure".
    hs = [0.0, 0.5, 0.51, 1.25, 1.26, 2.5, 2.51, 9.0, np.nan]
    assert sea_state_class(hs).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, -1]


def test_near_boundary_is_relative() -> None:
    got = near_boundary([0.49, 0.47, 1.29, 1.31, 2.41, 2.39], rel_tol=0.04)
    assert got.tolist() == [True, False, True, False, True, False]


def _days(start: str, end: str) -> pd.DataFrame:
    t = pd.date_range(start, end, freq="1D", tz="UTC") + pd.Timedelta(hours=17)
    return add_calendar(pd.DataFrame({"time_utc": t}), tz="America/New_York")


def test_year_split_with_validation_weeks_and_embargo() -> None:
    df = _days("2022-01-01", "2026-06-30")
    spec = {
        "train_years": [2022, 2023, 2024],
        "test_years": [2025],
        "later_years": [2026],
        "val": {"group_col": "local_week", "fraction": 0.15, "seed": 0},
        "embargo": "72h",
    }
    lab = year_split(df, spec).labels
    year = df["local_year"]
    assert set(lab[year <= 2024]) == {TRAIN, VAL}
    assert set(lab[year == 2026]) == {TEST_LATER}
    first_days = df["local_day"].isin(["2025-01-01", "2025-01-02", "2025-01-03"])
    assert (lab[first_days] == EXCLUDED).all()
    assert (lab[(year == 2025) & ~first_days] == TEST).all()
    share = (lab == VAL).sum() / (year <= 2024).sum()
    assert 0.10 < share < 0.20
    # whole weeks: no week is in both train and val
    weeks = df.groupby("local_week")[[]].size()
    per_week = lab[year <= 2024].groupby(df["local_week"]).nunique()
    assert (per_week == 1).all() and len(weeks) > 0


def test_calendar_uses_local_days() -> None:
    df = add_calendar(
        pd.DataFrame({"time_utc": pd.to_datetime(["2025-01-01T03:00:00Z"], utc=True)}),
        tz="America/New_York",
    )
    assert df["local_day"].iloc[0] == "2024-12-31" and df["local_year"].iloc[0] == 2024


def test_baselines_without_images() -> None:
    y = np.array([1, 1, 1, 0, 2, 2, 2, 1])
    month = np.array([1, 1, 1, 1, 7, 7, 7, 7])
    train = np.array([True] * 6 + [False] * 2)
    assert majority_baseline(y, train).tolist() == [1] * 8
    assert monthly_baseline(y, month, train).tolist() == [1, 1, 1, 1, 2, 2, 2, 2]


def test_nowcast_interpolates_and_respects_gaps() -> None:
    idx = pd.to_datetime(
        ["2025-01-01T00:00Z", "2025-01-01T01:00Z", "2025-01-01T10:00Z", "2025-01-01T11:00Z"],
        utc=True,
    )
    s = pd.Series([1.0, 2.0, 5.0, 6.0], index=idx)
    t = pd.Series(
        pd.to_datetime(["2025-01-01T00:30Z", "2025-01-01T05:00Z", "2025-01-02T00:00Z"], utc=True)
    )
    got = nowcast_at(t, s, max_gap="3h")
    assert got[0] == pytest.approx(1.5)
    assert np.isnan(got[1]) and np.isnan(got[2])


def test_linear_correction() -> None:
    x = np.array([1.0, 2.0, 3.0, np.nan])
    a, b = linear_correction(x, 0.5 + 2 * x, np.ones(4, bool))
    assert (a, b) == pytest.approx((0.5, 2.0))


def test_package_path() -> None:
    t = pd.Timestamp("2024-06-15T17:00:08Z")
    assert package_path(t, "x.jpg") == "images/2024/06/15/x.jpg"


def test_report_counts_and_intervals() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)
    days = np.repeat(np.arange(40), 10)
    rep = report(y, y, groups=days, names=NAMES, n_boot=50)
    assert rep["balanced_accuracy"]["value"] == pytest.approx(1.0)
    assert rep["n_days"] == 40
    assert sum(s["stills"] for s in rep["support"].values()) == 400
    majority = report(y, np.ones(400, int), groups=days, names=NAMES, n_boot=50)
    assert majority["f1_macro"]["value"] < 0.2  # never-predicted classes count as 0
    assert majority["quadratic_weighted_kappa"]["value"] == pytest.approx(0.0)
    nb = np.zeros(400, bool)
    nb[:100] = True
    assert (
        report(y, y, groups=days, names=NAMES, near_boundary=nb, n_boot=10)[
            "without_near_boundary"
        ]["n"]
        == 300
    )


def test_confusion_svg() -> None:
    svg = confusion_svg(
        [[5, 1, 0, 0], [2, 50, 3, 0], [0, 4, 20, 1], [0, 0, 2, 3]], NAMES, title="t"
    )
    assert svg.startswith("<svg") and ">50<" in svg and "n = 55" in svg


# --------------------------------------------------------------------------- torch parts


def test_training_pieces() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from PIL import Image

    from wave_analysis.models.sea_state import (
        TrainSettings,
        build_model,
        build_transforms,
        class_weights,
        crop,
        input_size,
        lr_lambda,
    )

    s = TrainSettings.from_config({"input": "sea_crop", "sea_crop": [0.27, 0.6], "epochs": 3})
    assert input_size(s, (509, 900)) == (168, 900)
    img = Image.new("RGB", (900, 509))
    assert crop(img, s).size == (900, 168)
    x = build_transforms(s, (168, 900), train=True)(crop(img, s))
    assert tuple(x.shape) == (3, 168, 900)
    with pytest.raises(ValueError, match="fixed"):
        build_transforms(TrainSettings(augment={"horizontal_flip": True}), (10, 10), train=True)
    w = class_weights([1, 1, 1, 0], 4)
    assert w[2] == 0 and w[0] > w[1] and float(w.mean()) == pytest.approx(1.0)
    assert lr_lambda(0, warmup=10, total=100) == pytest.approx(0.1)
    assert lr_lambda(100, warmup=10, total=100) == pytest.approx(0.0)
    net = build_model("resnet18", 4, pretrained=False).eval()
    with torch.no_grad():
        assert tuple(net(x[None]).shape) == (1, 4)
