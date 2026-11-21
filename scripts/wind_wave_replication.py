"""Apply the frozen wind-wave weights to a closed C4b window, or replay the historical benchmark."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import wind_wave_reliability as wr
from wave_analysis.config import repo_root
from wave_analysis.ingest.checksum import sha256_file

ROOT = repo_root()
REPORTS = Path(__file__).resolve().parents[1] / "reports/wind-wave-reliability"
WEIGHTS = REPORTS / "fits.json"
WEIGHTS_SHA256 = "307c923d92793c9c23714ad5053f42dd4c2b86c8e3a0bbc853387280621a2a2a"
KEYS = ["sample_id", "station_id", "station_group", "cv_fold"]
PRIMARY = ["constant", "varweight"]


def closed_window(november_cut, followup_cut):
    def cutoff(record):
        end = pd.Timestamp(record["until_utc"])
        if record["rule_met"] is not True or end.tzinfo is None or pd.isna(end):
            raise ValueError("expected a completed, timezone-aware C4b cutoff")
        return end.tz_convert("UTC")

    prior = cutoff(november_cut)
    if prior < pd.Timestamp("2026-11-21 16:00Z"):
        raise ValueError("November confirmation has not reached its registered cutoff")
    start = max(pd.Timestamp("2026-11-28 00:00Z"), prior.normalize() + pd.Timedelta(days=1))
    end = start + pd.Timedelta(days=14)
    if cutoff(followup_cut) < end:
        raise ValueError(f"follow-up coverage must reach {end.isoformat()}")
    return start, end


def input_rows(path, labels_path, arrangement):
    rows = pd.read_parquet(
        path,
        columns=[*KEYS, "observed", "forecast", "camera", "fusion", "varweight"],
        filters=[
            ("arrangement", "==", arrangement),
            ("baseline", "==", "model_wind"),
            ("target", "==", wr.WIND),
        ],
    )
    labels = pd.read_csv(
        labels_path,
        usecols=[*KEYS, "time_utc", "wind_speed_mps", "wind_wave_label_m"],
        dtype={"station_id": str, "cv_fold": str},
    ).rename(columns={"wind_speed_mps": "wind"})
    rows["cv_fold"] = rows.cv_fold.astype(str)
    labels["time_utc"] = pd.to_datetime(labels.time_utc, utc=True)
    if (
        labels.time_utc.isna().any()
        or labels.sample_id.duplicated().any()
        or rows.sample_id.duplicated().any()
    ):
        raise ValueError("missing timestamp or duplicate sample identifier")
    matched = rows.merge(labels, on=KEYS, how="left", validate="1:1", indicator=True)
    if matched._merge.ne("both").any() or not np.allclose(
        matched.observed, matched.wind_wave_label_m, rtol=0, atol=1e-5
    ):
        raise ValueError("C4b rows and package labels do not agree")
    cols = ["forecast", "camera", "fusion", "varweight", "wind", "observed"]
    if not np.isfinite(matched[cols].to_numpy()).all():
        raise ValueError("C4b reported an eligible row with an incomplete input")
    return matched.drop(columns=["_merge", "wind_wave_label_m"]), labels


def apply_frozen(rows, fits):
    """Compute all three historical height blends from inputs available at prediction time."""
    out = rows.copy()
    if not set(out.cv_fold).issubset(fits):
        raise ValueError("unknown geographic fold")
    out["disagreement"] = out.camera - out.forecast
    for arm in ("constant", "two_inputs", "reliability"):
        out[arm] = np.nan
    for k, part in out.groupby("cv_fold"):
        fit = fits[k]
        out.loc[part.index, "constant"] = (
            part.forecast + fit["reliability"]["constant"] * part.disagreement
        )
        for arm, features in (("two_inputs", wr.INPUTS[:2]), ("reliability", wr.INPUTS)):
            rule = fit[arm]
            # Cell indices use the training feature order even if a JSON reader reorders keys.
            cuts = {name: rule["cuts"][name] for name in features}
            w = np.asarray(rule["weights"])[wr.bin_ids(part, cuts)]
            out.loc[part.index, arm] = part.forecast + w * part.disagreement
    return out


def replay(frozen):
    path = ROOT / "runs/forecast-fusion-nested-v1/rows.parquet"
    if sha256_file(path) != frozen["historical_rows_sha256"]:
        raise ValueError("historical C4b archive differs from the frozen reference")
    labels = ROOT / "data/processed/packages/ndbc_buoycam_heights_v2cvn/labels.csv"
    rows, _ = input_rows(path, labels, "nested")
    scores, _ = wr.score(
        "unseen", apply_frozen(rows, frozen["folds"]), wr.TRANSFER_ARMS, "reliability", []
    )
    keys = ["population", "scope", "unit"]
    actual = scores.astype({"unit": str}).set_index(keys).sort_index()
    reference = pd.read_csv(REPORTS / "scores.csv", dtype={"unit": str}).query(
        "population == 'unseen'"
    )
    reference = reference.set_index(keys).sort_index()
    pd.testing.assert_frame_equal(
        actual, reference[actual.columns], atol=1e-10, rtol=0, check_dtype=False
    )
    print(
        f"Matched {len(scores)} pooled, fold and group score rows on {len(rows):,} historical strips."
    )


def score_followup(args, frozen):
    if args.out.exists():
        raise FileExistsError(args.out)
    base = json.loads((args.c4b_run / "claims.json").read_text())
    start, end = closed_window(json.loads(args.november_cut.read_text()), base["cut"])
    path = args.c4b_run / "c4b_rows.parquet"
    rows, labels = input_rows(path, args.labels, "fresh")
    if len(rows) != base["C4b"]["model_wind"][wr.WIND]["strips"]:
        raise ValueError("C4b row count differs from its recorded result")
    selected = rows.time_utc.between(start, end, inclusive="left")
    predictions = apply_frozen(rows[selected], frozen["folds"])
    available = labels.time_utc.between(start, end, inclusive="left")
    summary = {
        "start_utc_inclusive": start.isoformat(),
        "end_utc_exclusive": end.isoformat(),
        "available_package_strips": int(available.sum()),
        "matched_strips": len(predictions),
        "matched_buoys": predictions.station_id.nunique(),
        "matched_groups": predictions.station_group.nunique(),
        "weights_sha256": WEIGHTS_SHA256,
        "c4b_rows_sha256": sha256_file(path),
        "labels_sha256": sha256_file(args.labels),
        "both_primary_intervals_below_zero": None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as directory:
        stage = Path(directory) / "result"
        stage.mkdir()
        if len(predictions):
            scores, contrasts = wr.score(
                "replication",
                predictions,
                wr.TRANSFER_ARMS,
                "reliability",
                [*PRIMARY, "fusion", "forecast", "two_inputs"],
            )
            pooled = contrasts.query("scope == 'all' and weighting == 'strip'").set_index(
                "comparator"
            )
            summary["both_primary_intervals_below_zero"] = bool(
                (pooled.loc[PRIMARY, "hi_m"] < 0).all()
            )
            scores.to_csv(stage / "scores.csv", index=False)
            contrasts.to_csv(stage / "contrasts.csv", index=False)
        predictions.to_parquet(stage / "rows.parquet", index=False)
        (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        stage.rename(args.out)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmds = parser.add_subparsers(dest="command", required=True)
    cmds.add_parser("replay", help="verify all historical scores using the saved weights")
    score = cmds.add_parser("score", help="score the closed two-week replication")
    score.add_argument(
        "--november-cut", type=Path, required=True, help="original November claims/cut.json"
    )
    score.add_argument(
        "--c4b-run", type=Path, required=True, help="separate follow-up C4b output directory"
    )
    score.add_argument("--labels", type=Path, required=True, help="follow-up package labels.csv")
    score.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(WEIGHTS) != WEIGHTS_SHA256:
        raise ValueError("frozen weight checksum mismatch")
    frozen = json.loads(WEIGHTS.read_text())
    if args.command == "replay":
        replay(frozen)
    else:
        score_followup(args, frozen)


if __name__ == "__main__":
    main()
