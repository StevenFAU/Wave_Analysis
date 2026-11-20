"""Fetch just the displacement samples needed by the archived CDIP clip windows."""

import argparse
import hashlib
import re
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from wave_analysis.physics.timeseries import zero_crossing_analysis


def query(base, fields):
    url = base + ".ascii?" + urllib.parse.quote(fields, safe=",:")
    with urllib.request.urlopen(url, timeout=90) as res:
        raw = res.read()
    text = raw.decode().split("---------------------------------------------", 1)[1].strip()
    return text, hashlib.sha256(raw).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if (a.out / "displacement.csv").exists():
        raise FileExistsError(a.out / "displacement.csv")
    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.index).query("status=='archived'").copy()
    df["day"] = pd.to_datetime(df.start_s, unit="s", utc=True).dt.date
    output = []
    for url, group in df.groupby("opendap"):
        meta, _ = query(url, "xyzStartTime,xyzSampleRate,xyzFilterDelay")
        params = {k: float(v) for k, v in (line.split(", ") for line in meta.splitlines())}
        fs = params["xyzSampleRate"]
        start = params["xyzStartTime"] - params["xyzFilterDelay"]
        for day, g in group.groupby("day"):
            lo, hi = np.ceil(
                (np.array([g.start_s.min(), g.end_s.max()]) - start) * fs - 1e-7
            ).astype(int)
            if lo < 0:
                raise ValueError("record begins before displacement time base")
            sel = f"[{lo}:1:{hi - 1}]"
            cache = a.out / f"{g.deployment.iloc[0]}-{day}-{lo}-{hi}.npz"
            if not cache.exists():
                text, digest = query(url, f"xyzFlagPrimary{sel},xyzZDisplacement{sel}")
                blocks = dict(re.findall(r"(xyz\w+)\[\d+\]\s*\n([^\n]+)", text))
                z = np.fromstring(blocks["xyzZDisplacement"], sep=",", dtype=float)
                q = np.fromstring(blocks["xyzFlagPrimary"], sep=",", dtype=int)
                if len(z) != hi - lo or len(q) != hi - lo:
                    raise ValueError("incomplete remote sample interval")
                tmp = cache.with_suffix(".partial.npz")
                np.savez_compressed(
                    tmp,
                    z=z,
                    q=q,
                    source=url,
                    sha256=digest,
                    first_index=lo,
                    rate=fs,
                    first_utc=start + lo / fs,
                )
                tmp.replace(cache)
            with np.load(cache) as saved:
                z, q = saved["z"], saved["q"]
                for r in g.itertuples():
                    i, j = (
                        np.ceil((np.array([r.start_s, r.end_s]) - start) * fs - 1e-7).astype(int)
                        - lo
                    )
                    eta = z[i:j]
                    flags = q[i:j]
                    row = {
                        "clip_id": r.clip_id,
                        "deployment": r.deployment,
                        "record_duration_s": r.end_s - r.start_s,
                        "reference_start_s": r.start_s,
                        "reference_end_s": r.end_s,
                        "reference_sha256": str(saved["sha256"]),
                        "qc_good_fraction": (flags == 1).mean(),
                        "status": "gap_or_flagged",
                    }
                    clean = np.isfinite(eta).all() and (abs(eta) < 999).all()
                    if clean and np.isin(flags, [1, 2]).all():
                        try:
                            waves = zero_crossing_analysis(eta, fs)
                            row.update(
                                status="good" if (flags == 1).all() else "includes_unevaluated",
                                n_waves=waves.n_waves,
                                mean_height_m=waves.H_mean,
                                max_height_m=waves.H_max,
                                tz_s=waves.T_z,
                            )
                        except ValueError:
                            row["status"] = "no_complete_waves"
                    output.append(row)
            print(cache.name, flush=True)
    pd.DataFrame(output).to_csv(a.out / "displacement.csv", index=False, mode="x")


if __name__ == "__main__":
    main()
