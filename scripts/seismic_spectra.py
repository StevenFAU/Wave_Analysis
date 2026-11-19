"""Extract response-corrected hourly spectra from archived vertical seismograms."""

import argparse
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime, read, read_inventory
from scipy.signal import welch


def process(files, inventory, out):
    @lru_cache(maxsize=64)
    def response_power(channel_id, epoch, fs):
        response = inventory.get_response(channel_id, UTCDateTime(epoch))
        values, frequency = response.get_evalresp_response(1 / fs, round(256 * fs), output="VEL")
        return frequency, np.abs(values) ** 2

    out.mkdir(parents=True, exist_ok=True)
    for file in sorted(files):
        day = date.fromisoformat(file.stem.rsplit(".", 1)[1])
        dest = out / (file.stem + ".csv")
        if dest.exists():
            continue
        # One output row per UTC hour, including explicit reasons for exclusions.
        stream = read(str(file))
        if len({tr.id for tr in stream}) != 1:
            raise ValueError(f"ambiguous channel: {file.name}")
        rows = []
        for h in range(24):
            start = UTCDateTime(day.isoformat()) + 3600 * h
            selected = stream.slice(start, start + 3600, nearest_sample=False).copy()
            row = {"utc": str(start), "status": "incomplete"}
            rows.append(row)
            if not selected or selected.get_gaps():
                continue
            selected.merge(method=0)
            tr = selected[0]
            fs = tr.stats.sampling_rate
            x = tr.data[: round(fs * 3600)]
            if (
                len(x) != round(fs * 3600)
                or np.ma.count_masked(x)
                or abs(tr.stats.starttime - start) > 1 / fs
            ):
                continue
            net, sta, loc, chan = tr.id.split(".")
            epochs = [
                c
                for n in inventory.select(network=net, station=sta, location=loc, channel=chan)
                for s in n
                for c in s
                if c.start_date <= start
                and (c.end_date is None or c.end_date >= start + 3600 - 1 / fs)
            ]
            if len(epochs) != 1:
                row["status"] = "response_epoch_missing_or_ambiguous"
                continue
            f, raw = welch(x.astype(float), fs=fs, nperseg=round(fs * 256), detrend="linear")
            freq, gain = response_power(tr.id, float(epochs[0].start_date) + 1, fs)
            if not np.array_equal(f, freq):
                raise ValueError("mismatched calibration frequencies")
            row["status"] = "ok"
            for name, lo, hi in (("primary", 0.05, 0.1), ("secondary", 0.1, 0.3)):
                band = (f >= lo) & (f < hi)
                power = raw[band] / gain[band]
                if not np.all(np.isfinite(power)) or np.max(power) <= 0:
                    row["status"] = "invalid_power"
                    break
                row[name + "_velocity_variance"] = np.trapezoid(power, f[band])
                row[name + "_peak_hz"] = f[band][np.argmax(power)]
            if row["status"] == "ok":
                row["wave_period_proxy_s"] = 2 / row["secondary_peak_hz"]
        temp = dest.with_suffix(".part")
        pd.DataFrame(rows).to_csv(temp, index=False)
        temp.replace(dest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("files", nargs="+", type=Path)
    a = p.parse_args()
    process(a.files, read_inventory(a.inventory), a.out)


if __name__ == "__main__":
    main()
