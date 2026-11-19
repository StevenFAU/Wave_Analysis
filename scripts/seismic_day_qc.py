"""Inspect one-channel miniSEED and measure the uncalibrated microseism band per hour."""

import argparse
from pathlib import Path

import numpy as np
from obspy import read
from scipy.signal import welch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file", nargs="+", type=Path)
    a = p.parse_args()
    for file in a.file:
        stream = read(str(file))
        gaps = stream.get_gaps()
        stream.merge(method=0)
        if len(stream) != 1:
            raise ValueError("expected one station/location/channel")
        trace = stream[0]
        fs = trace.stats.sampling_rate
        start = trace.stats.starttime
        print(trace.id, start, trace.stats.endtime, "Hz", fs, "gaps/overlaps", len(gaps))
        peaks = []
        for h in range(24):
            day = start.__class__(start.date.isoformat())
            x = trace.slice(day + h * 3600, day + (h + 1) * 3600 - 1 / fs).data
            if len(x) != round(fs * 3600) or np.ma.count_masked(x):
                continue
            f, power = welch(x.astype(float), fs=fs, nperseg=round(fs * 256), detrend="linear")
            band = (f >= 0.1) & (f <= 0.3)
            peaks.append(f[band][np.argmax(power[band])])
        print(
            "complete hours",
            len(peaks),
            "median secondary peak Hz",
            np.median(peaks) if peaks else None,
        )


if __name__ == "__main__":
    main()
