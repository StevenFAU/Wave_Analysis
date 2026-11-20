"""Collect NDBC station water depths with the station page and retrieval date."""

import argparse
import re
import urllib.request
from datetime import UTC, datetime
from html import unescape
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    rows = []
    for sid in sorted(pd.read_csv(a.labels, usecols=["station_id"], dtype=str).station_id.unique()):
        url = "https://www.ndbc.noaa.gov/station_page.php?station=" + sid
        row = {
            "station_id": sid,
            "source": url,
            "retrieved_utc": datetime.now(UTC).isoformat(),
        }
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                text = unescape(re.sub("<[^>]*>", " ", response.read().decode(errors="replace")))
            m = re.search(r"Water depth:\s*([\d,.]+)\s*m\b", text, re.I)
            if m:
                row["depth_m"] = float(m[1].replace(",", ""))
        except OSError as exc:
            row["error"] = str(exc)
        rows.append(row)
    pd.DataFrame(rows).to_csv(a.out, index=False, mode="x")


if __name__ == "__main__":
    main()
