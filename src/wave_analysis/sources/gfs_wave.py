"""Read the first six forecast hours plus the next boundary from WW3 point bulletins."""

import re
import tarfile
from collections.abc import Collection
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def short_leads(text: str) -> list[dict[str, Any]]:
    """Parse valid forecast hours 0 through 6 for one issue and station."""
    issue = re.search(r"Cycle\s*:\s*(\d{8})\s+(\d+) UTC", text)
    station = re.search(r"Location\s*:\s*(\S+)", text)
    if not issue or not station:
        raise ValueError("missing issue or location in station bulletin")
    start = datetime.strptime(issue[1], "%Y%m%d").replace(tzinfo=UTC, hour=int(issue[2]))
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 8 or not re.fullmatch(r"\d+\s+\d+", cells[0]):
            continue
        day, hour = map(int, cells[0].split())
        valid = start + timedelta(hours=len(rows))
        if (day, hour) != (valid.day, valid.hour):
            raise ValueError(f"missing/unexpected hour at {station[1]}, expected {valid}")
        total = re.fullmatch(r"\s*(?:(\d+\.\d+)\s+)?(\d+)\s*(\d*)\s*", cells[1])
        if total is None:
            raise ValueError(f"invalid WW3 total/count: {cells[1]}")

        row: dict[str, Any] = {
            "station_id": station[1],
            "issue_utc": start,
            "valid_utc": valid,
            "lead_h": len(rows),
            "hs_m": float(total[1]) if total[1] else None,
            "systems": int(total[2]),
            "extra_systems": int(total[3] or 0),
        }
        for i, cell in enumerate(cells[2:], 1):
            parts = cell.replace("*", " ").split()
            if parts and len(parts) != 3:
                raise ValueError(f"unexpected system in {station[1]}: {cell}")
            for col, value in zip(
                ("hs_m", "tp_s", "direction_deg"), map(float, parts), strict=bool(parts)
            ):
                row[f"system{i}_{col}"] = value
            row[f"system{i}_present"] = bool(parts)
            row[f"system{i}_wind"] = "*" in cell
        rows.append(row)
        if len(rows) == 7:
            break
    if len(rows) != 7:
        raise ValueError(f"incomplete short leads for {station[1]}")
    return rows


def read_cycle(path: str | Path, stations: Collection[str]) -> list[dict[str, Any]]:
    """Read selected archive members in memory without extracting external paths."""
    rows: list[dict[str, Any]] = []
    with tarfile.open(path) as tar:
        for member in tar:
            name = Path(member.name).name.split(".")
            if (
                member.isfile()
                and len(name) == 3
                and name[0] == "gfswave"
                and name[1] in stations
                and name[2] == "bull"
            ):
                stream = tar.extractfile(member)
                if stream is None:
                    raise ValueError(member.name)
                with stream:
                    part = short_leads(stream.read().decode("ascii"))
                if part[0]["station_id"] != name[1]:
                    raise ValueError(f"location mismatch in {member.name}")
                rows.extend(part)
    return rows
