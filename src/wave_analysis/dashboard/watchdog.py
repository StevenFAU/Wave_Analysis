"""Collector watchdog: alert when the primary collector host goes quiet.

A host cannot report that it is switched off, so this check runs somewhere
else: the scheduled workflow ``.github/workflows/collector-watchdog.yml``
reads the ``status.json`` that the primary host publishes hourly to the
``dashboard-data`` branch (:mod:`.status`) and opens a GitHub issue when any
of these is older than its limit:

=============== ========================== =====================================
Check           Field                      Limit
=============== ========================== =====================================
``collector``   ``collector.last_run``     :data:`MAX_SILENCE_H` (last archiver run)
``publisher``   ``generated_at``           :data:`MAX_SILENCE_H` (last status push)
``offsite``     ``offsite.last_sync``      :data:`MAX_OFFSITE_H` (daily sync)
=============== ========================== =====================================

Why 3 h: the archiver runs hourly, so 3 h means at least two missed runs, and
an alert then leaves more than two days of the ~70 h backfill window to bring
a host back (P0 in ``docs/operations/collection_plan.md``). The report states
when buoy-camera images start to be lost for good: the first image not yet
archived is due one hour after ``archive.last_image`` and stays within the
backfill window for :data:`BACKFILL_HOURS` more.

Only the standard library is used, so the workflow runs it from a checkout
without installing the package::

    PYTHONPATH=src python -m wave_analysis.dashboard.watchdog status.json --report report.md

Exit status: 0 healthy, 1 alert (the issue title is printed), 2 unreadable status.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Hours without an archiver run or a status push before alerting.
MAX_SILENCE_H = 3.0
#: Hours without an offsite sync before alerting (the sync runs daily).
MAX_OFFSITE_H = 30.0
#: Mirrors ``buoycam.DEFAULT_BACKFILL_HOURS`` (not imported: that module needs pandas).
BACKFILL_HOURS = 70
#: Repository for links in the report (GitHub Actions sets ``GITHUB_REPOSITORY``).
DEFAULT_REPOSITORY = "StevenFAU/Wave_Analysis"

_CHECKS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("collector", ("collector", "last_run"), "last buoy-camera archiver run"),
    ("publisher", ("generated_at",), "last dashboard status push"),
    ("offsite", ("offsite", "last_sync"), "last offsite sync"),
)


@dataclass(frozen=True)
class Check:
    """One freshness check of the published status."""

    name: str
    label: str
    last: dt.datetime | None
    limit_h: float
    now: dt.datetime

    @property
    def age_h(self) -> float | None:
        """Hours since the timestamp, or ``None`` if it is missing."""
        if self.last is None:
            return None
        return (self.now - self.last).total_seconds() / 3600

    @property
    def ok(self) -> bool:
        """True if the timestamp exists and is within the limit."""
        age = self.age_h
        return age is not None and age <= self.limit_h


def _parse_time(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        t = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.UTC)


def _field(status: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = status
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def check_status(
    status: dict[str, Any],
    now: dt.datetime,
    *,
    max_silence_h: float = MAX_SILENCE_H,
    max_offsite_h: float = MAX_OFFSITE_H,
) -> list[Check]:
    """Freshness of each published timestamp as of ``now`` (a missing one fails)."""
    limits = {"collector": max_silence_h, "publisher": max_silence_h, "offsite": max_offsite_h}
    return [
        Check(name, label, _parse_time(_field(status, path)), limits[name], now)
        for name, path, label in _CHECKS
    ]


def loss_deadline(status: dict[str, Any]) -> dt.datetime | None:
    """When the first image not yet archived leaves the backfill window."""
    last = _parse_time(_field(status, ("archive", "last_image")))
    if last is None:
        return None
    return last + dt.timedelta(hours=1 + BACKFILL_HOURS)


def _fmt(t: dt.datetime | None) -> str:
    return "never" if t is None else t.astimezone(dt.UTC).strftime("%Y-%m-%d %H:%M UTC")


def alert_title(checks: Sequence[Check]) -> str:
    """Issue title naming the failed checks and the oldest of their timestamps."""
    failed = [c for c in checks if not c.ok]
    names = ", ".join(c.name for c in failed)
    known = [c.last for c in failed if c.last is not None]
    since = f" since {_fmt(min(known))}" if known else ""
    return f"Collector watchdog: {names} silent{since}"


def render_report(
    status: dict[str, Any],
    checks: Sequence[Check],
    now: dt.datetime,
    repository: str = DEFAULT_REPOSITORY,
) -> str:
    """Markdown body for the alert issue and the workflow summary."""
    healthy = all(c.ok for c in checks)
    lines = [
        f"**{'Healthy' if healthy else 'Alert'}** as of {_fmt(now)}.",
        "",
        "| Check | What | Last | Age (h) | Limit (h) | OK |",
        "|---|---|---|---|---|---|",
    ]
    for c in checks:
        age = "n/a" if c.age_h is None else f"{c.age_h:.1f}"
        lines.append(
            f"| `{c.name}` | {c.label} | {_fmt(c.last)} | {age} | {c.limit_h:g} | "
            f"{'yes' if c.ok else '**no**'} |"
        )
    deadline = loss_deadline(status)
    if not healthy and deadline is not None:
        newest = _parse_time(_field(status, ("archive", "last_image")))
        left = (deadline - now).total_seconds() / 3600
        lines += [
            "",
            f"Newest archived buoy-camera image: {_fmt(newest)}. Images not yet archived "
            f"start to be lost for good at about **{_fmt(deadline)}** ({left:.0f} h from now) "
            "unless the archiver runs on either host before then.",
        ]
    if not healthy:
        docs = f"https://github.com/{repository}/blob/main/docs/operations/data_collection.md"
        lines += [
            "",
            "Only the primary host publishes this status, so a second collector host may "
            f"still be archiving. To recover, see [data collection]({docs}#running-it); "
            "afterwards check `uv run python scripts/buoycam_coverage.py --days 4`.",
            "",
            "This issue is updated hourly and closed when the host reports again.",
        ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Check a status file; print ``healthy`` or the alert title (see module notes)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("status", type=Path, help="status.json from the dashboard-data branch")
    ap.add_argument("--report", type=Path, help="write the Markdown report here")
    ap.add_argument("--now", help="ISO time to check against (default: current UTC time)")
    args = ap.parse_args(argv)
    try:
        status = json.loads(args.status.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"cannot read {args.status}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(status, dict):
        print(f"{args.status} is not a JSON object", file=sys.stderr)
        return 2
    now = _parse_time(args.now) if args.now else dt.datetime.now(dt.UTC)
    if now is None:
        print(f"bad --now: {args.now!r}", file=sys.stderr)
        return 2
    checks = check_status(status, now)
    report = render_report(
        status, checks, now, os.environ.get("GITHUB_REPOSITORY") or DEFAULT_REPOSITORY
    )
    if args.report:
        args.report.write_text(report, encoding="utf-8")
    if all(c.ok for c in checks):
        print("healthy")
        return 0
    print(alert_title(checks))
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
