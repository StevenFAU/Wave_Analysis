"""Assemble the dashboard: live data directory and the static site.

``build_live`` runs on the collector host (hourly) and writes::

    <out>/status.json              archive and collector status
    <out>/seastate/<station>.json  recent realtime observations per station
    <out>/manifest.json            file list with sizes and SHA-256

``build_site`` runs in CI and writes the deployable site::

    <out>/index.html, assets/...   copied from dashboard/
    <out>/data/catalog.json        from :func:`catalog.build_catalog`
    <out>/data/live/...            bundled copy of the live data (fallback)
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from wave_analysis.dashboard import seastate
from wave_analysis.dashboard.catalog import (
    SCHEMA as CATALOG_SCHEMA,
)
from wave_analysis.dashboard.catalog import (
    build_catalog,
    docs_url,
    load_camera_sites,
    write_json,
)
from wave_analysis.dashboard.status import SCHEMA as STATUS_SCHEMA
from wave_analysis.dashboard.status import StatusInputs, build_status, latest_listing

LIVE_SCHEMA = "wave-analysis/dashboard-live@1"


def seastate_station_ids(root: Path, buoycam_root: Path) -> list[str]:
    """Camera stations in the newest listing plus every camera-site reference buoy."""
    _, listing = latest_listing(buoycam_root)
    ids = set(listing["station_id"].astype(str)) if not listing.empty else set()
    for site in load_camera_sites(root / "data" / "registry" / "camera_sites.yaml"):
        for ref in site.references:
            if ref.network in {"NDBC", "CDIP"}:
                ids.add(ref.station_id.upper())
    return sorted(ids)


def _file_manifest(out: Path) -> list[dict[str, Any]]:
    files = []
    for p in sorted(out.rglob("*.json")):
        if p.name == "manifest.json":
            continue
        data = p.read_bytes()
        files.append(
            {
                "path": p.relative_to(out).as_posix(),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return files


def build_live(
    root: Path,
    inputs: StatusInputs,
    out: Path,
    *,
    cache_dir: Path,
    fetch: seastate.Fetcher | None,
    now: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Refresh the sea-state cache (when ``fetch`` is given) and write live data to ``out``."""
    now = now or pd.Timestamp.now(tz="UTC")
    ids = seastate_station_ids(root, inputs.buoycam_root)
    refresh: dict[str, Any] | None = None
    if fetch is not None:
        refresh = asdict(seastate.refresh(ids, cache_dir, fetch, now=now))
    caches = {p.name: seastate.load_cache(cache_dir, p) for p in seastate.PRODUCTS}
    latest = seastate.latest_values(caches, now=now)

    status = build_status(inputs, now=now, latest_obs=latest)
    status["seastate"] = {
        "source": "NDBC realtime2 (.txt, .spec), provisional and not quality-controlled",
        "hours": seastate.PUBLISH_HOURS,
        "stations": [],
        "refresh": (
            {k: v for k, v in refresh.items() if k != "failures"}
            | {"failures": len(refresh["failures"])}
            if refresh
            else None
        ),
    }
    if out.exists():
        shutil.rmtree(out)
    (out / "seastate").mkdir(parents=True)
    for sid in ids:
        payload = seastate.station_payload(sid, caches, now=now)
        if payload is None:
            continue
        payload["generated_at"] = status["generated_at"]
        write_json(payload, out / "seastate" / f"{sid}.json")
        status["seastate"]["stations"].append(sid)
    write_json(status, out / "status.json")
    manifest = {
        "schema": LIVE_SCHEMA,
        "generated_at": status["generated_at"],
        "files": _file_manifest(out),
    }
    write_json(manifest, out / "manifest.json")
    return {
        "stations": len(status["stations"]),
        "seastate_files": len(status["seastate"]["stations"]),
        "refresh": refresh,
    }


def build_site(root: Path, out: Path, *, live_dir: Path | None = None) -> dict[str, Any]:
    """Copy the front end, write the catalog, and bundle live data if available."""
    src = root / "dashboard"
    if not (src / "index.html").exists():
        raise FileNotFoundError(f"{src}/index.html not found")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        src, out, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.test.*", "package.json")
    )
    catalog = build_catalog(root)
    write_json(catalog, out / "data" / "catalog.json")
    bundled = False
    if live_dir is not None and (live_dir / "status.json").exists():
        shutil.copytree(live_dir, out / "data" / "live", dirs_exist_ok=True)
        bundled = True
    (out / ".nojekyll").touch()
    return {
        "datasets": len(catalog["datasets"]),
        "references": len(catalog["references"]),
        "sites": len(catalog["sites"]),
        "live_bundled": bundled,
    }


_ASSET_REF = re.compile(r"""(?:href|src)=["']([^"'#?]+)["']""")
_JS_IMPORT = re.compile(
    r"""(?:^|\s)(?:import|export)\s[^;]*?from\s+["'](\.{1,2}/[^"']+)["']""", re.S
)


def _strict_json(path: Path) -> Any:
    def reject(token: str) -> Any:
        raise ValueError(f"non-standard JSON constant {token}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def check_site(site: Path, *, require_docs: bool = False) -> list[str]:
    """Problems in a built site (empty list = OK).

    Checks: strict JSON and schema of the data files; the bundled live data
    against its manifest (size and SHA-256); local assets referenced by
    ``index.html`` and relative ES-module imports; and, with ``require_docs``,
    that every documentation page the catalog links to was built.
    """
    problems: list[str] = []
    index = site / "index.html"
    if not index.exists():
        return [f"{index} missing"]
    for ref in _ASSET_REF.findall(index.read_text(encoding="utf-8")):
        if ref.startswith(("http:", "https:", "mailto:", "data:")) or ref.endswith("/"):
            continue
        if not (site / ref).exists():
            problems.append(f"index.html references missing file {ref}")
    for js in sorted((site / "assets" / "js").rglob("*.js")):
        for spec in _JS_IMPORT.findall(js.read_text(encoding="utf-8")):
            if not (js.parent / spec).resolve().exists():
                problems.append(f"{js.relative_to(site)} imports missing module {spec}")

    try:
        catalog = _strict_json(site / "data" / "catalog.json")
    except (OSError, ValueError) as exc:
        return [*problems, f"data/catalog.json: {exc}"]
    if catalog.get("schema") != CATALOG_SCHEMA:
        problems.append(f"catalog schema {catalog.get('schema')!r} != {CATALOG_SCHEMA!r}")

    live = site / "data" / "live"
    if (live / "status.json").exists():
        try:
            status = _strict_json(live / "status.json")
            if status.get("schema") != STATUS_SCHEMA:
                problems.append(f"status schema {status.get('schema')!r} != {STATUS_SCHEMA!r}")
            for sid in status.get("seastate", {}).get("stations", []):
                if not (live / "seastate" / f"{sid}.json").exists():
                    problems.append(f"status lists sea state for {sid} but the file is missing")
            manifest = _strict_json(live / "manifest.json")
            for f in manifest.get("files", []):
                data = (live / f["path"]).read_bytes()
                if len(data) != f["bytes"] or hashlib.sha256(data).hexdigest() != f["sha256"]:
                    problems.append(f"data/live/{f['path']} does not match its manifest entry")
        except (OSError, ValueError, KeyError) as exc:
            problems.append(f"data/live: {exc}")

    if require_docs:
        urls: set[str] = set()

        def walk(items: list[dict[str, Any]]) -> None:
            for it in items:
                if "url" in it and not it.get("external"):
                    urls.add(it["url"])
                walk(it.get("children", []))

        walk(catalog["docs"]["nav"])
        urls.update(docs_url(p) for p in catalog["docs"]["titles"])
        for ref in catalog["references"]:
            urls.update(docs_url(p) for p in ref["mentions"])
            if ref["notes"]:
                urls.add(docs_url(ref["notes"]))
        urls.update(d["doc_url"] for d in catalog["datasets"] if d["doc_url"])
        for url in sorted(urls):
            path = url.split("#", 1)[0]
            target = site / path / "index.html" if path.endswith("/") else site / path
            if not target.exists():
                problems.append(f"catalog links to {url}, which the docs build did not produce")
    return problems
