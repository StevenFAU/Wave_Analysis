"""Static dashboard catalog: datasets, camera sites, stations, references, docs.

Everything here is derived from files committed to the repository, so the
catalog is reproducible from a commit. The sources of truth stay where they
are:

* ``data/registry/datasets.yaml``: dataset registry (validated by
  :mod:`wave_analysis.registry`);
* ``data/registry/camera_sites.yaml``: camera sites and their wave references;
* ``data/registry/stations.parquet`` and ``ndbc_station_summary.csv``: NDBC
  station metadata and per-product archive years;
* ``data/registry/data_dictionary.yaml``: what each collection holds and what
  its variables mean (validated by :mod:`wave_analysis.data_dictionary`);
* ``docs/literature/bibliography.bib`` and the *Verification index* table in
  ``docs/literature/source_verification.md``;
* the Markdown under ``docs/`` (titles, citation mentions, navigation).

:func:`build_catalog` fails loudly on dangling references (a verification row
for a key not in the bibliography, a site referencing an unknown station or
dataset, a dictionary entry naming an unknown dataset or doc) so that CI
catches them.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from wave_analysis import __version__
from wave_analysis.config import load_yaml
from wave_analysis.dashboard.bibtex import BibEntry, ascii_fold, delatex, parse_bibtex
from wave_analysis.data_dictionary import DataDictionary, check_references, load_data_dictionary
from wave_analysis.processing.geospatial import haversine_m
from wave_analysis.registry import DatasetEntry, validate_registry

SCHEMA = "wave-analysis/dashboard-catalog@1"
DEFAULT_REPOSITORY = "StevenFAU/Wave_Analysis"
LIVE_DATA_BRANCH = "dashboard-data"
HISTORY_PRODUCTS = ("stdmet", "swden", "swdir", "swdir2", "swr1", "swr2", "adcp")
_CDIP_IN_NAME = re.compile(r"\((\d{3})\)\s*$")


# --------------------------------------------------------------------------- #
# Camera sites
# --------------------------------------------------------------------------- #


class SiteReference(BaseModel):
    """A wave instrument proposed as the reference for a camera site."""

    model_config = ConfigDict(extra="forbid")

    station_id: str
    network: Literal["CDIP", "NDBC", "USGS", "other"]
    role: Literal["primary", "secondary", "offshore"]
    cdip_id: str | None = None
    # Only for instruments absent from the NDBC station table:
    name: str | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    coordinate_source: str | None = None
    period: str | None = None


class CameraSite(BaseModel):
    """A public camera site (see ``data/registry/camera_sites.yaml``)."""

    model_config = ConfigDict(extra="forbid")

    site_id: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str
    provider: str
    dataset_id: str | None = None
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    approximate: bool = False
    coordinate_source: str
    period: str
    references: list[SiteReference]


def load_camera_sites(path: Path) -> list[CameraSite]:
    """Load and validate the camera-site registry."""
    raw = load_yaml(path)
    sites = [CameraSite.model_validate(s) for s in raw.get("sites", [])]
    ids = [s.site_id for s in sites]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate site_id in {path}: {dupes}")
    return sites


# --------------------------------------------------------------------------- #
# Docs
# --------------------------------------------------------------------------- #


def docs_url(path: str) -> str:
    """Site-relative URL of a Markdown file under ``docs/`` in the MkDocs build.

    ``docs/methodology/models.md`` -> ``docs/methodology/models/``;
    ``index.md`` and ``README.md`` map to their directory (MkDocs treats a
    ``README.md`` as the directory index). Anchors are kept.
    """
    path, _, anchor = path.partition("#")
    rel = path.removeprefix("docs/")
    if not rel.endswith(".md"):
        url = f"docs/{rel}"
    else:
        stem = rel[: -len(".md")]
        parent, _, name = stem.rpartition("/")
        if name in {"index", "README"}:
            url = f"docs/{parent}/" if parent else "docs/"
        else:
            url = f"docs/{stem}/"
    return f"{url}#{anchor}" if anchor else url


def read_docs(docs_root: Path) -> dict[str, str]:
    """All Markdown under ``docs_root``, keyed by repository-relative path."""
    out: dict[str, str] = {}
    for p in sorted(docs_root.rglob("*.md")):
        out[f"docs/{p.relative_to(docs_root).as_posix()}"] = p.read_text(encoding="utf-8")
    return out


def doc_title(text: str, fallback: str) -> str:
    """First level-1 heading of a Markdown document."""
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def mkdocs_nav(mkdocs_yml: Path) -> list[dict[str, Any]]:
    """The MkDocs navigation tree as ``[{title, url} | {title, children}]``."""
    # mkdocs.yml may carry !!python tags in other projects; this one does not.
    cfg = yaml.safe_load(mkdocs_yml.read_text(encoding="utf-8"))

    def walk(items: Iterable[Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, str):
                out.append({"title": item, "url": docs_url(item)})
                continue
            for title, target in item.items():
                if isinstance(target, list):
                    out.append({"title": title, "children": walk(target)})
                elif re.match(r"^[a-z]+://", str(target)):
                    out.append({"title": title, "url": str(target), "external": True})
                else:
                    out.append({"title": title, "url": docs_url(str(target)), "path": target})
        return out

    return walk(cfg.get("nav", []))


# --------------------------------------------------------------------------- #
# References
# --------------------------------------------------------------------------- #


def verification_index(markdown: str) -> dict[str, dict[str, str]]:
    """Parse the *Verification index* table of ``source_verification.md``."""
    m = re.search(r"^## Verification index\s*$", markdown, flags=re.M)
    if m is None:
        return {}
    out: dict[str, dict[str, str]] = {}
    for line in markdown[m.end() :].splitlines():
        if line.startswith("## "):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 3 or not line.lstrip().startswith("|"):
            continue
        key, level, basis = cells
        if key in {"Key", ""} or set(key) <= {"-", ":"}:
            continue
        if key in out:
            raise ValueError(f"verification index lists {key!r} twice")
        out[key] = {"level": level, "basis": basis}
    return out


def _mention_patterns(entry: BibEntry) -> list[re.Pattern[str]]:
    pats = [re.compile(rf"\b{re.escape(entry.key)}\b")]
    surname = entry.first_author_surname
    year = entry.year
    if surname and year:
        name = re.escape(ascii_fold(surname))
        # "Quach et al. 2021", "Choi et al. (2020)", "Kim, Cho & Lee (2023)":
        # surname, then "et al." or up to 40 characters without sentence or
        # table punctuation, then the year.
        pats.append(
            re.compile(rf"\b{name}\b(?:\s+et\s+al\.?|[^.;\n|()]{{0,40}}?)[\s,]*\(?{year}[a-z]?\b")
        )
    return pats


def find_mentions(entries: Iterable[BibEntry], docs: Mapping[str, str]) -> dict[str, list[str]]:
    """Docs that cite each entry by key or by first-author surname and year.

    Author-year matching is deliberately loose (it can match two papers by the
    same first author in the same year); the dashboard labels it "mentioned in".
    """
    folded = {path: ascii_fold(text) for path, text in docs.items()}
    out: dict[str, list[str]] = {}
    for e in entries:
        pats = _mention_patterns(e)
        out[e.key] = [p for p, text in folded.items() if any(r.search(text) for r in pats)]
    return out


def screened_items(landscape_md: str) -> list[dict[str, Any]]:
    """Bullets under "Screened only (S)" in the landscape review (not evidence yet)."""
    m = re.search(r"^\*\*Screened only \(S\)[^\n]*$", landscape_md, flags=re.M)
    if m is None:
        return []
    items: list[dict[str, Any]] = []
    for line in landscape_md[m.end() :].lstrip("\n").splitlines():
        if not line.startswith("- "):
            break
        text = line[2:].strip()
        items.append(
            {
                "text": text,
                "dois": re.findall(r"doi:(10\.\S+?)(?=[\s,;)]|$)", text),
                "arxiv": re.findall(r"arXiv:(\d{4}\.\d{4,5})", text),
            }
        )
    return items


def _venue(f: Mapping[str, str]) -> str | None:
    if "journal" in f:
        return delatex(f["journal"])
    if "booktitle" in f:
        return delatex(f["booktitle"])
    if f.get("archiveprefix", "").lower() == "arxiv" and "eprint" in f:
        return f"arXiv:{f['eprint']}"
    for k in ("publisher", "institution", "howpublished"):
        if k in f:
            return delatex(f[k])
    return None


def _author_short(entry: BibEntry) -> str:
    surname = entry.first_author_surname or "Anon."
    n = len(entry.authors)
    if n == 2:
        second = entry.authors[1]
        s2 = second.split(",")[0] if "," in second else second.split()[-1]
        return f"{surname} & {s2.strip()}"
    return f"{surname} et al." if n > 2 else surname


def reference_records(
    entries: list[BibEntry],
    levels: Mapping[str, Mapping[str, str]],
    docs: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Display records for every bibliography entry."""
    mentions = find_mentions(entries, docs)
    note_pages = {p.rsplit("/", 1)[-1][:-3]: p for p in docs if "/literature/papers/" in p}
    out = []
    for e in entries:
        f = e.fields
        doi = f.get("doi")
        arxiv = f.get("eprint") if f.get("archiveprefix", "").lower() == "arxiv" else None
        url = (
            f"https://doi.org/{doi}"
            if doi
            else f.get("url") or (f"https://arxiv.org/abs/{arxiv}" if arxiv else None)
        )
        notes = note_pages.get(e.key.lower())
        out.append(
            {
                "key": e.key,
                "type": e.entry_type,
                "section": e.section,
                "title": delatex(f.get("title", "")),
                "authors": e.authors,
                "author_short": _author_short(e),
                "year": e.year,
                "venue": _venue(f),
                "volume": f.get("volume"),
                "pages": delatex(f["pages"]) if "pages" in f else None,
                "doi": doi,
                "arxiv": arxiv,
                "url": url,
                "verification": dict(levels[e.key]) if e.key in levels else None,
                "mentions": [p for p in mentions[e.key] if p != notes],
                "notes": notes,
                "bibtex": e.raw,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Stations
# --------------------------------------------------------------------------- #


def _clean(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float | np.floating):
        return None if math.isnan(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def station_records(
    stations: pd.DataFrame, summary: pd.DataFrame, ids: Iterable[str]
) -> dict[str, dict[str, Any]]:
    """Metadata and NDBC archive history for the given station ids."""
    st = stations.assign(station_id=stations["station_id"].astype(str).str.upper())
    st = st.drop_duplicates("station_id").set_index("station_id")
    sm = summary.assign(station_id=summary["station_id"].astype(str).str.upper())
    sm = sm.drop_duplicates("station_id").set_index("station_id")
    out: dict[str, dict[str, Any]] = {}
    for sid in sorted({str(i).upper() for i in ids}):
        if sid not in st.index:
            continue
        r = st.loc[sid]
        name = _clean(r.get("name"))
        cdip = _CDIP_IN_NAME.search(name) if isinstance(name, str) else None
        rec: dict[str, Any] = {
            "name": name,
            "lat": _clean(r.get("latitude")),
            "lon": _clean(r.get("longitude")),
            "owner": _clean(r.get("owner")),
            "program": _clean(r.get("program")),
            "platform_type": _clean(r.get("platform_type")),
            "hull": _clean(r.get("hull")),
            "payload": _clean(r.get("payload")),
            "cdip_id": cdip.group(1) if cdip else None,
            "history": {},
            "realtime_products": [],
        }
        if sid in sm.index:
            s = sm.loc[sid]
            for prod in HISTORY_PRODUCTS:
                years = _clean(s.get(f"{prod}_years"))
                if years:
                    rec["history"][prod] = {
                        "first": int(_clean(s.get(f"{prod}_first"))),
                        "last": int(_clean(s.get(f"{prod}_last"))),
                        "years": int(years),
                    }
            rt = _clean(s.get("realtime_products"))
            if isinstance(rt, str) and rt:
                rec["realtime_products"] = sorted(rt.split(","))
        out[sid] = rec
    return out


def site_records(
    sites: list[CameraSite], stations: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Camera sites with each reference's position and distance (km) resolved."""
    out = []
    for s in sites:
        refs = []
        for ref in s.references:
            meta = stations.get(ref.station_id.upper(), {})
            lat = ref.latitude if ref.latitude is not None else meta.get("lat")
            lon = ref.longitude if ref.longitude is not None else meta.get("lon")
            if lat is None or lon is None:
                raise ValueError(
                    f"{s.site_id}: reference {ref.station_id} is not in the station table "
                    "and has no explicit coordinates"
                )
            km = float(haversine_m(s.latitude, s.longitude, lat, lon)) / 1000.0
            refs.append(
                {
                    **ref.model_dump(exclude_none=True),
                    "name": ref.name or meta.get("name"),
                    "latitude": lat,
                    "longitude": lon,
                    "distance_km": round(km, 1),
                }
            )
        out.append({**s.model_dump(exclude={"references"}), "references": refs})
    return out


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


def dataset_records(registry: Mapping[str, Any], sites: list[CameraSite]) -> list[dict[str, Any]]:
    """Validated registry entries, with doc URLs and the camera sites that use them."""
    problems = validate_registry(dict(registry))
    if problems:
        raise ValueError("dataset registry is invalid:\n  " + "\n  ".join(problems))
    by_dataset: dict[str, list[str]] = {}
    for s in sites:
        if s.dataset_id:
            by_dataset.setdefault(s.dataset_id, []).append(s.site_id)
    out = []
    for raw in registry["datasets"]:
        e = DatasetEntry.model_validate(raw)
        rec = e.model_dump(mode="json")
        rec["doc_url"] = docs_url(e.doc) if e.doc else None
        rec["sites"] = by_dataset.get(e.dataset_id, [])
        out.append(rec)
    return out


def dictionary_record(dd: DataDictionary) -> dict[str, Any]:
    """The data dictionary as JSON, with each documentation path's site URL."""
    rec = dd.model_dump(mode="json")
    for item in [*rec["collections"], *rec["tables"]]:
        item["doc_url"] = docs_url(item["doc"])
    return rec


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def _git_commit(root: Path) -> str | None:
    env = os.environ.get("GITHUB_SHA")
    if env:
        return env
    try:
        res = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return res.stdout.strip() or None


def _latest_camera_registry(registry_dir: Path) -> Path | None:
    snaps = sorted(registry_dir.glob("ndbc_cameras_*.csv"))
    return snaps[-1] if snaps else None


def build_catalog(root: Path, *, now: dt.datetime | None = None) -> dict[str, Any]:
    """Assemble the catalog from the repository at ``root``."""
    reg_dir = root / "data" / "registry"
    lit = root / "docs" / "literature"
    docs = read_docs(root / "docs")

    sites = load_camera_sites(reg_dir / "camera_sites.yaml")
    registry = load_yaml(reg_dir / "datasets.yaml")
    datasets = dataset_records(registry, sites)
    known_datasets = {d["dataset_id"] for d in datasets}
    for s in sites:
        if s.dataset_id and s.dataset_id not in known_datasets:
            raise ValueError(f"{s.site_id}: unknown dataset_id {s.dataset_id!r}")
    dictionary = load_data_dictionary(reg_dir / "data_dictionary.yaml")
    problems = check_references(dictionary, datasets=known_datasets, docs=docs)
    if problems:
        raise ValueError("data dictionary is invalid:\n  " + "\n  ".join(problems))

    stations_df = pd.read_parquet(reg_dir / "stations.parquet")
    summary_df = pd.read_csv(reg_dir / "ndbc_station_summary.csv", dtype={"station_id": str})
    cam_path = _latest_camera_registry(reg_dir)
    camera_ids: list[str] = []
    if cam_path is not None:
        camera_ids = (
            pd.read_csv(cam_path, dtype={"station_id": str})["station_id"].str.upper().tolist()
        )
    ref_ids = [r.station_id for s in sites for r in s.references]
    stations = station_records(stations_df, summary_df, [*camera_ids, *ref_ids])
    site_recs = site_records(sites, stations)

    entries = parse_bibtex((lit / "bibliography.bib").read_text(encoding="utf-8"))
    levels = verification_index((lit / "source_verification.md").read_text(encoding="utf-8"))
    keys = {e.key for e in entries}
    unknown = sorted(set(levels) - keys)
    if unknown:
        raise ValueError(f"verification index lists keys not in bibliography.bib: {unknown}")
    references = reference_records(entries, levels, docs)
    landscape = (lit / "landscape_review.md").read_text(encoding="utf-8")

    now = now or dt.datetime.now(dt.UTC)
    repo = os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPOSITORY)
    return {
        "schema": SCHEMA,
        "build": {
            "built_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "commit": _git_commit(root),
            "software_version": __version__,
            "repository": repo,
            "live_data_url": f"https://raw.githubusercontent.com/{repo}/{LIVE_DATA_BRANCH}/",
            "camera_registry": cam_path.name if cam_path else None,
        },
        "registry_version": registry.get("registry_version"),
        "datasets": datasets,
        "dictionary": dictionary_record(dictionary),
        "sites": site_recs,
        "stations": stations,
        "references": references,
        "screened": screened_items(landscape),
        "docs": {
            "nav": mkdocs_nav(root / "mkdocs.yml"),
            "titles": {p: doc_title(t, p) for p, t in docs.items()},
        },
    }


def to_json(obj: Any) -> str:
    """Strict JSON (NaN/inf rejected) with compact separators."""
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def write_json(obj: Any, path: Path) -> Path:
    """Write ``obj`` as strict JSON, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(to_json(obj), encoding="utf-8")
    tmp.replace(path)
    return path
