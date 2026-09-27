"""Provenance records for derived artefacts, and RO-Crate packaging.

Every derived artefact (standardized table, synchronised dataset, ML manifest,
model result) is accompanied by a ``*.provenance.yaml`` answering the questions
of the repository specification §13: what sources, which version, when
downloaded, which script, which configuration, which Git commit, what changed.

For publishable releases :func:`write_ro_crate` produces a minimal but valid
RO-Crate 1.1 ``ro-crate-metadata.json`` (https://w3id.org/ro/crate/1.1) listing
the data entities, their checksums, and the creating action.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

from wave_analysis import __version__
from wave_analysis.config import portable_path
from wave_analysis.ingest.checksum import sha256_file
from wave_analysis.ingest.manifest import utcnow

_TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "pyarrow", "pydantic", "httpx", "torch", "xarray")


def git_state(repo: Path | None = None) -> dict[str, Any]:
    """Current commit, branch and dirty flag (``None`` values outside a Git checkout).

    ``dirty`` means a tracked file differs from the commit; untracked files
    (local tool settings, fresh outputs) do not count.
    """

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True, check=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def software_environment() -> dict[str, Any]:
    """Python, platform, and versions of key scientific packages."""
    pkgs: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            pkgs[name] = version(name)
        except PackageNotFoundError:
            pkgs[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "wave_analysis": __version__,
        "packages": pkgs,
    }


def config_hash(config: Mapping[str, Any]) -> str:
    """SHA-256 of a canonical JSON serialisation of a configuration mapping."""
    blob = json.dumps(config, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def build_provenance(
    *,
    artifact_id: str,
    inputs: Iterable[str | Path],
    outputs: Iterable[str | Path],
    command: str | None = None,
    config: Mapping[str, Any] | None = None,
    upstream_manifests: Iterable[str | Path] = (),
    notes: str | None = None,
    repo: Path | None = None,
) -> dict[str, Any]:
    """Assemble a provenance record (checksums every input and output file)."""
    return {
        "artifact_id": artifact_id,
        "created_at": utcnow().isoformat().replace("+00:00", "Z"),
        "git": git_state(repo),
        "pipeline": {
            "command": command or " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
            "config": dict(config) if config else None,
            "config_sha256": config_hash(config) if config else None,
        },
        "software": software_environment(),
        "input_files": [{"path": portable_path(p), "sha256": sha256_file(p)} for p in inputs],
        "upstream_manifests": [portable_path(p) for p in upstream_manifests],
        "output_files": [{"path": portable_path(p), "sha256": sha256_file(p)} for p in outputs],
        "notes": notes,
    }


def write_provenance(record: Mapping[str, Any], path: str | Path) -> Path:
    """Write a provenance record as YAML."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        yaml.safe_dump(dict(record), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return p


def write_ro_crate(
    crate_dir: str | Path,
    *,
    name: str,
    description: str,
    license_id: str,
    data_files: Iterable[str | Path],
    author_name: str,
    author_orcid: str | None = None,
    instrument: str = "https://github.com/StevenFAU/Wave_Analysis",
) -> Path:
    """Write a minimal RO-Crate 1.1 metadata file for files inside ``crate_dir``.

    File paths are recorded relative to the crate root with their SHA-256 and
    size; a ``CreateAction`` links the data to the software that produced it.
    """
    root = Path(crate_dir)
    author_id = author_orcid or f"#{author_name.replace(' ', '_')}"
    files = []
    for f in data_files:
        fp = Path(f)
        rel = fp.relative_to(root) if fp.is_absolute() else fp
        full = root / rel
        files.append(
            {
                "@id": str(rel),
                "@type": "File",
                "contentSize": str(full.stat().st_size),
                "sha256": sha256_file(full),
            }
        )
    graph: list[dict[str, Any]] = [
        {
            "@id": "ro-crate-metadata.json",
            "@type": "CreativeWork",
            "conformsTo": {"@id": "https://w3id.org/ro/crate/1.1"},
            "about": {"@id": "./"},
        },
        {
            "@id": "./",
            "@type": "Dataset",
            "name": name,
            "description": description,
            "datePublished": utcnow().date().isoformat(),
            "license": {"@id": license_id},
            "author": {"@id": author_id},
            "hasPart": [{"@id": f["@id"]} for f in files],
        },
        {"@id": author_id, "@type": "Person", "name": author_name},
        {
            "@id": "#create",
            "@type": "CreateAction",
            "instrument": {"@id": instrument},
            "result": [{"@id": f["@id"]} for f in files],
            "endTime": utcnow().isoformat(),
        },
        {
            "@id": instrument,
            "@type": "SoftwareSourceCode",
            "name": "Wave_Analysis",
            "version": __version__,
        },
        *files,
    ]
    out = root / "ro-crate-metadata.json"
    out.write_text(
        json.dumps({"@context": "https://w3id.org/ro/crate/1.1/context", "@graph": graph}, indent=2)
    )
    return out
