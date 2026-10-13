"""Configuration loading and repository paths.

Configurations are plain YAML under ``configs/`` and are always stored
alongside the artefacts they produced (via provenance records), so a result
can be regenerated from the exact settings used.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def repo_root(start: Path | None = None) -> Path:
    """Locate the repository root (directory containing ``pyproject.toml``).

    Overridable with the ``WAVE_ANALYSIS_ROOT`` environment variable.
    """
    env = os.environ.get("WAVE_ANALYSIS_ROOT")
    if env:
        return Path(env).resolve()
    p = (start or Path.cwd()).resolve()
    for cand in (p, *p.parents):
        if (cand / "pyproject.toml").exists() and (cand / "src" / "wave_analysis").exists():
            return cand
    return p


def data_dir(kind: str = "raw", root: Path | None = None) -> Path:
    """Path to ``data/<kind>`` (``WAVE_ANALYSIS_DATA`` overrides ``data/``)."""
    base = (
        Path(os.environ["WAVE_ANALYSIS_DATA"])
        if "WAVE_ANALYSIS_DATA" in os.environ
        else repo_root(root) / "data"
    )
    return base / kind


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    with Path(path).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a mapping")
    return data


def portable_path(path: str | Path, root: Path | None = None) -> str:
    """Path relative to the repository root when inside it (POSIX separators), else unchanged.

    Committed manifests and provenance records must not embed machine-specific
    absolute paths. The path is tried as written before symbolic links are
    followed, so a file under a collection that is a link to another drive
    (``data/raw/webcoos_timex -> /mnt/...``) is still recorded by its place in
    the repository.
    """
    p = Path(path)
    base = repo_root(root)
    for candidate in (Path(os.path.abspath(p)), p.resolve()):
        try:
            return candidate.relative_to(base).as_posix()
        except ValueError:
            continue
    return str(path)
