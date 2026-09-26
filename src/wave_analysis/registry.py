"""Dataset registry schema (``data/registry/datasets.yaml``).

The registry is the verified catalogue of candidate public sources. Each entry
records *how we know* what it claims: a ``verification`` block with status,
date, and evidence URL. Unverified claims are allowed (the registry is also a
research to-do list) but must be labelled as such.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wave_analysis.schemas.sensor import Modality, SensorRole


class VerificationStatus(StrEnum):
    """How thoroughly an entry's access and content claims have been checked."""

    VERIFIED = "verified"  # accessed programmatically; formats confirmed on real files
    DOCUMENTED = "documented"  # confirmed from provider documentation only
    CANDIDATE = "candidate"  # plausible, not yet checked


class Verification(BaseModel):
    """Evidence for a registry entry."""

    model_config = ConfigDict(extra="forbid")

    status: VerificationStatus
    date: dt.date | None = None
    evidence: list[str] = Field(default_factory=list)
    notes: str | None = None


class Access(BaseModel):
    """How the data are obtained."""

    model_config = ConfigDict(extra="forbid")

    public: bool
    authentication_required: bool
    protocols: list[str]
    endpoints: list[str] = Field(default_factory=list)
    rate_limit_guidance: str | None = None


class License(BaseModel):
    """Terms of use."""

    model_config = ConfigDict(extra="forbid")

    status: str
    url: str | None = None
    notes: str | None = None


class DatasetEntry(BaseModel):
    """One registry entry."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str
    provider: str
    modalities: list[Modality]
    intended_roles: list[SensorRole]
    access: Access
    data_types: list[str]
    temporal: dict[str, Any] = Field(default_factory=dict)
    spatial: dict[str, Any] = Field(default_factory=dict)
    formats: list[str] = Field(default_factory=list)
    documentation: list[str] = Field(default_factory=list)
    citation: str | None = None
    license: License
    phase: str
    adapter: str | None = None
    doc: str | None = None
    verification: Verification
    notes: str | None = None


def validate_registry(data: dict[str, Any]) -> list[str]:
    """Return a list of human-readable problems (empty if the registry is valid)."""
    problems: list[str] = []
    entries = data.get("datasets")
    if not isinstance(entries, list):
        return ["top-level key 'datasets' must be a list"]
    seen: set[str] = set()
    for i, raw in enumerate(entries):
        try:
            e = DatasetEntry.model_validate(raw)
        except ValidationError as exc:
            ident = raw.get("dataset_id", f"#{i}") if isinstance(raw, dict) else f"#{i}"
            for err in exc.errors():
                loc = ".".join(str(x) for x in err["loc"])
                problems.append(f"{ident}: {loc}: {err['msg']}")
            continue
        if e.dataset_id in seen:
            problems.append(f"duplicate dataset_id {e.dataset_id!r}")
        seen.add(e.dataset_id)
        if e.verification.status == VerificationStatus.VERIFIED and not e.verification.evidence:
            problems.append(f"{e.dataset_id}: 'verified' entries must list evidence URLs")
    return problems
