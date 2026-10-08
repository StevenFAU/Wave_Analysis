"""Data dictionary schema (``data/registry/data_dictionary.yaml``).

The dictionary says what each data collection is for (its *roles*) and what
every field or variable in it means, in plain language. It is the single
source for the dashboard's Data view and for ``docs/datasets/inventory.md``
(rendered by ``scripts/mkdocs_data_dictionary.py``). Counts and sizes are not part
of it: those come from the collector's live status, whose collection ids match
the dictionary's.

:func:`load_data_dictionary` validates the file on its own: unique ids, known
roles and groups, canonical keys present in :mod:`wave_analysis.schemas.variables`,
and crosswalk cells that name a variable of the collection (or a column of the
table) their column points to. :func:`check_references` adds the checks that
need the rest of the repository (dataset registry ids, documentation paths).

Text fields use two bits of markup, understood by both renderers:
```` `code` ```` and ``_{x}`` for a subscript.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wave_analysis.config import load_yaml
from wave_analysis.schemas.variables import VARIABLES

_ID = r"^[a-z0-9_]+$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Role(_Strict):
    """A job data does in the project (pictures, labels, ...)."""

    id: str = Field(pattern=_ID)
    label: str
    short: str
    description: str


class Group(_Strict):
    """A section of the inventory that collections are listed under."""

    id: str = Field(pattern=_ID)
    title: str
    intro: str


class Fact(_Strict):
    """A labelled fact about a collection (image size, record length, ...)."""

    label: str
    value: str


class Variable(_Strict):
    """One field, variable or file of a collection."""

    name: str
    meaning: str
    unit: str | None = None
    role: str | None = None
    source: str | None = None  # where the value comes from (file name, EXIF, ...) or a size
    canonical: str | None = None  # key in schemas/variables.py
    group: str | None = None  # starts a sub-heading in the variable table


class Collection(_Strict):
    """A data collection; ``id`` matches ``status.json`` ``collections[].id``."""

    id: str = Field(pattern=_ID)
    group: str
    name: str
    subtitle: str
    provider: str
    datasets: list[str] = Field(min_length=1)  # dataset registry ids
    doc: str
    roles: list[str] = Field(min_length=1)
    where: str
    cadence: str
    facts: list[Fact] = Field(default_factory=list)
    variables: list[Variable] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Table(_Strict):
    """A table the project built (pairing, audit, benchmark, ledger, ...)."""

    id: str = Field(pattern=_ID)
    name: str
    roles: list[str] = Field(min_length=1)
    shape: str
    path: str
    doc: str
    holds: str
    columns: list[str] = Field(default_factory=list)  # columns the crosswalk may name


class Cell(_Strict):
    """A crosswalk cell: variable names, a note, or both (empty: not available)."""

    vars: list[str] = Field(default_factory=list)
    note: str | None = None


class CrosswalkColumn(_Strict):
    """A crosswalk column: one collection's or one table's names."""

    id: str = Field(pattern=_ID)
    label: str
    collection: str | None = None
    table: str | None = None

    @model_validator(mode="after")
    def _one_target(self) -> CrosswalkColumn:
        if (self.collection is None) == (self.table is None):
            raise ValueError(f"crosswalk column {self.id!r}: give exactly one of collection, table")
        return self


class CrosswalkRow(_Strict):
    """One physical quantity across sources."""

    quantity: str
    cells: dict[str, Cell]
    note: str | None = None

    @field_validator("cells", mode="before")
    @classmethod
    def _shorthand(cls, v: Any) -> Any:
        # "WVHT" -> {vars: [WVHT]}; [WDIR, WSPD] -> {vars: [...]}; null -> {}
        if not isinstance(v, dict):
            return v
        out: dict[str, Any] = {}
        for k, c in v.items():
            if c is None:
                out[k] = {}
            elif isinstance(c, str):
                out[k] = {"vars": [c]}
            elif isinstance(c, list):
                out[k] = {"vars": c}
            else:
                out[k] = c
        return out


class Crosswalk(_Strict):
    """The same quantity under each source's name."""

    intro: str
    columns: list[CrosswalkColumn]
    rows: list[CrosswalkRow]


class DataDictionary(_Strict):
    """The whole file."""

    dictionary_version: str
    checked: dt.date
    roles: list[Role]
    groups: list[Group]
    collections: list[Collection]
    tables: list[Table]
    crosswalk: Crosswalk

    @model_validator(mode="after")
    def _consistent(self) -> DataDictionary:
        problems = internal_problems(self)
        if problems:
            raise ValueError("; ".join(problems))
        return self


def _duplicates(ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    dup: list[str] = []
    for i in ids:
        if i in seen:
            dup.append(i)
        seen.add(i)
    return dup


def internal_problems(dd: DataDictionary) -> list[str]:
    """Problems found within the file itself (see module docstring)."""
    problems: list[str] = []
    roles = {r.id for r in dd.roles}
    groups = {g.id for g in dd.groups}
    for kind, ids in (
        ("role", [r.id for r in dd.roles]),
        ("group", [g.id for g in dd.groups]),
        ("collection or table", [c.id for c in dd.collections] + [t.id for t in dd.tables]),
        ("crosswalk column", [c.id for c in dd.crosswalk.columns]),
    ):
        problems += [f"duplicate {kind} id {d!r}" for d in _duplicates(ids)]

    for c in dd.collections:
        if c.group not in groups:
            problems.append(f"{c.id}: unknown group {c.group!r}")
        problems += [f"{c.id}: unknown role {r!r}" for r in c.roles if r not in roles]
        for v in c.variables:
            if v.role is not None and v.role not in roles:
                problems.append(f"{c.id}.{v.name}: unknown role {v.role!r}")
            if v.canonical is not None and v.canonical not in VARIABLES:
                problems.append(f"{c.id}.{v.name}: unknown canonical variable {v.canonical!r}")
    for t in dd.tables:
        problems += [f"{t.id}: unknown role {r!r}" for r in t.roles if r not in roles]

    names = {c.id: {v.name for v in c.variables} for c in dd.collections}
    columns = {t.id: set(t.columns) for t in dd.tables}
    targets: dict[str, set[str]] = {}
    for col in dd.crosswalk.columns:
        if col.collection is not None:
            if col.collection not in names:
                problems.append(
                    f"crosswalk column {col.id!r}: unknown collection {col.collection!r}"
                )
            targets[col.id] = names.get(col.collection or "", set())
        else:
            if col.table not in columns:
                problems.append(f"crosswalk column {col.id!r}: unknown table {col.table!r}")
            targets[col.id] = columns.get(col.table or "", set())
    for row in dd.crosswalk.rows:
        for col_id, cell in row.cells.items():
            if col_id not in targets:
                problems.append(f"crosswalk {row.quantity!r}: unknown column {col_id!r}")
                continue
            problems += [
                f"crosswalk {row.quantity!r}: {name!r} is not a variable of column {col_id!r}"
                for name in cell.vars
                if name not in targets[col_id]
            ]
    return problems


def check_references(
    dd: DataDictionary, *, datasets: Iterable[str], docs: Iterable[str]
) -> list[str]:
    """References to the rest of the repository: registry ids and doc paths."""
    known = set(datasets)
    doc_paths = set(docs)
    problems = [
        f"{c.id}: unknown dataset id {d!r}"
        for c in dd.collections
        for d in c.datasets
        if d not in known
    ]
    docs_of = [(c.id, c.doc) for c in dd.collections] + [(t.id, t.doc) for t in dd.tables]
    for item_id, doc in docs_of:
        if doc.partition("#")[0] not in doc_paths:
            problems.append(f"{item_id}: documentation {doc!r} does not exist")
    return problems


def load_data_dictionary(path: Path) -> DataDictionary:
    """Parse and validate ``data_dictionary.yaml``."""
    return DataDictionary.model_validate(load_yaml(path))
