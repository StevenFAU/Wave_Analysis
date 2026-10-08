"""MkDocs hook: render ``data/registry/data_dictionary.yaml`` into the inventory page.

``docs/datasets/inventory.md`` holds the prose; the tables come from the data
dictionary, the same file the dashboard's Data view reads, so the two cannot
disagree. Each ``<!-- dictionary:<part> -->`` marker on that page is replaced
at build time, with ``<part>`` one of ``roles``, ``overview``, ``collections``,
``tables``, ``crosswalk``. Registered in ``mkdocs.yml`` under ``hooks``.

The dictionary is validated by :func:`wave_analysis.data_dictionary.load_data_dictionary`
first, so ``mkdocs build --strict`` fails on an invalid file.
"""

from __future__ import annotations

import posixpath
import re
import unicodedata
from pathlib import Path
from typing import Any

from wave_analysis.data_dictionary import Collection, DataDictionary, Variable, load_data_dictionary

PAGE = "datasets/inventory.md"
_MARKER = re.compile(r"<!--\s*dictionary:(\w+)\s*-->")
_SUB = re.compile(r"_\{([^}]+)\}")


def _text(s: str | None) -> str:
    """Dictionary markup -> Markdown table-cell text."""
    if not s:
        return ""
    return _SUB.sub(r"<sub>\1</sub>", s).replace("|", "\\|").strip()


def _link(doc: str) -> str:
    """Path of a ``docs/...`` file relative to the inventory page."""
    path, _, anchor = doc.partition("#")
    rel = posixpath.relpath(path.removeprefix("docs/"), posixpath.dirname(PAGE))
    return f"{rel}#{anchor}" if anchor else rel


def _roles(dd: DataDictionary, ids: list[str]) -> str:
    labels = {r.id: r.label for r in dd.roles}
    return ", ".join(labels[i] for i in ids)


def _table(head: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def render_roles(dd: DataDictionary) -> str:
    return _table(
        ["Role", "Its job", "What it means"],
        [[f"**{r.label}**", _text(r.short), _text(r.description)] for r in dd.roles],
    )


def render_overview(dd: DataDictionary) -> str:
    groups = {g.id: g.title for g in dd.groups}
    rows = [
        [
            f"[{_text(c.name)}](#{_slug(c.name)})",
            _roles(dd, c.roles),
            _text(c.where),
            _text(c.cadence),
            groups[c.group],
        ]
        for c in dd.collections
    ]
    return _table(["Collection", "Role", "Where", "How often", "Section"], rows)


def _variables(dd: DataDictionary, variables: list[Variable]) -> str:
    has = {k: any(getattr(v, k) for v in variables) for k in ("unit", "role", "source")}
    head = ["Variable", "Meaning"] + [
        h for k, h in (("unit", "Unit"), ("role", "Role"), ("source", "Source")) if has[k]
    ]
    labels = {r.id: r.label for r in dd.roles}
    rows: list[list[str]] = []
    for v in variables:
        if v.group:
            rows.append([f"**{_text(v.group)}**"] + [""] * (len(head) - 1))
        meaning = _text(v.meaning)
        if v.canonical:
            meaning += f" → `{v.canonical}`"
        row = ["<br>".join(f"`{n}`" for n in v.name.split(", ")), meaning]
        if has["unit"]:
            row.append(_text(v.unit))
        if has["role"]:
            row.append(labels.get(v.role or "", ""))
        if has["source"]:
            row.append(_text(v.source))
        rows.append(row)
    return _table(head, rows)


def _collection(dd: DataDictionary, c: Collection) -> str:
    parts = [
        f"#### {_text(c.name)}",
        "",
        f"{_text(c.subtitle)}. {_text(c.provider)} · role: {_roles(dd, c.roles)} · "
        f"[datasheet]({_link(c.doc)}) · status id `{c.id}`",
        "",
        f"**Where:** {_text(c.where)}. **How often:** {_text(c.cadence)}.",
    ]
    if c.facts:
        parts += [
            "",
            _table(["Fact", "Value"], [[_text(f.label), _text(f.value)] for f in c.facts]),
        ]
    if c.variables:
        parts += ["", _variables(dd, c.variables)]
    parts += [""] + [f"- {_text(n)}" for n in c.notes]
    return "\n".join(parts).rstrip()


def render_collections(dd: DataDictionary) -> str:
    out: list[str] = []
    for g in dd.groups:
        items = [c for c in dd.collections if c.group == g.id]
        if not items:
            continue
        out += [f"### {_text(g.title)}", "", _text(g.intro), ""]
        out += [_collection(dd, c) + "\n" for c in items]
    return "\n".join(out).rstrip()


def render_tables(dd: DataDictionary) -> str:
    return _table(
        ["Table", "Role", "What it holds", "Size", "Where"],
        [
            [
                f"[{_text(t.name)}]({_link(t.doc)})",
                _roles(dd, t.roles),
                _text(t.holds),
                _text(t.shape),
                f"`{t.path}`",
            ]
            for t in dd.tables
        ],
    )


def render_crosswalk(dd: DataDictionary) -> str:
    cw = dd.crosswalk
    rows = []
    for r in cw.rows:
        cells = []
        for col in cw.columns:
            cell = r.cells.get(col.id)
            if cell is None or (not cell.vars and not cell.note):
                cells.append("–")
                continue
            names = " ".join(f"`{n}`" for n in cell.vars)
            note = _text(cell.note)
            cells.append(f"{names} ({note})" if names and note else names or f"*{note}*")
        quantity = _text(r.quantity) + (f"<br><small>{_text(r.note)}</small>" if r.note else "")
        rows.append([quantity, *cells])
    return _text(cw.intro) + "\n\n" + _table(["Quantity", *[c.label for c in cw.columns]], rows)


def _slug(title: str) -> str:
    """The anchor MkDocs' toc extension gives a heading (Python-Markdown's slugify)."""
    s = unicodedata.normalize("NFKD", _SUB.sub(r"\1", title)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    return re.sub(r"[-\s]+", "-", s)


RENDERERS = {
    "roles": render_roles,
    "overview": render_overview,
    "collections": render_collections,
    "tables": render_tables,
    "crosswalk": render_crosswalk,
}


def render_page(markdown: str, dd: DataDictionary) -> str:
    """Replace every marker in ``markdown``; an unknown marker is an error."""

    def sub(m: re.Match[str]) -> str:
        part = m.group(1)
        if part not in RENDERERS:
            raise ValueError(f"{PAGE}: unknown dictionary marker {part!r}")
        return RENDERERS[part](dd)

    return _MARKER.sub(sub, markdown)


def on_page_markdown(markdown: str, *, page: Any, config: Any, files: Any) -> str:
    if page.file.src_uri != PAGE:
        return markdown
    root = Path(config["config_file_path"]).parent
    dd = load_data_dictionary(root / "data" / "registry" / "data_dictionary.yaml")
    return render_page(markdown, dd)
