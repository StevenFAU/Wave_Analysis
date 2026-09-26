"""Minimal BibTeX reader for the project bibliography.

It handles what ``docs/literature/bibliography.bib`` uses: ``@type{key, ...}``
entries with brace- or quote-delimited fields (nested braces allowed),
``%`` comment lines, and section banners (a comment of dashes followed by a
label, on the same line or the next comment line). It is not a general BibTeX
implementation: ``@string`` macros and ``#`` concatenation are not supported,
and a file that uses them raises ``ValueError`` instead of being misread.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_ENTRY_START = re.compile(r"@(?P<type>[A-Za-z]+)\s*\{\s*(?P<key>[^,\s]+)\s*,")
_BANNER = re.compile(r"^%\s*-{5,}\s*(?P<label>.*)$")
_FIELD_NAME = re.compile(r"\s*(?P<name>[A-Za-z][A-Za-z0-9_-]*)\s*=\s*")

# LaTeX accent commands used in author names -> combining characters.
_ACCENTS = {
    "'": "\u0301",
    "`": "\u0300",
    "^": "\u0302",
    '"': "\u0308",
    "~": "\u0303",
    "c": "\u0327",
    "v": "\u030c",
    "=": "\u0304",
}
_ACCENT_CMD = re.compile(r"\{?\\(['`^\"~cv=])\s*\{?([A-Za-z])\}?\}?")


@dataclass(frozen=True)
class BibEntry:
    """One bibliography entry, with fields cleaned for display."""

    key: str
    entry_type: str
    fields: dict[str, str]
    section: str | None
    raw: str
    authors: list[str] = field(default_factory=list)

    @property
    def year(self) -> int | None:
        """Four-digit year from the ``year`` field, if any."""
        m = re.search(r"\d{4}", self.fields.get("year", ""))
        return int(m.group()) if m else None

    @property
    def first_author_surname(self) -> str | None:
        """Family name of the first author; a corporate author is returned whole."""
        if not self.authors:
            return None
        name = self.authors[0]
        first_raw = _split_top_level(self.fields.get("author", ""))[0].strip()
        if first_raw.startswith("{") and _closing_brace(first_raw, 0) == len(first_raw) - 1:
            return name  # "{IAHR Working Group ...}": one braced corporate name
        return name.split(",")[0].strip() if "," in name else name.split()[-1]


def _closing_brace(text: str, i: int) -> int:
    """Index of the brace closing the one opened at ``text[i]`` (-1 if none)."""
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return j
    return -1


def delatex(text: str) -> str:
    """Turn BibTeX field text into plain Unicode (accents, braces, dashes)."""
    out = _ACCENT_CMD.sub(lambda m: unicodedata.normalize("NFC", m[2] + _ACCENTS[m[1]]), text)
    out = out.replace("---", "\u2014").replace("--", "\u2013").replace("\\&", "&")
    out = out.replace("{", "").replace("}", "").replace("~", "\u00a0")
    return re.sub(r"\s+", " ", out).strip()


def ascii_fold(text: str) -> str:
    """Strip diacritics (``Guimarães`` -> ``Guimaraes``) for matching."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def split_authors(value: str) -> list[str]:
    """Split an author field on top-level ``and``; names are de-LaTeXed."""
    return [delatex(p) for p in _split_top_level(value) if p.strip()]


def _split_top_level(value: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    i = 0
    while i < len(value):
        ch = value[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif depth == 0 and value.startswith(" and ", i):
            parts.append(value[start:i])
            start = i + 5
            i += 4
        i += 1
    parts.append(value[start:])
    return parts


def _read_value(body: str, i: int) -> tuple[str, int]:
    """Read one field value starting at ``body[i]``; return (value, next index)."""
    if body[i] == "{":
        depth = 0
        for j in range(i, len(body)):
            if body[j] == "{":
                depth += 1
            elif body[j] == "}":
                depth -= 1
                if depth == 0:
                    return body[i + 1 : j], j + 1
        raise ValueError("unbalanced braces in field value")
    if body[i] == '"':
        depth = 0
        for j in range(i + 1, len(body)):
            if body[j] == "{":
                depth += 1
            elif body[j] == "}":
                depth -= 1
            elif body[j] == '"' and depth == 0:
                return body[i + 1 : j], j + 1
        raise ValueError("unterminated quoted field value")
    m = re.match(r"[A-Za-z0-9_.:/-]+", body[i:])
    if m is None:
        raise ValueError(f"unexpected field value at {body[i : i + 20]!r}")
    if not m.group().isdigit():
        raise ValueError(f"@string macros are not supported: {m.group()!r}")
    return m.group(), i + m.end()


def _parse_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    i = 0
    while i < len(body):
        m = _FIELD_NAME.match(body, i)
        if m is None:
            if body[i:].strip(" \t\r\n,") == "":
                break
            raise ValueError(f"cannot parse field near {body[i : i + 30]!r}")
        value, i = _read_value(body, m.end())
        rest = body[i:].lstrip()
        if rest.startswith("#"):
            raise ValueError("'#' concatenation is not supported")
        fields[m["name"].lower()] = value
        comma = body.find(",", i)
        i = len(body) if comma == -1 else comma + 1
    return fields


def _entry_end(text: str, open_brace: int) -> int:
    depth = 0
    for j in range(open_brace, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return j
    raise ValueError("unbalanced braces: entry never closes")


def _section_label(raw: str) -> str:
    label = re.sub(r"\([^)]*\)", "", raw).strip().rstrip(",.;:").strip()
    return label[:1].upper() + label[1:] if label else label


def parse_bibtex(text: str) -> list[BibEntry]:
    """Parse a ``.bib`` file into entries, in file order, with section labels."""
    entries: list[BibEntry] = []
    section: str | None = None
    awaiting_label = False
    pos = 0
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for ln in lines:
        offsets.append(offsets[-1] + len(ln))

    line_no = 0
    prev_comment = False
    while line_no < len(lines):
        line = lines[line_no]
        stripped = line.strip()
        start = offsets[line_no]
        if start < pos:  # inside an entry already consumed
            line_no += 1
            continue
        if stripped.startswith("%"):
            m = _BANNER.match(stripped)
            if m is not None:
                label = m["label"].strip()
                if label:
                    section = _section_label(label)
                    awaiting_label = False
                else:
                    # A bare rule opens a banner block, or closes one it follows.
                    awaiting_label = not prev_comment
            elif awaiting_label:
                section = _section_label(stripped.lstrip("% "))
                awaiting_label = False
            prev_comment = True
            line_no += 1
            continue
        prev_comment = False
        special = re.match(r"@(string|preamble)\b", stripped, flags=re.I)
        if special:
            raise ValueError(f"@{special.group(1)} is not supported")
        m = _ENTRY_START.search(text, start)
        if m is None or m.start() >= offsets[line_no + 1]:
            line_no += 1
            continue
        open_brace = text.index("{", m.start())
        end = _entry_end(text, open_brace)
        body = text[m.end() : end]
        fields = _parse_fields(body)
        entries.append(
            BibEntry(
                key=m["key"],
                entry_type=m["type"].lower(),
                fields=fields,
                section=section,
                raw=text[m.start() : end + 1],
                authors=split_authors(fields["author"]) if "author" in fields else [],
            )
        )
        pos = end + 1
        line_no += 1
    keys = [e.key for e in entries]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise ValueError(f"duplicate bibliography keys: {dupes}")
    return entries
