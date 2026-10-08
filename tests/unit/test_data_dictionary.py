"""Data dictionary: validation, agreement with the adapters, and the docs hook."""

from __future__ import annotations

import copy
import importlib.util
import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from wave_analysis.config import load_yaml, repo_root
from wave_analysis.dashboard.catalog import build_catalog, read_docs
from wave_analysis.data_dictionary import DataDictionary, check_references, load_data_dictionary
from wave_analysis.registry import validate_registry
from wave_analysis.sources.cdip import BULK_MAP
from wave_analysis.sources.era5 import PARAMETERS
from wave_analysis.sources.ndbc.source import SPEC_MAP, STDMET_MAP

ROOT = repo_root(Path(__file__))
PATH = ROOT / "data/registry/data_dictionary.yaml"

#: Every collection id the live status can publish (dashboard/status.py and
#: dashboard/collections.py). The dictionary must describe each of them.
LIVE_IDS = {
    "ndbc_buoycam",
    "ndbc_realtime",
    "pacioos_waimea",
    "webcoos",
    "era5_waves",
    "cdip",
    "ndbc_history",
    "external",
}


@pytest.fixture(scope="module")
def dd() -> DataDictionary:
    return load_data_dictionary(PATH)


def _raw() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def test_dictionary_describes_every_live_collection(dd):
    assert {c.id for c in dd.collections} == LIVE_IDS


def test_dictionary_references_resolve(dd):
    registry = load_yaml(ROOT / "data/registry/datasets.yaml")
    assert validate_registry(registry) == []
    datasets = {d["dataset_id"] for d in registry["datasets"]}
    assert check_references(dd, datasets=datasets, docs=read_docs(ROOT / "docs")) == []
    assert check_references(dd, datasets=set(), docs=set())  # and it does report problems


def _canonical(dd: DataDictionary, collection: str) -> dict[str, str]:
    c = next(c for c in dd.collections if c.id == collection)
    return {v.name: v.canonical for v in c.variables if v.canonical}


@pytest.mark.parametrize(
    ("collection", "adapter"),
    [
        (
            "ndbc_realtime",
            {k: v[0] for k, v in STDMET_MAP.items()} | {k: v[0] for k, v in SPEC_MAP.items()},
        ),
        ("cdip", BULK_MAP),
        ("era5_waves", {p.short: p.variable for p in PARAMETERS}),
    ],
)
def test_canonical_keys_match_the_adapters(dd, collection, adapter):
    described = _canonical(dd, collection)
    for name, key in adapter.items():
        assert described.get(name) == key, f"{collection}.{name}: adapter maps to {key}"


def test_ndbc_history_columns_are_the_standard_file(dd):
    # The history collection points at the realtime standard-file table rather
    # than repeating it; make sure that table has every stdmet column.
    names = {v.name for c in dd.collections if c.id == "ndbc_realtime" for v in c.variables}
    assert set(STDMET_MAP) <= names


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d["collections"][0].update(roles=["nonsense"]), "unknown role 'nonsense'"),
        (lambda d: d["collections"][0].update(group="nowhere"), "unknown group 'nowhere'"),
        (
            lambda d: next(c for c in d["collections"] if c["id"] == "cdip")["variables"][0].update(
                canonical="wave_height_guess"
            ),
            "unknown canonical variable 'wave_height_guess'",
        ),
        (
            lambda d: d["crosswalk"]["rows"][0]["cells"].update(ndbc="WAVEHEIGHT"),
            "'WAVEHEIGHT' is not a variable",
        ),
        (lambda d: d["crosswalk"]["rows"][0]["cells"].update(nope="WVHT"), "unknown column 'nope'"),
        (
            lambda d: d["tables"].append(copy.deepcopy(d["tables"][0])),
            "duplicate collection or table id",
        ),
        (lambda d: d["crosswalk"]["columns"][0].update(table="waimea_pairing"), "exactly one of"),
    ],
)
def test_invalid_dictionary_is_rejected(edit, message):
    raw = _raw()
    edit(raw)
    with pytest.raises(ValidationError, match=re.escape(message)):
        DataDictionary.model_validate(raw)


def test_crosswalk_shorthand(dd):
    row = next(r for r in dd.crosswalk.rows if r.quantity == "Wind")
    assert row.cells["ndbc"].vars == ["WDIR", "WSPD", "GST"]
    assert row.cells["era5"].vars == [] and row.cells["era5"].note == "not held"


def test_catalog_carries_the_dictionary_with_doc_urls():
    cat = build_catalog(ROOT)
    d = cat["dictionary"]
    first = d["collections"][0]
    assert (first["id"], first["doc_url"]) == ("ndbc_buoycam", "docs/datasets/ndbc/")
    assert all(t["doc_url"].startswith("docs/") for t in d["tables"])


# ----------------------------------------------------------------------------- docs hook


def _hook():
    spec = importlib.util.spec_from_file_location(
        "mkdocs_data_dictionary", ROOT / "scripts/mkdocs_data_dictionary.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_docs_hook_replaces_every_marker(dd):
    hook = _hook()
    page = (ROOT / "docs/datasets/inventory.md").read_text(encoding="utf-8")
    markers = set(re.findall(r"<!--\s*dictionary:(\w+)\s*-->", page))
    assert markers == set(hook.RENDERERS)
    out = hook.render_page(page, dd)
    assert "dictionary:" not in out
    assert "| `WVHT` |" in out and "→ `wave_significant_height`" in out
    assert "H<sub>m0</sub>" in out  # _{m0} markup
    # Overview links point at the headings the collections section creates.
    anchors = set(re.findall(r"\]\(#([a-z0-9-]+)\)", out))
    headings = {hook._slug(m) for m in re.findall(r"^#### (.+)$", out, flags=re.M)}
    assert anchors and anchors <= headings
    with pytest.raises(ValueError, match="unknown dictionary marker"):
        hook.render_page("<!-- dictionary:nope -->", dd)
