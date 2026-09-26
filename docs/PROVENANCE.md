# Provenance

Every artefact must answer: *what source produced this, which version, when
was it downloaded, what script processed it, with which configuration, at which
Git commit, and what changed?*

## 1. Raw acquisition → manifest

Each HTTP request by `ingest.Downloader.fetch` produces a `ManifestEntry`, which
is appended to `data/manifests/raw/<source>.csv` (committed to Git):

| Field | Purpose |
|---|---|
| `source_id`, `product`, `station_id`, `period` | What was requested |
| `url` | Exact upstream location |
| `local_path` | Where the bytes are stored (`data/raw/…`, git-ignored) |
| `sha256`, `size_bytes` | Identity of the bytes |
| `http_status`, `http_last_modified`, `http_etag` | Server metadata at retrieval |
| `retrieved_at` | UTC time of retrieval |
| `status` | `verified`, `unchanged`, **`changed`** (upstream revised the file), **`not_found`** (absence recorded as evidence), `failed` |
| `software_version` | Package version that fetched it |

Bytes are stored exactly as served (gzip is not decompressed), and writes are
atomic. When the downloader is given the previous checksum, re-fetching a
"historical" file that the provider has silently revised yields `changed`.

## 2. Derived artefacts → provenance record

`datasets.provenance.build_provenance` writes a `*.provenance.yaml` beside every
derived artefact:

```yaml
artifact_id: ndbc_inventory_20260926
created_at: 2026-09-26T01:12:40Z
git: {commit: <sha>, branch: main, dirty: false}
pipeline:
  command: wave-analysis ndbc inventory
  config: {...}
  config_sha256: <canonical-JSON hash>
software:
  python: 3.12.13
  wave_analysis: 0.1.0
  packages: {numpy: ..., pandas: ..., ...}
input_files:  [{path: ..., sha256: ...}, ...]
upstream_manifests: [...]
output_files: [{path: ..., sha256: ...}, ...]
```

**A result produced from a dirty working tree is not publishable.** The `dirty`
flag makes this visible.

## 3. Observation-level lineage

Every standardized row carries `source_file`, `source_url`, `source_checksum`,
`processing_level`, and `processing_version`. Derived rows carry the estimator in
`method`. Any label can therefore be traced to its raw bytes.

## 4. Sample-level lineage

`MultimodalSample.provenance` lists every upstream source, URL, and checksum
behind a synchronised sample, plus the pipeline version and Git commit.

## 5. ML manifests

`datasets.builder.build_manifest` writes three files: the manifest, an
**exclusion ledger** (sample id → first rule that excluded it), and a provenance
record including the split protocol, its parameters, and the leakage-audit
result. The population a metric applies to is therefore always stated.

## 6. Release packaging

For a published dataset, `datasets.provenance.write_ro_crate` produces an
RO-Crate 1.1 `ro-crate-metadata.json` (data entities with SHA-256 and size, a
`CreateAction` linking them to the software version, licence, and author). The
crate plus the provenance YAMLs are archived on Zenodo with a DataCite DOI
(`docs/CITATION_POLICY.md`).
