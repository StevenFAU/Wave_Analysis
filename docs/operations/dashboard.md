# Public Dashboard

The dashboard at **<https://stevenfau.github.io/Wave_Analysis/>** shows what
the project is collecting and how complete it is, the recent sea state at
every camera buoy, the public data sources in the registry, and the literature
behind the project's claims. This documentation is served under
[`/docs/`](https://stevenfau.github.io/Wave_Analysis/docs/). The design decision is
[ADR 0009](../decisions/0009-public-dashboard.md).

## What each view shows

| View | Content | Built from |
|---|---|---|
| **Overview** | Collector health, images archived, 7-day coverage, archive size and projected growth, images per hour by illumination, collector runs, latest daylight images | live status |
| **Cameras** | Map of NDBC buoy cameras (coloured by current H_s), shore-camera sites and their reference buoys, with base layers for streets and GEBCO bathymetry. Sortable station table with 7-day coverage strips | live status, catalog |
| **Station** (`#/station/<id>`) | Image viewer for the last ~72 h, including a six-view split. For each image, the nearest buoy observation: H_s, periods, direction, swell and wind-sea, wind, water temperature. Linked sea-state charts, an hour-by-day coverage grid, NDBC archive years per product, and the camera sites that use the buoy as a reference | live status, sea state, catalog |
| **Archive** | Coverage heatmap (station × hour) with window, region and gap filters; images per day; storage; minute-stamp distribution; integrity checks; realtime-spectra snapshots; the run log | live status |
| **Sources** (`#/sources/<id>`) | The dataset registry with status, modality and role filters. Each entry shows access, endpoints, evidence, licence, citation and related camera sites | catalog |
| **Literature** (`#/literature/<key>`) | The bibliography grouped by section, with each entry's verification level and basis, the docs that cite it, DOI/arXiv links and BibTeX. Screened (unread) items are listed separately | catalog |
| **About** | Data flow, freshness, schemas, caveats, documentation index | catalog, live status |

Every chart has a table view, and every URL can be shared as a deep link: a
specific image is `#/station/41010/<unix-seconds>`, and archive filters go in
the query (`#/archive?h=24&region=46`).

## How it works

```
collector host (hourly, :52)                GitHub
───────────────────────────                 ──────
archive + ledger ──┐
                   ├─ wave-analysis dashboard live ─► status.json, seastate/*.json
NDBC realtime2 ────┘      (range requests)                │
                                                          ▼
                          scripts/publish_dashboard.sh ─► branch dashboard-data
                                                          (one commit, force-pushed)

repository (main) ── .github/workflows/site.yml ─► catalog.json + front end + MkDocs
                                                   ─► GitHub Pages
browser: site from Pages; live data from raw.githubusercontent.com/<repo>/dashboard-data/
         (falls back to the copy bundled at the last deploy)
```

- **Catalog** (`data/catalog.json`, `wave-analysis dashboard catalog`): built in
  CI from files in the repository: `data/registry/datasets.yaml`,
  `data/registry/camera_sites.yaml`, `stations.parquet` and
  `ndbc_station_summary.csv`, `docs/literature/bibliography.bib`, and the
  *Verification index* table in `source_verification.md`, plus the MkDocs
  navigation and Markdown. The build fails on a verification row for an
  unknown key, a site that references an unknown station or dataset, or an
  invalid registry.
- **Live data** (`wave-analysis dashboard live`): built on the collector host
  from the archive, its ledger, and the sea-state cache. It contains **no
  local paths, host names or user names**, and a test enforces this. A
  `manifest.json` records the size and SHA-256 of every file.
- **Publication** (`scripts/publish_dashboard.sh`): writes the files as a
  parentless commit with git plumbing and a temporary index, then force-pushes it
  to `dashboard-data`. The working tree, the real index and the checked-out
  branch are never touched, and the branch never accumulates history.
- **Site** (`.github/workflows/site.yml`): on each push to `main`, and every 6 h
  to refresh the bundled fallback, CI builds the front end and catalog,
  bundles the current `dashboard-data`, builds MkDocs into `docs/`, runs
  `wave-analysis dashboard check --require-docs`, and deploys to Pages. Pull
  requests build and check without deploying.

### Sea state

NDBC's realtime files (`realtime2/<id>.txt`, `.spec`) are newest-first, and the
server honours HTTP range requests, so the host fetches only the head of each
file. That is about 6 KB per station per hour, against ~600 KB for the whole
45-day file. The first run, or the first after a long outage, requests up to
96 KB. Rows are merged into a rolling 8-day cache in
`data/interim/dashboard/seastate/`. NDBC ignores `Range` on gzip-encoded
responses, so requests ask for `Accept-Encoding: identity`; the refresh
summary counts any oversized response. About 200 requests an hour at 1 per
second take about 3.5 minutes.

These values are **provisional realtime data for display**. They are not the
project's research record: labels come from the archived realtime snapshots
and NDBC historical files. The browser cannot read NDBC's files directly,
because NDBC sends no CORS headers.

### Images

The public site never hosts archive images. The viewer loads images from
NDBC's own server, so only the ~72 h that NDBC keeps can be shown. Older
images exist only in the project archive and its offsite copy. Citable
releases go to Zenodo (ADR 0008).

## Status file reference

`status.json` (`wave-analysis/dashboard-status@1`):

| Field | Meaning |
|---|---|
| `generated_at`, `window` | Build time. The window is `hours` UTC hours ending at the hour of `as_of`, which is the newest listing snapshot, i.e. the last collector run |
| `collector.last_run`, `collector.runs[]` | Each run starts with a listing request. The images and 404s fetched until the next listing count towards that run |
| `archive` | Totals from the files on disk: images, bytes, first and last image, stations; coverage, illumination and minute stamps in the window; ledger counts |
| `daily` | Images and bytes per UTC day over the whole archive. Each station also carries its own `daily` array, aligned with `daily.dates` |
| `stations[].coverage` | One character per hour. `0`–`5`: archived, with minute stamp = digit × 10. `x`: every candidate URL returned 404 (never published). `?`: due but neither archived nor 404 (missed by the collector). `.`: not expected (station not yet active, or hour still pending) |
| `stations[].illumination` | One character per hour, from the sun's elevation at minute 10: `d` day (≥ 6°), `l` low sun (0–6°), `t` twilight (0 to −12°), `n` night |
| `stations[].latest_obs` | Latest realtime values: waves, wind, temperature and spectral summary, each from its own newest row, and only if under 3 h old |
| `realtime[]`, `offsite` | Realtime-spectra snapshots on disk by date; time of the last successful offsite copy (`data/raw/.offsite_last_sync`) |
| `seastate` | Which stations have `seastate/<id>.json`, and the refresh summary (requests, bytes, failures) |

An hour becomes *due* 50 minutes after it starts. NDBC posts images ~20–30
minutes after their :10 stamp, and the collector runs at :40.

## Running it

Build and view locally. `_site/` is git-ignored.

```bash
uv run wave-analysis dashboard live --out /tmp/live            # needs the local archive; --no-fetch skips NDBC
uv run wave-analysis dashboard build --out _site --live /tmp/live
uv run mkdocs build --strict -d _site/docs
uv run wave-analysis dashboard check --site _site --require-docs
uv run python scripts/serve_dashboard.py                     # http://127.0.0.1:8765/
```

Publish from the collector host. This needs push access to `origin` that works
without interaction, for example an SSH key without a passphrase or a
credential helper that systemd can use:

```bash
scripts/publish_dashboard.sh --dry-run      # build and commit locally, do not push
scripts/install_collectors.sh --dashboard   # hourly timer at minute 52
journalctl --user -u wave-analysis-dashboard -n 20
```

GitHub Pages must be set to **GitHub Actions** as its source (Settings → Pages),
which is a one-time setting.

## Freshness and failure modes

| Symptom on the site | Meaning | What to check |
|---|---|---|
| "Collector late" (last run 80 min–3 h ago) or "Collector stalled" (> 3 h) | The archiver has not run | `systemctl --user list-timers 'wave-analysis-*'`, `journalctl --user -u wave-analysis-buoycam` |
| "data N h ago" grows while the collector is OK | The publisher is not running or cannot push | `journalctl --user -u wave-analysis-dashboard` |
| "(cached copy)" | The browser could not read the `dashboard-data` branch and used the snapshot bundled at the last deploy | GitHub status, or whether the branch exists |
| A red cell or "missed by the collector" | Due hour with neither an image nor a recorded 404 | Downtime longer than the 70 h backfill; the ledger has the details |
| Station without sea state | NDBC publishes no realtime file for it, or the range request failed | `status.json` → `seastate.refresh.failures` |

## Accessibility and design

- Colours follow a validated palette, checked with the data-viz palette
  validator in both themes: the three series colours pass all-pairs colour-vision
  separation, and the illumination and wave-height ramps are single-hue
  ordinal ramps.
- Status is never shown by colour alone: an icon shape and a text label
  go with it. Charts have hover and keyboard read-outs plus a table view.
- Light and dark themes follow the system setting and can be switched in the header.
- Layout works down to phone width. On phones the image viewer starts
  in the six-view split, because a full 2880 × 300 strip would be about 36 px tall.

## Third-party services used by the page

| Service | Used for | Terms |
|---|---|---|
| NOAA NDBC (`www.ndbc.noaa.gov/images/buoycam/`) | Camera images (loaded directly by the viewer's browser) | U.S. Government work, public domain |
| OpenStreetMap tile servers | Street base map | [Tile usage policy](https://operations.osmfoundation.org/policies/tiles/): attribution, light use |
| GEBCO WMS (`wms.gebco.net`) | Bathymetry base map | Attribution to the GEBCO Compilation Group |
| `raw.githubusercontent.com` | Live data from the `dashboard-data` branch | GitHub |

Leaflet 1.9.4 is vendored in `dashboard/assets/vendor/` with its licence and
npm integrity hash. No analytics or cookies are used, and `localStorage` holds
only the theme choice.
