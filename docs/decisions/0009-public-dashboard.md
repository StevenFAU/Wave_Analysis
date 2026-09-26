# ADR 0009: A static public dashboard, with live data published to a single-commit branch

**Status:** Accepted (2026-09-26)

## Context
The collectors (ADR 0008) run on one host, and their state (coverage, gaps,
runs, archive size) was visible only through `journalctl` and scripts. The
project's other outputs (the dataset registry, camera-site pairings,
bibliography and verification levels) were spread over YAML, BibTeX and
Markdown. A browsable view was requested on GitHub Pages.

Constraints:

- GitHub Pages serves static files. CI cannot see the archive, which lives
  only on the collector host.
- The repository is public, so whatever is published must not leak host
  details or credentials.
- NDBC sends no CORS headers, so a browser cannot read its data files.
  It can display its images with `<img>`.
- Committing hourly status to `main` would add about 8,800 commits a year of
  generated noise.

## Decision
1. **Static front end, no build step.** The site is plain HTML, CSS and ES
   modules in `dashboard/`. Leaflet is vendored with its integrity hash, so
   the site does not depend on a CDN. It reads only JSON.
2. **Two data products.**
   - The *catalog* is built in CI from committed files: registries,
     bibliography, the verification index, and docs.
   - The *live data* is built on the collector host: archive status, and sea
     state taken from the head of NDBC realtime files with HTTP range
     requests.

   Both are strict JSON with a schema tag. The build fails on dangling
   references.
3. **Live data on a single-commit branch.** `scripts/publish_dashboard.sh`
   commits the live files with git plumbing, as a commit with no parent, and
   force-pushes it to `dashboard-data` every hour. The browser reads that
   branch through `raw.githubusercontent.com`, which is CORS-enabled and
   cached for about 5 min. The Pages build bundles the branch's current
   content as a fallback.
4. **No image hosting.** The viewer shows images from NDBC's server for the
   ~72 h NDBC keeps them. The archive stays private until the Zenodo
   releases (ADR 0008).
5. **Pages via GitHub Actions** (`site.yml`). CI builds the dashboard and
   MkDocs (under `/docs/`), runs the front-end unit tests, and checks the
   built site before it deploys: assets, module imports, data schemas, the
   manifest hashes, and every docs page the catalog links to. Pull requests
   build and check without deploying.

## Consequences
- Live data can be at most about one hour old and needs no server. When the
  host is down, the site keeps showing the last published state, and its
  freshness indicator says how old that state is.
- `main` stays free of generated files. The data branch has no history, and
  the ledger remains the historical record.
- The host needs non-interactive push access, which is a credential on the
  collector machine. The publisher is installed separately
  (`install_collectors.sh --dashboard`) so that it can be left off.
- The dashboard adds about 200 small range requests an hour to NDBC (about
  6 KB each). The first run fetches about 20 MB. These requests are not in
  the research ledger, because the values are for display only.
- Map tiles depend on the OpenStreetMap and GEBCO public services. CARTO
  basemaps now require an API key, so they were not used.
