# Vendored libraries

Third-party code is copied here (not loaded from a CDN) so that the dashboard
works without depending on another service being available at runtime.

| Library | Version | Files | Licence | Source and integrity |
|---|---|---|---|---|
| Leaflet | 1.9.4 | `leaflet-1.9.4/leaflet.js`, `leaflet.css`, `images/layers*.png` | BSD-2-Clause (`leaflet-1.9.4/LICENSE`) | npm `leaflet@1.9.4`, `dist/`; tarball integrity `sha512-nxS1ynzJOmOlHp+iL3FyWqK89GtNL8U8rvlMOsQdTTssxZwCXh8N2NB3GDQOL+YR3XnWyZAxwQixURb+FA74PA==` (verified with `npm pack`, 2026-09-26) |

To update: `npm pack leaflet@<version>`, check that the integrity matches
`npm view leaflet@<version> dist.integrity`, copy the same files, and update
this table and the paths in `index.html`.
