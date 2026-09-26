# Acquisition

Code: `ingest/`, `sources/*`; CLI `wave-analysis ndbc …`.

## Sequence

1. **Discover.** Build an availability inventory indexed by
   *station + product + period*, never by provider as a whole. For NDBC,
   `run_inventory` parses the directory index of each product (one request per
   product), the realtime index, the station metadata, and the camera listing.
   The index pages are saved and checksummed, so the inventory itself is a
   reproducible artefact.
2. **Select.** Choose stations and periods from the inventory (e.g. stations
   with `directional_spectrum_years > 0` and a camera), and record the selection
   in an experiment/config file.
3. **Fetch.** `Downloader.fetch` downloads with a 1 request/s minimum interval
   (provider guidance), exponential back-off on 429/5xx, atomic writes, and
   SHA-256. **Every attempt**, including 404s, becomes a manifest row.
4. **Verify.** Re-fetches compare with the last checksum: `unchanged` or
   `changed`. Revised historical files are detected, not silently absorbed.

## Politeness and identification

Requests send `User-Agent: wave-analysis/<version> (research data pipeline;
+<repository URL>)`, so providers can identify and contact the project. No
personal email is embedded. Bulk jobs run serially at the configured interval.

## What is *not* acquired

- Entire upstream archives. Only the station-periods needed are fetched
  (repository specification §1).
- Anything requiring the acceptance of terms on the user's behalf. Sources
  needing credentials (Copernicus, some APIs) are configured by the user and
  documented in their datasheets.

## Prospective collection

When a provider publishes only "latest" data (NDBC buoy cameras), the archive
must be built forward in time. `wave-analysis ndbc cameras`, scheduled hourly,
downloads only new file names, saves each listing snapshot, and backfills
the last 70 hours from NDBC's ~72-hour retention
(`docs/datasets/ndbc.md#buoy-camera-archiving`). Realtime spectral files cover
45 days, so a second timer snapshots them on the 1st and 15th of each month.
Scheduling, storage, and the offsite copy are described in
`docs/operations/data_collection.md` (ADR 0008).
