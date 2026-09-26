# ADR 0008: Prospective collection runs continuously; the raw archive lives outside Git with an offsite copy

**Status:** Accepted (2026-09-26)

## Context
NDBC publishes buoy-camera images for about 72 hours and realtime spectra for
45 days. Nobody else archives the images, so anything not collected is lost
for good. The measured rates are about 1,900 images a day (about 35 GB a year) plus
twice-monthly realtime spectral snapshots (about 3 GB a year compressed). The
per-request ledger grows by about 2,000 rows a day. That is too much for Git
(ADR 0006), and it has to survive disk loss.

## Decision
1. Collectors run as systemd user timers from `deploy/systemd/`
   (installed by `scripts/install_collectors.sh`):
   - `wave-analysis ndbc cameras` hourly at minute 40, with a 70-hour
     backfill so downtime shorter than about 3 days loses nothing;
   - `wave-analysis ndbc download --camera-stations --realtime --gzip` on the
     1st and 15th of each month.
2. Image layout is `<station>/<YYYY>/<MM>/<file>.jpg`. The ledger is monthly
   CSV partitions in `_manifests/` inside the archive, not in `data/manifests/`.
   Entries are streamed to the ledger as requests complete.
3. The archive is append-only. It is copied daily to S3-compatible object
   storage (default: Cloudflare R2) with `rclone copy --immutable`, never
   `sync`, so a local deletion or corruption cannot propagate.
4. Frozen, citable releases go to Zenodo (tar shards within Zenodo's
   limits of 50 GB and 100 files per record), as ADR 0006 already requires.

## Consequences
- The raw archive exists in two places (local disk and object storage),
  plus versioned public releases.
- R2 was chosen for zero egress fees: training copies can be pulled to HPC or
  cloud GPUs repeatedly at no cost. The storage cost is small but not zero
  once past the 10 GB free tier (see `docs/operations/data_collection.md`).
- Collection depends on one host. Downtime shorter than the retention
  windows (72 h for images, 45 days for spectra) is recovered automatically;
  longer outages leave permanent gaps, and the ledger records them.
