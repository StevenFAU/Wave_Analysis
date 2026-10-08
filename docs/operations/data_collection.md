# Continuous Data Collection and Storage

Some of the data this project needs is **only temporarily public**. This page
covers what is collected continuously, how it runs, how it is stored, and how
to check that it is working. The rationale is in
[ADR 0008](../decisions/0008-prospective-collection-and-storage.md).

## What is collected

| Collector | Schedule | Upstream retention | What it saves |
|---|---|---|---|
| `wave-analysis ndbc cameras` | hourly, minute 40 | ~72 h per image | every new buoy-camera image and the listing JSON; backfills the last 70 h |
| `wave-analysis ndbc download --camera-stations --realtime --gzip` | 1st and 15th of each month, 03:20 | 45-day rolling file | `.txt`, `.spec`, `.data_spec`, `.swdir`, `.swdir2`, `.swr1`, `.swr2` for every camera station |
| `scripts/sync_offsite.sh` (optional) | daily, 04:15 | n/a | append-only exchange of `data/raw/` with object storage: pushes this host's files, pulls other hosts' files; writes `data/raw/.offsite_last_sync` |
| `scripts/publish_dashboard.sh` (optional) | hourly, minute 52 | n/a | archive status and recent sea state for the [dashboard](dashboard.md) (`--dashboard` installs it) |
| `wave-analysis webcoos download … --lookback 3D` (optional) | hourly, minute 25 | multi-year, but downloads of data older than 90 days need WebCOOS's agreement | one still per 30 min (daylight) from the five Yin et al. (2025) cameras (`@yin2025`) and, since 2026-10-08, eleven beachfront cameras within 25 km of a wave buoy (`@near_buoy`); needs the API token ([WebCOOS](../datasets/webcoos.md); `--webcoos` installs it) |

### Measured behaviour of the NDBC camera feed (2026-09-26)

- **Cadence:** one image per camera per hour, **day and night**. Night images
  are mostly dark but are kept (ADR 0001).
- **Minute stamp is not fixed per camera.** Of 5,471 images, 5,313 are stamped
  :10, 156 are stamped :00, and 2 are stamped :50. Some cameras (46089, 46050,
  42056) switch from hour to hour. The backfill tries the latest listed
  image's minute first, then :10 and :00. An earlier version that guessed a
  single minute wrongly marked about 330 existing images as missing. They
  were recovered on the next run.
- **Publication lag:** about 20 minutes (`Last-Modified` 00:31 for a 00:10
  image), hence the minute-40 schedule.
- **Retention:** superseded images stay downloadable at their own URL for
  about 72 hours. The oldest available image was exactly now − 72 h, and older
  requests return 404. The image directory is not listable (HTTP 403), so the
  backfill builds candidate file names from each camera's code and minute
  stamp.
- **Gaps:** about **1 %** of camera-hours are never published. In the first
  70 h backfill, 59 of about 5,530 camera-hours had no image at any minute
  stamp, 27 of them at one camera (44014). Each 404 is recorded in the ledger
  as `not_found` and not requested again.
- **Size:** about 43 KB per image on average. Median by hour is about 21 KB
  at night and 55–65 KB in daylight (Atlantic/Pacific daytime, UTC
  14–23 h).
- **First run (2026-09-26):** 5,392 images archived, covering 2026-09-23
  05:10 to 2026-09-26 03:10 UTC across 79 cameras.

### Volume and cost

| Stream | Per day | Per year |
|---|---|---|
| Camera images (79 cameras × 24 h × ~43 KB) | ~80 MB, ~1,900 files | **~30 GB**, ~690 k files |
| Listing snapshots (24 × 14 KB) | 0.3 MB | 0.1 GB |
| Realtime wave files (91 stations × 7 files, ~7.5 MB per station, gzip) | n/a | **~3 GB** (24 snapshots) |
| Ledger (CSV) | ~0.6 MB | ~0.2 GB |

Local disk is not a constraint (tens of GB a year). The risks are **losing
the only copy** and **the collector host being down longer than the
retention window**.

## Running it

```bash
uv sync                                  # creates .venv with the CLI
scripts/install_collectors.sh            # installs and starts the systemd user timers
loginctl enable-linger "$USER"           # keep timers running while logged out
```

Status and logs:

```bash
systemctl --user list-timers 'wave-analysis-*'
journalctl --user -u wave-analysis-buoycam -n 50
uv run python scripts/buoycam_coverage.py --days 3   # per-station coverage and gaps
```

`scripts/install_collectors.sh --remove` uninstalls. Each run takes a
non-blocking lock (`.archiver.lock`), so overlapping runs (for example a
manual run during the first long backfill) skip instead of competing. Manual
runs are safe at any time:

```bash
uv run wave-analysis ndbc cameras                   # latest + 70 h backfill
uv run wave-analysis ndbc cameras --backfill-hours 0
```

**Recovery.** If the host was down for less than about 3 days, the next run recovers
every image still on the server. If it was down longer, images older than 72 h
are gone. The ledger shows exactly which hours are missing, and the spectra for
those hours can still be downloaded later. The backfill fetches the oldest
hours first, across all cameras, because those are the next to be deleted.
After a long outage, let the first run finish (about 1 s per image, so close
to 1.5 h for a full 70 h backfill) before rebooting or shutting down.

### A second collector host

A second machine running the camera archiver means an image is lost only if
both hosts are down for longer than the backfill window. Each host archives
independently, and the hosts exchange files through the offsite remote
(`scripts/sync_offsite.sh`), so after a sync each holds the union.

- **Ledgers are per host:** `_manifests/<YYYY-MM>.<host>.csv`, with the host
  name from `WAVE_ANALYSIS_HOST` or else the system host name. Ledgers written
  before 2026-10-08 are `<YYYY-MM>.csv`. Readers take every file. An image
  fetched by both hosts has a row in each ledger; the dashboard counts images
  on disk and matches ledger rows to them by file name, but its list of
  collector runs mixes the two hosts.
- **The sync pushes, then pulls.** Images are compared by size, because the
  two hosts fetch the same image at different times and their modification
  times differ. A file with the same name and a different size stops the sync
  with an error. Ledgers are copied with `--update`, which is safe because each
  has a single writer.
- **Single-writer jobs stay on the primary host.** The realtime snapshots
  (`_manifests/realtime.csv`), WebCOOS (`_manifests/<camera>.csv`) and the
  dashboard publisher (the `dashboard-data` branch) each have one writer. The
  second host receives their files through the sync. The dashboard shows the
  second host's images after the next sync on the primary host.

On the second host, after `uv sync` and the R2 setup below:

```bash
scripts/install_collectors.sh --secondary --sync
```

## Storage plan

Three tiers, following the 3-2-1 rule (three copies, two media, one offsite):

1. **Working copy:** `data/raw/` on the collector host. Point
   `WAVE_ANALYSIS_DATA` at another disk to move it.
2. **Offsite copy:** S3-compatible object storage, updated daily by
   `scripts/sync_offsite.sh`. Raw files are pushed and pulled with
   `rclone copy --immutable`, so existing objects are never overwritten on
   either side and deletions are never propagated. The growing ledgers are
   copied with `--update`. The same remote is how collector hosts share
   images ([a second collector host](#a-second-collector-host)).
3. **Citable releases:** frozen snapshots on Zenodo with DOIs (for example,
   quarterly), packed as monthly tar shards because a record allows at most
   100 files and 50 GB. NOAA imagery is a U.S. Government work, so
   redistribution is permitted. Cite NDBC as the originator.

### Choosing the offsite store

| Option | Storage price | Egress | Fit |
|---|---|---|---|
| **Cloudflare R2 (Standard)**, recommended | 10 GB-month free, then $0.015/GB-month; Class A $4.50/M (1 M/month free) | **free** | Egress-free reads suit repeated training pulls to HPC or cloud GPUs. ~60 k uploads a month is well inside the free operations. |
| Backblaze B2 | first 10 GB free, then $6.95/TB-month | free up to 3× stored, then $0.01/GB | Cheaper storage. Egress is capped, which matters only with many full re-downloads. |
| R2 Infrequent Access | $0.01/GB-month, no free tier, $0.01/GB retrieval, 30-day minimum | free | Not worth it at this scale. |
| University storage (FAU Research Computing / HPC) | institutional | n/a | Best place for the **training copy**, next to the GPUs. Ask Research Computing for a project allocation. |

**The R2 free tier alone will not hold this archive.** 10 GB is reached
about 4 months into collection. After that the cost is about $0.45/month per
additional 30 GB: roughly **$1–2 for the first year** and **about $7 for the second**.
Community reports indicate that activating R2 requires a payment method on
file even when usage stays within the free tier.

### Setting up R2 (one-time, done by the account owner)

1. In the Cloudflare dashboard, enable R2 and create a private bucket, for example
   `wave-analysis-raw`.
2. Create an **R2 API token** with *Object Read & Write*, scoped to that bucket.
   Note the access key ID, secret, and account ID.
3. Install rclone (v1.59 or later), then run `rclone config` to create a remote named
   `r2`. The resulting config should look like:

   ```ini
   [r2]
   type = s3
   provider = Cloudflare
   access_key_id = <from step 2>
   secret_access_key = <from step 2>
   endpoint = https://<account_id>.r2.cloudflarestorage.com
   acl = private
   no_check_bucket = true
   ```

   (`no_check_bucket` is required for bucket-scoped tokens.)
4. Tell the sync job where to write, dry-run it, then enable the daily timer:

   ```bash
   mkdir -p ~/.config/wave-analysis
   echo 'WAVE_ANALYSIS_REMOTE=r2:wave-analysis-raw' > ~/.config/wave-analysis/sync.env
   WAVE_ANALYSIS_REMOTE=r2:wave-analysis-raw scripts/sync_offsite.sh --dry-run
   scripts/install_collectors.sh --sync               # primary host
   scripts/install_collectors.sh --secondary --sync   # a second collector host
   ```

Credentials stay in rclone's config (`~/.config/rclone/rclone.conf`) and never
in this repository.

## Incidents

| Period (UTC) | What happened | Effect |
|---|---|---|
| 2026-09-29 20:41 to 2026-10-01 12:18 | No collector run for about 40 h (host down) | Recovered by the 70 h backfill: the 12:18 run took 55 min and archived 3,101 images (219 never published). The dashboard snapshot of 12:52 was taken mid-run and briefly showed the Pacific stations empty for 2026-09-30 |
| 2026-10-01 13:13 to 23:38 | No collector run for about 10 h (host down again) | Recovered by the 23:38 backfill (615 images). Seven-day coverage as of 23:49: 12,874 archived, 83 never published, and 164 gaps, all at 2026-10-01 21:00 and 22:00 UTC (78 stations each). The next two runs (2026-10-02 00:42 and 01:41) filled them: at 01:45 the 3-day window had no unexplained gap. Separately, station 46061's camera stopped publishing on 2026-09-29 |
| 2026-10-04 20:43 to 2026-10-08 00:32 | No collector run for about 76 h (host powered off). The first two recovery runs (00:32 and 01:10) were cut short by a reboot and a shutdown; a run from 03:43 to 04:32 and the new second host (02:10 to 03:50) finished the backfill | **638 camera-hours lost for good:** every station from 2026-10-04 20:00 to 2026-10-05 02:00 (7 h × 78 stations, older than the 70 h window when collection resumed), plus 51 stations at 03:00 and 41 at 04:00. Those 92 were inside the window, but the interrupted runs walked each station newest-first and never reached them; the backfill now fetches the oldest hours first. Dashboard seven-day coverage as of 2026-10-08 12:42: 12,166 archived, 147 never published, 637 gaps |
| Until 2026-10-08 | No offsite copy (`offsite.last_sync` null) | Resolved: the first upload to R2 (11.4 GiB, 58,975 files) finished 2026-10-08 06:12; daily since |

The first two outages were recovered only because each lasted less than the
70 h backfill window; the third was not. See P0 in the
[collection plan](collection_plan.md).

## Alerts

`.github/workflows/collector-watchdog.yml` runs hourly on GitHub, not on a
collector host, because a host that is switched off cannot report it. It
reads `status.json` from the `dashboard-data` branch and opens an issue
labelled `collector-alert` when:

- no buoy-camera archiver run (`collector.last_run`) for more than **3 h**;
- no dashboard status push (`generated_at`) for more than **3 h**;
- no offsite sync (`offsite.last_sync`) for more than **30 h**.

The issue gives the time at which images start to be lost for good (the
newest archived image + 71 h, the edge of the 70 h backfill window). It is
refreshed hourly and closed when the host reports again. GitHub notifies the
repository owner of new issues, by email if notifications are set to email.
Only the primary host publishes the status, so the alert means "the primary
host is quiet", even if a second host is still collecting. Run the check by
hand with:

```bash
git fetch origin dashboard-data && git show FETCH_HEAD:status.json > /tmp/status.json
uv run python -m wave_analysis.dashboard.watchdog /tmp/status.json
```

## Verifying integrity

Every ledger row carries the SHA-256 of the payload as served. Realtime
snapshots stored with `--gzip` are hashed before compression, so verify them
with `sha256_file(path, gunzip=True)`. For the offsite copy,
`rclone check data/raw r2:wave-analysis-raw/raw --one-way` confirms that every
local file exists remotely with the same hash.
