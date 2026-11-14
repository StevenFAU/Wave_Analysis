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
  second host's images after the next sync on the primary host. The WebCOOS
  historical download (`webcoos historical`) writes `<camera>.<host>.csv`
  instead, so it can run on either host, though on only one at a time.
- **A host can skip collections in the pull.** The laptop has no room for
  the WebCOOS history (about 325 GB), so its `sync.env` has
  `WAVE_ANALYSIS_SYNC_PULL_SKIP="webcoos webcoos_timex webcoos_brt"`. It still
  pushes any files of its own in them.
- **A collection can live on another drive.** Make its directory under
  `data/raw/` a symlink; the sync follows symlinks in both directions
  (`--copy-links`). If the drive is not mounted the sync fails, so the
  dashboard's offsite status goes stale instead of the collection being
  skipped silently. On the desktop, `webcoos`, `webcoos_timex` and
  `webcoos_brt` have been links to a second drive since 2026-10-12
  (`/mnt/wavedata/wave_analysis/`, a 1 TB SSD on a USB adapter, mounted from
  `/etc/fstab` with `nofail`). Unplugging it stops the hourly WebCOOS
  collector, the historical download and the sync until it is mounted again.
  Ledgers record files under a linked collection as `data/raw/...`, like all
  others.

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
   either side and deletions are never propagated. Before pushing, every
   file written since the last successful exchange is checked against its
   ledger's SHA-256, and damaged files are held back
   ([verifying integrity](#verifying-integrity)). A file that differs from
   its offsite copy no longer stops the exchange: the others are still
   copied, and the exchange fails at the end with the list and the command
   that settles it. The growing ledgers are
   copied with `--update`, from a copy made just before the upload: rclone
   rejects a file that changes in transit, and a collector may be writing
   its ledger at that moment (the WebCOOS historical download does for
   days). The same remote is how collector hosts share images
   ([a second collector host](#a-second-collector-host)).
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
| 2026-10-13 08:16 to 08:46 | The daily exchange uploaded the images, then failed at the ledgers: the WebCOOS historical download was adding a row to its ledger every two seconds, and rclone rejects a file that changes during its upload | That ledger had no offsite copy, the second host's files were not pulled and `offsite.last_sync` stayed at 2026-10-12 08:32. Fixed the same day: ledgers are uploaded from a copy made just before (the ledger reached R2 at 14:16) |
| 2026-10-18 about 14:08 | Unclean reset of the desktop (the journal stops without a shutdown) during the WebCOOS historical download | The last ~30 s of writes to the data drive were lost: the oakisland_west stills ledger ended in 1,882 NUL bytes and 27 stills were empty files. Repaired the same day: the ledger was cut at the first NUL, the stills were fetched again, and every ledger row written since 10-17 12:00 was checked against its file (notes in `/mnt/wavedata/wave_analysis/_crash_2026-10-18/`) |
| 2026-11-02 about 22:13 | Unclean reset of the laptop | 28 buoy-camera images of 2026-10-31 05:00 to 06:10 left as empty files on the laptop; R2 holds good copies from the desktop. The laptop's exchange fails on them. **Open:** to be settled on the laptop with `wave-analysis archive repair` (see [after an unclean shutdown](#after-an-unclean-shutdown)) |
| 2026-11-04 02:41 and 04:44 (11-03 21:41 and 23:44 EST) | Two unclean resets of the desktop, with nothing logged before either | The brt ledger of oakisland_west ended in 2,160 NUL bytes; 9 brt stills of 2025-03-25/26 were damaged (8 empty, 1 the right size but all NUL bytes); 20 buoy-camera images of 11-04 01:10 and a listing were empty. All were fetched again and checked on 11-04, but the exchange of 11-04 09:20 had already uploaded 29 damaged copies, and the exchange never overwrites a file: every exchange from 11-05 to 11-10 stopped on the 28 that differed in size, before the ledgers and the pull (`offsite.last_sync` stuck at 11-04 10:05; watchdog issue #15). The NUL-filled still had the right size, so no size comparison could see it. **2026-11-10:** the whole archive (629,917 files, 294 GB) checked against its ledgers: nothing damaged locally; R2 compared by MD5: exactly those 29 objects damaged, overwritten with the checked local copies. Cause: downloads and ledger rows were not synced to disk before counting as written. Fixed the same day: they are synced, empty files are fetched again, and the exchange checks new files against their ledgers and holds back damaged ones |
| 2026-11-11 19:34:58 (14:34:58 EST) to 2026-11-13 21:17 | The data drive's USB link dropped mid-write (a plain `USB disconnect` with no UAS error or reset before it: unplugged, or the enclosure lost power), and the desktop was shut down normally 3 min later. The drive came back unmounted (`/etc/fstab` mounts it only at boot) | The ext4 journal aborted; the September sample download died with an I/O error; hourly WebCOOS collection failed for 50 h on the dangling links; the exchanges of 11-12 failed, one after 5.5 h of trying to pull the WebCOOS files back through those links (watchdog issue #16). **2026-11-13:** file system checked (not clean), repaired with `e2fsck -f -y` and checked clean through udisks, then mounted; every file written since the last exchange matches its ledger (20,001 WebCOOS files, 14.9 GB, and 4,188 on the internal disk); the last still and its ledger row were written 2 s before the drop. Nothing lost: the hourly collector's 3-day look-back covers the gap, and the sample resumed. The exchange now stops at once when a linked collection's drive is not mounted (commit `fdff589`). To unplug the drive: stop the WebCOOS timers, `udisksctl unmount -b /dev/disk/by-label/wavedata`; to bring it back: `udisksctl mount -b /dev/disk/by-label/wavedata` (asks for the password) |
| 2026-11-14 16:12:14 to 16:13 | The desktop's GPU overheated during a timing test and its driver powered the desktop off (`GPU over temperature range(SW CTF)`; an orderly shutdown, file systems synced) | Collectors down about a minute; the timers resumed at boot. The WebCOOS September sample (time exposures, at oakisland_west) and the ERA5 download for the buoy cameras were restarted by hand and resume where they stopped. GPU training is on hold until the card's cooling is checked ([desktop GPU](cluster.md#training-on-the-desktop-gpu-rocm)) |
| 2026-11-14 17:20 and 17:27 | Two hard resets of the desktop: it froze each time about 25 s after connecting to a university VPN with OpenVPN 2.7, whose data-channel offload (the kernel's `ovpn` module, kernel 7.0.0-43) locked up a CPU core (`soft lockup` in `ovpn_tcp_send_skb`). Found on the next connection: a route the VPN pushes covers the VPN server itself, and the local profile ignored the instruction that normally brings a direct route to the server, so the VPN's own packets looped into its tunnel (629 sent, 0 received). The profile now sets `disable-dco` and a direct route to the server (`route remote_host 255.255.255.255 net_gateway`) | `archive verify --since last-exchange` afterwards: 15,290 files ok, 1 empty upstream, all 54 ledgers without bad rows or NUL bytes; 3 empty `.partial-*` files of interrupted WebCOOS downloads removed. The WebCOOS September sample was restarted; the collectors resumed with their timers |

The first two outages were recovered only because each lasted less than the
70 h backfill window; the third was not. See P0 in the
[collection plan](collection_plan.md). The resets of 2026-10-18 and 2026-11-03
left no trace in the journal (no machine-check, thermal, disk or GPU message,
no crash dump), which points to the power supply; see
[after an unclean shutdown](#after-an-unclean-shutdown).

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

Every ledger row carries the SHA-256 and size of the payload as served. The
whole archive is checked against them with:

```bash
.venv/bin/wave-analysis archive verify                        # every file (about 15 min for 270 GB)
.venv/bin/wave-analysis archive verify --since last-exchange  # files written since the last exchange
.venv/bin/wave-analysis archive verify --collections webcoos_brt
```

It follows the links to collections on other drives, reads every ledger
(`_manifests/*.csv` and `data/manifests/raw/*.csv`) and writes a report to
`data/interim/archive_verify/<UTC time>/`: `problems.csv` (every path that is
not `ok`), `files.csv.gz` (every file with its SHA-256 and MD5), `ledgers.csv`
and `summary.txt`. It exits with 1 if a file or a ledger is damaged. Statuses:

| Status | Meaning | Action |
|---|---|---|
| `ok` | size and SHA-256 match a ledger row for this path | none |
| `empty_upstream` | 0 bytes, and the ledger records that the source served 0 bytes | none: kept as evidence (some WebCOOS stills are empty on WebCOOS's side); not usable as an image |
| `empty`, `size_mismatch`, `sha256_mismatch`, `unreadable` | the bytes are not the bytes that were downloaded | `archive repair` |
| `suspect` | no ledger row, and the file is empty or ends in NUL bytes | fetch again or remove by hand |
| `no_ledger` | no ledger row (the run lists under `_listings/` are records made here, by design) | none |
| `missing` | a ledger row names the file, but it is not on disk | `archive repair` |
| `partial` | a temporary file left by an interrupted download | delete it |

Realtime snapshots stored with `--gzip` are compared after decompression, as
their ledger rows describe the payload as served (`sha256_file(path,
gunzip=True)` by hand). A file passes if it matches any row for its path: two
hosts may each have fetched it, and a re-fetch adds a row.

**The offsite copy.** The report's MD5s are the ETags that R2 reports for
single-part uploads, so the offsite copy is compared without downloading it:

```bash
.venv/bin/wave-analysis archive compare-remote data/interim/archive_verify/<UTC time>/
```

It writes `remote_compare.csv` and `repair_candidates.txt` (paths whose
offsite size or MD5 differs) next to the report. It replaces
`rclone check data/raw r2:wave-analysis-raw/raw --one-way --copy-links`,
which would read the 270 GB a second time. On the laptop, pass
`--skip webcoos webcoos_timex webcoos_brt` for the collections it does not
pull.

**Settling a difference.** The ledgers decide which copy is right:

```bash
.venv/bin/wave-analysis archive repair --from <list> --dry-run   # see the decision for each path
.venv/bin/wave-analysis archive repair --from <list>             # then do it
.venv/bin/wave-analysis archive repair <path> --refetch          # neither copy is right: ask the source again
```

For each path it downloads the offsite object to a temporary folder, checks
both copies against the ledger, and then uploads the local file over the
offsite object (`push`), replaces the local file with the offsite one
(`pull`), or, with `--refetch`, downloads the file again from its ledger URL
and keeps it only if it matches the ledger. When both copies match a ledger
row but differ (an upstream revision, e.g. an ERA5 month re-requested when it
went from preliminary to final), the copy matching the newest row wins. Buoy-camera images older than
about 72 h cannot be fetched again; WebCOOS keeps its archive. When neither
copy is right and the source no longer has the bytes, the file is reported as
`lost` and left as it is (ADR 0001: annotate, don't delete). Every decision is
appended to `data/interim/archive_repair/log.csv`. Uploads are read back and
checked before they count as done.

## After an unclean shutdown

A power cut or hard reset (the journal of the previous boot stops without a
shutdown: `journalctl --list-boots`, then `journalctl -b -1 -n 20`) can leave
files that were written in the last seconds empty, or the right size and full
of NUL bytes. Since 2026-11-10 downloads and ledger rows are synced to disk
before they count as written, so this should no longer happen; the steps below
check that it did not.

1. **Mount the data drive** if it is not (`ls /mnt/wavedata/wave_analysis`;
   if empty, run `sudo mount /mnt/wavedata` in a system terminal, outside the
   Claude app).
2. **Check the files written around the reset** (time in UTC, a little before
   the reset):

   ```bash
   .venv/bin/wave-analysis archive verify --since 2026-11-04T02:00:00Z
   ```

   Look at `DAMAGED` lines and at `damaged ledger(s)` in the summary. Delete
   `partial` files.
3. **A ledger with NUL bytes:** copy it aside unchanged (keep it as found),
   then cut it at the first NUL byte, e.g.
   `python -c "import sys; p=sys.argv[1]; d=open(p,'rb').read(); open(p,'wb').write(d[:d.index(b'\0')])" <ledger>`.
   Rows lost with the tail make their files `no_ledger`; the next run of that
   collector fetches them again.
4. **Damaged files:** `archive repair --from <list> --dry-run`, then without
   `--dry-run` (add `--refetch` where neither copy is right). Buoy-camera
   images can be fetched again for about 72 h after they were taken, so do
   this first.
5. **Run the exchange** (`systemctl --user start wave-analysis-offsite-sync`)
   and check that `data/raw/.offsite_last_sync` is updated. The exchange
   checks every file written since the last successful exchange anyway: a
   damaged file is held back and named in its log, and a file that differs
   from its offsite copy is listed in `data/interim/offsite/conflicts.txt`
   with the `archive repair` command to run.
6. Add a row to [Incidents](#incidents).

Three resets in three weeks with nothing in the journal before them (no
machine-check, thermal, disk or GPU message, and no crash dump) point to the
power supply rather than to software. A UPS for the desktop would turn a power
cut into a clean shutdown.
