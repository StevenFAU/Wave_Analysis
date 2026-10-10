# WebCOOS — Webcam Coastal Observation System

Registry id `webcoos` · module `wave_analysis.sources.webcoos` · **v0.2.0** ·
API verified 2026-09-26.

## Why it matters

WebCOOS (led by SECOORA, funded by NOAA IOOS) runs coastal webcams with
archived one-minute stills and video. The five cameras of the **Yin et al.
(2025)** wave-breaking dataset, the held-out test set of Kamagata et al. (2026),
are WebCOOS cameras. WebCOOS can therefore give instrument labels to the same
views used in the literature, which those papers lacked.

| Camera | Place | Stills (inventory, 2026-09-26) | Nearest wave buoy |
|---|---|---|---|
| `currituck_hampton_inn` | Corolla, NC (36.3779, −75.8258) | 2022-01-04 → present; 1.11 M, 692 GB | CDIP 433 / 44056, 22.2 km |
| `jennette_north` | Jennette's Pier, Nags Head, NC (35.9123, −75.5965) | inventory timed out (HTTP 504); none in the last day | CDIP 243 / 44086, 18.6 km |
| `jennette_south` | Jennette's Pier, Nags Head, NC (35.9063, −75.5942) | 2022-10-03 → 2026-09-16; 1.01 M, 634 GB | CDIP 243 / 44086, 18.6 km |
| `oakisland_east` | Oak Island, NC (33.9126, −78.2243) | 2022-01-01 → present; 977 k, 295 GB | CDIP 200 / 41108, 28.9 km |
| `oakisland_west` | Oak Island, NC (33.9126, −78.2243) | 2021-12-03 → present; 921 k, 289 GB | CDIP 200 / 41108, 28.9 km |

Positions are from the API; distances from the camera-site registry.
`jennette_north` has had no stills since 2026-09-16 and is listed as archived;
`jennette_south` resumed by 2026-10-08.

**Beyond these five (2026-10-08).** WebCOOS now lists 86 cameras, 41 of them
within 25 km of a buoy that publishes directional spectra, several with
multi-year archives (for example `currituck_sailfish` from 2022,
`masonboro_inlet` from 2023, and `cocoabeach`, Florida, 2023-05 to 2024-09
beside CDIP 143). See the [data source scan](../literature/data_source_scan_2026-10.md#2-webcoos-beyond-the-five-yin-et-al-cameras-q).

**Collected since 2026-10-08 (`@near_buoy`).** Eleven beachfront cameras whose
midday view shows open surf, chosen from those within 25 km of a buoy that
publishes directional spectra (`NEAR_BUOY_CAMERAS` in
`sources/webcoos.py`, with each buoy and distance):

| Camera | Buoy, km | Camera | Buoy, km |
|---|---|---|---|
| `san_elijo` | CDIP 262 (46274), 5.8 | `crescent_city` | NDBC 46027, 19.8 |
| `westport_north`, `westport_south` | CDIP 036 (46211), 9.5 | `oceancity` | CDIP 263 (44084), 19.7 |
| `masonboro_inlet` | CDIP 150 (41110), 10.2 | `pgtalabama_south`, `pgtalabama_west` | NDBC 42012, 23.3 |
| `currituck_sailfish` | NDBC 44056 (Duck FRF), 17.8 | `stinson` | NDBC 46026, 23.8 |
| | | `vabeach_hamptonos` | CDIP 147 (44099), 23.8 |

Left out after looking at a midday still: `oceanshores_east`/`_west` (inside
Grays Harbor), `kahului_road`/`_harbor` (Kahului Harbor), `ferrybeach_north`
(river mouth behind a jetty) and `campellis_south` (harbour). The
`pgtalabama_*` stills carry a burned-in time overlay at the top left; mask it.
The hourly collector runs `webcoos download @yin2025 @near_buoy --lookback 3D`;
the last 85 days of the new cameras were backfilled once, inside the 90-day
guideline. CDIP 200's
historic record ends on 2026-07-20 and it had no realtime file on 2026-09-26,
so recent Oak Island images have no nearby buoy until it is back. Stills exist
in daylight only (about 300–700 a day per camera).

## Access

An API token is required (HTTP 403 without it). Request one from WebCOOS, then
save it where the code looks for it, without it appearing on screen or in the
shell history:

```bash
mkdir -p -m 700 ~/.config/wave-analysis && (umask 077 && read -rsp "WebCOOS token: " t && printf '%s\n' "$t" > ~/.config/wave-analysis/webcoos_token && unset t && echo " saved")
```

`$WEBCOOS_API_TOKEN` overrides the file. The code refuses a token file that
others can read, sends the token only to `app.webcoos.org`, and never writes
it to a log, ledger or listing.

API base `https://app.webcoos.org/webcoos/api/v1`, header `Authorization: Token <key>`:

| Endpoint | Content |
|---|---|
| `/assets/` | Cameras (paginated). Each has feeds → products (`one-minute-stills`, `video-archive`, …) → services (e.g. `oakisland_west-one-minute-stills-s3`) |
| `/services/<service>/inventory/` | Daily bins with image counts and bytes. Slow: minutes, and HTTP 504 for some cameras |
| `/elements/?service=…&starting_after=…&starting_before=…&page_size=1000` | Image files: capture time to the second (`data.extents.temporal.min`), file name, size, and a URL on a public S3 bucket. `page_size` is honoured (default 100) |

Image files are downloaded **without** the token, from S3.

## Usage guidelines

From <https://webcoos.org/usage-guidelines> (read 2026-09-26):

- Use must be lawful, ethical and responsible. WebCOOS "reserve[s] the right to
  request removal or restrict use of imagery".
- "Download only a reasonable amount of data." Access may be banned for use
  that strains their systems.
- **"If you would like to download a large amount of data that is greater than
  90 days old, please contact webcoos@secoora.org first."** The downloader
  enforces this: if more than 500 selected images are older than 90 days, it
  stops unless run with `--historical-approved`, which should only be used once
  WebCOOS has agreed.
- **Citation** in research publications: "SECOORA is acknowledged for leading
  the WebCOOS program, which collected [images of interest] in the WebCOOS
  dataset in this study," citing WebCOOS as "WebCOOS [year(s) of imagery]".
- Redistribution of images is not addressed: ask before publishing images or
  releasing them in a dataset.

## Collecting

```bash
uv run wave-analysis webcoos cameras --state "North Carolina"
uv run wave-analysis webcoos inventory currituck_hampton_inn              # slow
uv run wave-analysis webcoos download currituck_hampton_inn --lookback 7D --list-only
uv run wave-analysis webcoos download currituck_hampton_inn oakisland_west --lookback 7D
scripts/install_collectors.sh --webcoos      # hourly, the five Yin cameras, 3-day look-back
```

`download` lists a camera's stills, keeps the one nearest each point of a time
grid (`--every 30min`, within `--tolerance 5min`), and downloads those not yet
archived. Pairing uses each image's exact capture time; the grid only thins
one-minute stills to the cadence of the wave records.

Layout under `data/raw/webcoos/`:

| Path | Content |
|---|---|
| `<camera>/<YYYY>/<MM>/<DD>/<file>.jpg` | Stills by UTC capture date, with WebCOOS's file names (e.g. `oakisland_west-2026-09-25-150048Z.jpg`) |
| `_listings/<camera>_<stamp>.csv` | The elements selected by each run: capture time, name, URL, size, grid time, offset |
| `_manifests/<camera>.csv` | Per-request ledger: URL, SHA-256, size, status; sizes are checked against the API |

The hourly collector keeps each request small and the data recent (well inside
the 90-day window). It started on 2026-09-26, so earlier years need the
historical request below.

## Historical data

The 2021–2026 archive (4.0 million stills and 1.9 TB for the four cameras whose
inventory returned) is exactly what the 90-day guideline covers, so we asked
WebCOOS first.

**WebCOOS's answer (Josh Rhoades, email of 2026-10-09).** Bulk requests
should go to their **on-premise archive**, not the API or S3:
`https://stage-ams.srv.axds.co/archive/jpg/<group>/<camera>/<YYYY>/<MM>/<DD>/*.jpg`.
They gave a `wget --mirror` example limited to `--wait=1s --limit-rate=20m`,
and asked to be told when the transfer starts so they can monitor it. What
the WebCOOS/SECOORA acknowledgement should look like was passed to two
colleagues; record their answer here when it comes.

| Camera | Group | Years in the archive (2026-10-09) |
|---|---|---|
| `currituck_hampton_inn` | `noaa` | 2022–2026 |
| `currituck_sailfish` | `noaa` | 2022–2026 |
| `jennette_north` | `noaa` | 2022–2026 |
| `jennette_south` | `noaa` | 2022–2026 |
| `oakisland_east` | `uncw` | 2021–2026 |
| `oakisland_west` | `uncw` | 2021–2026 |
| `masonboro_inlet` | `uncw` | 2023–2026 |
| `cocoabeach` | `uncw` | 2023–2024 |

The archive is an nginx directory listing with exact sizes. It holds the same
files as S3: the 15 stills the collector had for 2026-09-21 at
`currituck_hampton_inn` have the same names and sizes there, and one checked
has the same SHA-256. File names carry the capture time to the second (equal
to the API's capture time for all 43,999 stills listed up to 2026-10-09).

Other products under `/archive/<product>/<group>/<camera>/`, one day at
`currituck_hampton_inn` (2026-09-21) for scale:

| Product | Content | That day | Years |
|---|---|---|---|
| `jpg` | stills, every 1–2 min in daylight | 369 files, 176 MB (0.48 MB each) | 2021/2023 → |
| `timex` | time exposure (mean) of one 10-min video clip (`timex.<camera>-<stamp>Z.jpg`) | 41 files, 20 MB (0.50 MB) | 2024 → (`currituck_hampton_inn` from 2024-03-13; cocoabeach 2024–2025) |
| `brt` | brightest value of each pixel over the same clip | 41 files, 36 MB (0.89 MB) | 2024 → (cocoabeach 2024–2025) |
| `mp4` | video clips, mostly 10 min | 102 files, 5.2 GB (51 MB) | not checked |
| `15-second-timex` | — | none that day | not checked |

`annotation/` and `jsonl/` were not examined. The cameras record from about
sunrise to sunset (stills 07:09–19:09 EDT, video from 06:48, on 2026-09-21);
there are no night images to collect.

**What `timex` and `brt` are** (worked out from the archive, 2026-10-09; the
files carry no metadata):

- **They are made from one video clip each.** On 2026-09-21 all 41 `timex`
  and `brt` time stamps equal the start of an `mp4` clip. Only the full clips
  have them: the next clip starts 9.8–10.6 min later, while the 60 clips
  without them are 2–10 min apart (median 4.9). A full clip is 600.0 s at
  20 fps, 12,000 frames (from the `mp4` header). So a `timex` or `brt`
  covers the **10 minutes starting at its time stamp**, and its listing time
  (about 11 minutes later) is when it was written.
- **`brt` is the brightest value of each pixel over the clip, `timex` its
  time average.** At `currituck_hampton_inn` on 2024-03-14, `brt` is at
  least as bright as the `timex` of the same clip at 100 % of pixels (two
  clips; JPEG tolerance 8 levels), but only at 86 % against the `timex` of a
  clip 30 minutes later. `timex` is smoother than a still (mean
  neighbouring-pixel difference 4.9 vs 7.3 grey levels) at the same mean
  brightness (103); `brt` is brighter (122) and sharper (9.1), as breaking
  crests would make it. A maximum and a high percentile cannot be told apart
  this way.
- **The stills are frames of the same video**, extracted with ffmpeg (their
  JPEG comment is `Lavc58.134.100`), at the same 2688×1520 size.

### What we collect, and why

One image per buoy record, of each product, for every year and camera; then a
small sample at full cadence to test whether more images per record help:

| Part | Command (`--product`, `--every`) | Size |
|---|---|---|
| Stills every 30 min, all years, 8 cameras | `stills`, `30min` | ~150 GB, ~290,000 files |
| `timex` and `brt` every 30 min, 2024 on | `timex` / `brt`, `30min` | ~150–200 GB, ~215,000 files |
| Test sample: every `timex`/`brt` for one month, 8 cameras; every still for one month at `currituck_hampton_inn` and `jennette_south` | `--every all --start … --end …` | ~25 GB, ~40,000 files |

About 325–375 GB and ten days at WebCOOS's pace (the `timex` files of March
2024 are 2688×1520 and 0.7–0.9 MB, larger than in 2026). The reasons:

- **Labels, not images, limit training.** A wave record is a 1600-s statistic
  every 30 min. More images inside a record share its label: they are not
  independent examples, and they cannot beat the label's own noise
  (consecutive records differ by a median 4 %; one record's 90 % CI is
  −10/+15 %, [wave parameters](../methodology/wave_parameters.md)).
- **What is scarce is conditions and places.** Large waves are rare (27 days
  with H_s ≥ 4 m in five Waimea years), and models fail on new sites
  (Kamagata et al. 2026: held-out R² = 0.107). Years and cameras address that;
  extra frames per half hour do not.
- **Time exposures are a different input, not more of the same.** A still is
  one instant of a 27-minute statistic. `timex` averages 10 minutes and `brt`
  keeps where waves broke, so they integrate over many waves as the buoy does.
  No machine-learning study in the [landscape review](../literature/landscape_review.md)
  uses them as input. Whether they beat stills is an experiment: all three
  products exist for 2024–2026, so they can be compared on the same labels,
  cameras and splits.
- **Density can be bought later.** The sample measures whether averaging
  several images per record helps. If it does, a rerun with `--every 10min`
  (or `all`) fetches only the images not yet held, since the 30-minute grid
  times are part of the denser grid. The archive stays at WebCOOS.

Not collected: every still (2.5–4 TB, months at this pace, mostly
near-duplicates), every `timex`/`brt` (another ~300 GB of the same labels,
until the sample says otherwise), and video in bulk (~5 GB per camera-day,
about 60 TB; see the pilot below). The sample month is chosen from the buoy
records so that it spans calm and storm conditions.

```bash
# after telling WebCOOS that the transfer starts (stills, then timex, then brt):
systemd-run --user --collect --unit wave-analysis-webcoos-historical \
    --working-directory="$PWD" "$PWD/scripts/webcoos_historical.sh"
journalctl --user -u wave-analysis-webcoos-historical -f    # one line per camera-month
# one product only: add --setenv=WEBCOOS_PRODUCTS=stills (or "timex brt") to systemd-run
# the sample, afterwards, the same way, e.g.:
#   .venv/bin/wave-analysis webcoos historical <cameras> --product timex --every all \
#       --start 2025-01-01 --end 2025-02-01 --historical-approved
```

A transient unit stops at a reboot (and at logout without lingering); rerun
the same command to resume.

`historical` does not mirror whole days. It lists each day directory, keeps one
image per 30 minutes as `download` does (`--every`, `--tolerance`, `--offset`;
grid times are split by month, each getting one image; `--every all` keeps
every image), skips images already archived (same name and size, e.g. fetched
from S3 by the hourly collector), and fetches the rest. Stills go to
`data/raw/webcoos/`, the other products to `data/raw/webcoos_timex/` and
`data/raw/webcoos_brt/` with the same layout.

- **Pace:** one request at a time; at least `--wait` (1 s, the minimum) after
  each response before the next request; each file at no more than 20 MiB/s.
- **Resumable:** rerun the same command. Each whole past month's listing is
  saved once to `_listings/onprem/<camera>/<YYYY-MM>.csv` (with `listed_at`)
  and reused, so a rerun does not list those days again (`--relist` does).
- **Ledger:** `_manifests/<camera>.<host>.csv`, separate from the hourly
  collector's `<camera>.csv`, so every ledger has one writer when hosts exchange
  the archive.
- **One run at a time:** a lock next to the product roots
  (`data/raw/.webcoos-historical.lock`) keeps a second historical run, of any
  product, from opening a second connection. The hourly collector has its
  own lock and keeps running.
- `--list-only` lists and reports what would be fetched; `--limit N` stops
  after N new stills per camera.
- **Disk guard:** a run stops (exit code 4) before a camera-month when the
  archive's drive has less than `--min-free-gb` free (default 50), so it
  cannot fill the disk the collectors write to.
- **Another drive:** a product root may be a symlink to a directory on
  another drive (e.g. `data/raw/webcoos_timex`). The offsite exchange follows
  it; a run fails, rather than writing elsewhere, if that drive is not
  mounted.

**Volume.** Up to 2026-10-09 the hourly collector kept 21–27 stills a day per
camera at 0.26–0.90 MB each (mean by camera). The eight cameras have about 33
camera-years of stills (about 22 of `timex`/`brt`). At about 1.4 s a file
(1 s wait plus transfer, measured 2026-10-09) the plan above takes about ten
days, plus about nine hours of listings. A first test (2022-01-10 to
2022-01-12, `currituck_hampton_inn`) listed 364 stills, kept 12 on the grid
and fetched 3 as expected.

## Video archive (pilot, not yet collected)

Every Yin et al. (2025) camera has a `video-archive` product next to its
stills (`/assets/`). Video is what the stills cannot give:

- **Wave motion.** Periods are 5–15 s; stills one minute apart cannot resolve
  them. At 5–10 fps, timestacks, celerity and dispersion give period and
  direction physically (landscape review §2.1), and temporal models have
  access to the information that helped in every study that tested it.
- **The literature's own test views.** Kamagata et al. (2026) tested on
  clips from these cameras without instrument labels. Paired with CDIP
  records, the same views become an instrument-referenced test.

Plan:

1. **Inventory first** (no downloads): list the `video-archive` elements for
   one week per camera and record clip length, frame rate, resolution,
   bitrate, and how far back the archive goes (Q-D9).
2. **Ask WebCOOS** in the same email as the historical stills request:
   purpose, cameras, clip cadence, volume, and whether derived datasets may
   include frames. The on-premise archive has an `mp4/` tree next to `jpg/`
   (seen 2026-10-09); ask whether the pilot should use it.
3. **Pilot** on two cameras whose reference buoy is online:
   `currituck_hampton_inn` (CDIP 433) and `jennette_south` (CDIP 243;
   buoy-pair check: CDIP 243 and Oregon Inlet, 29 km apart on the same coast,
   differ by a median 7 %, `docs/methodology/synchronization.md`). Not Oak
   Island while CDIP 200 is offline. One clip of 2–5 min starting inside each
   CDIP record (every 30 min) in daylight, for 2–4 weeks, within the 90-day
   window.
4. **Budget from the pilot.** As a rough guide (assumed 4–8 Mbit/s), a 5-min
   clip is 150–300 MB, so 20 clips a day from 2 cameras is 6–12 GB a day or
   2–4 TB a year. Decide cadence and retention from the measured numbers.

Pairing uses `pair_interval_overlap`: a clip inside a 1600-s CDIP sample has
overlap fraction 1.

## Open questions

- Frame geometry and calibration per camera (none published in the API).
- Whether WebCOOS permits releasing derived datasets that include the images.
- The Oak Island reference buoy: CDIP 200 has no data after 2026-07-20.
