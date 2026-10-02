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

Positions are from the API; distances from the camera-site registry. CDIP 200's
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
inventory returned) is exactly what the 90-day guideline covers. Thinned to one
still per 30 minutes in daylight it is about 8,000–9,000 stills per camera-year
(21–25 a day in the first test) and roughly 4 GB. Before any such
download, email webcoos@secoora.org with the purpose, cameras, period, cadence
and volume, and the acknowledgement that will be used; record their answer in
this document.

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
   include frames.
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
