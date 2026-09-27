# PacIOOS beach cameras (Oahu, 2009–2013)

Registry id `pacioos_beachcam` · module `wave_analysis.sources.pacioos` ·
**v0.2.0** · verified 2026-09-26.

## What it is

Four retired beach cameras operated for the Pacific Islands Ocean Observing
System (PacIOOS) by the University of Hawaii (PI: M. A. Merrifield), published
on the PacIOOS ERDDAP server. Each camera is a tabledap dataset with one row per
image; the images are served from ERDDAP's `files/` tree.

| Dataset | View | Images | Size | Period (UTC) |
|---|---|---:|---:|---|
| `beachcam_003` | Waimea Bay (beach) | 8,935 | 1.85 GB | 2009-02-06 – 2013-10-15 |
| `beachcam_004` | Waimea Bay (offshore) | 8,919 | 1.42 GB | 2009-02-06 – 2013-10-15 |
| `beachcam_001` | Waikiki (oblique) | 15,564 | — | 2009–2013 |
| `beachcam_002` | Waikiki (nadir) | 15,590 | — | 2009–2013 |

Waimea Bay counts and sizes are from the full indexes of 2026-09-26; Waikiki
counts from a query of the same date. Images are hourly JPEGs from about 06:00
to 17:00 local time (e.g. 1624 × 1254 px).

## Why Waimea Bay

The Waimea pair is the project's first **historical** camera + buoy benchmark:
it needs no prospective collection. CDIP 106 (a directional Waverider, also
NDBC 51201) is 6.3 km offshore and was recording through the whole camera
period ([CDIP](cdip.md)). North Shore winter swell spans small to very large
surf, so the range of wave heights is wide. 96.4 % of the image hours of both
cameras have a CDIP 106 sample starting within [t − 30 min, t + 1 h).

The buoy is in 200 m of water and the cameras watch the bay and surf zone.
Waves change as they shoal and break, so buoy H_s describes the incident
offshore sea state, not the wave heights visible in the image. Any result on
this benchmark must state that, and models should be compared with
non-vision baselines that use the same buoy (landscape review §7).

## Facts from the dataset metadata

From `erddap/info/beachcam_003/index.csv`, and checked on the full indexes:

- **Time zones.** `time` is UTC. File names and the `files/` directories use
  Hawaii Standard Time (UTC−10): `beachcam_003.20120105T170000.jpg` is
  2012-01-06 03:00 UTC. All 17,854 Waimea file names are exactly `time` − 10 h.
- **The index time is not the capture time.** "Minutes and seconds always
  rounded to 00:00 in filename. Actual time of photo may occur anytime within
  the specified hour. See timestamp on photo image for more precise time." An
  image labelled 17:00 was taken between 17:00 and 18:00. Pairing must use the
  whole hour, or read the timestamp burned into the image (done: see *Audit and
  capture times*, which also found a period when neither holds). The burned-in
  text must also be masked before training.
- **Sizes.** Each index row has the file size in bytes; the archiver checks
  every download against it.
- **Licence.** "The data may be used and redistributed for free but is not
  intended for legal use, since it may contain inaccuracies." Acknowledge
  PacIOOS. Citation form: "PacIOOS. [year-of-data-download], [Title], [Data
  access URL], accessed [date-of-access]."

## Archive

```bash
uv run wave-analysis pacioos beachcam --index-only     # fetch and check the indexes
uv run wave-analysis pacioos beachcam                  # Waimea pair (default); resumable
uv run wave-analysis pacioos beachcam beachcam_001 --start 2012-01-01 --end 2012-02-01
```

Layout under `data/raw/pacioos/beachcam/`:

| Path | Content |
|---|---|
| `<dataset>/<YYYY>/<MM>/<DD>/<file>.jpg` | Images, mirroring ERDDAP's `files/` tree (HST dates) |
| `_index/<dataset>_<stamp>.csv` | The index as served at each run |
| `_manifests/<dataset>.csv` | Per-request ledger: URL, SHA-256, size, status; a size that differs from the index is noted |

The ledger is kept with the archive, like the buoy-camera ledger (ADR 0008),
and is copied offsite with it. A run skips images already archived with the
ledger's size and URLs recorded as 404, so an interrupted run resumes where it
stopped. Requests are sequential at most one per second; the Waimea pair takes
about 11 hours.

The first full download was started on 2026-09-26 as a one-off user service:

```bash
journalctl --user -u wave-analysis-pacioos-waimea -f       # progress every 500 images
systemd-run --user --unit=wave-analysis-pacioos-waimea \
  --working-directory="$PWD" .venv/bin/wave-analysis pacioos beachcam   # resume after a reboot
```

## Audit and capture times (2026-09-27)

`scripts/waimea_extract.py` then `scripts/waimea_timing_audit.py` (outputs in
`data/interim/waimea_audit/`; `image_times.parquet` has one row per image with
its trusted UTC capture time or none).

**Archive.** 17,689 of the 17,854 indexed images are archived. Each matches its
ledger SHA-256 and the index size, ends with a JPEG EOI marker, and decodes at
1624 × 1254 RGB; no two are identical. The other 165 return HTTP 404 from
PacIOOS: `beachcam_003` 2010-01-12 to 01-22 (112) and `beachcam_004`
2009-12-29 to 2010-01-03 (53).

**Coverage.** The period 2009-02 to 2013-10 has long gaps in both cameras:
2009-04-14 to 09-23 (162 days), 2010-08-12 to 11-03 (83), 2011-02-18 to 10-25
(249), 2013-02-09 to 05-23 (103), and several of 1-6 weeks. The views are
stable: monthly median views move by at most about 7 px (of 1624) with no
re-aiming.

**Caption.** Every image carries `YYYY.MM.DD HH:MM:SS   F: YYYYMMDDTHHMMSS`
(bottom-left, 10 px monospaced text, followed by "C.Kontoes, T.Hilmer and
D.Young, UH"). All 17,689 captions were read by template matching; the two
copies of the time agree within 1 s on every image. There is no EXIF.

**Which time to use.** Normally the caption falls inside the hour named in the
file name: photos at hh:55 (60 %) or, in 2010-2011, hh:23-24. So the file name
is usually 25-55 min early and the caption is the capture time. Not trusted
(638 images, `time_source = uncertain`):

- 23 May - 27 July 2013 (636 images): captions often 1-10 h after the named
  hour, and some daylight images are stamped 19:33-23:37 HST, so the camera
  clock was wrong. The file name is plausible but not verified; excluded.
- 27 Feb 2010, 08:00 files (2 images): stamped about 18:00 (the day of the
  Chile tsunami warning).

**Checks of the caption time** (17,051 trusted images):

- Time zone: read as HST, no image is taken with the sun more than 6 deg below
  the horizon; read as UTC, 13,752 would be. The captions are HST.
- Clock offset: images without any bright region occur only with the sun below
  -3 deg (5 of 1,147 twilight images). Clock shifts from -26 to +4 min explain
  the dark and bright twilight images equally well; larger shifts do not. This
  bound is coarse: auto-exposure keeps images bright to about -3 deg, so few
  images are dark.
- Date: the white-water fraction in the water blocks (chosen from the view)
  correlates with CDIP 106 Hs (Spearman 0.56 for `beachcam_003`, 0.19 for
  `beachcam_004`). Over all years the peak is at +6 h (003) and 0 h (004), and
  the correlation falls within a day; year by year it peaks within half a day
  of zero in 2009, 2010, 2012 and 2013. 2011 (800 images) is inconclusive with
  this crude index (peaks of 0.18 and 0.08). The index resolves dates, not
  hours.
- Hour scale: hour-to-hour changes of the index do not correlate with buoy
  changes (rho about 0.01), so it cannot test timing within hours.
- Visual: the images at the highest buoy Hs (5-6 m) show the bay full of
  breaking waves; those at the lowest (0.7-0.8 m) a glassy bay; also in 2011.

## Next steps

- Pairing table: for each trusted image, every CDIP 106 record overlapping
  its capture time, with offsets; label rules chosen later.
- Record exclusion reasons (uncertain time, no buoy sample, twilight, fog,
  lens drops) in the manifest builder's ledger. Mask the caption band before
  training.
- Near-duplicate and blocked-time splits: consecutive hours of one swell are
  strongly correlated, so splits must be by time block (`datasets.splits`).
