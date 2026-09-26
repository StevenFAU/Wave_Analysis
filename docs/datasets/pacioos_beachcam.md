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
  whole hour, or read the timestamp burned into the image. The burned-in text
  must also be masked before training.
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

## Next steps

- Pair images with CDIP 106 records by hour window; record the exclusion
  reasons (no buoy sample, night, fog, lens) in the manifest builder's ledger.
- Measure the burned-in timestamp and caption band, and decide whether to read
  capture times from it.
- Near-duplicate and blocked-time splits: consecutive hours of one swell are
  strongly correlated, so splits must be by time block (`datasets.splits`).
