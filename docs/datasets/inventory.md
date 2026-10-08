# Data Inventory

What the project holds, what each collection is for, and what every field or
variable in it means, followed by an explanation of where buoy wave numbers
come from. The dashboard's
[Data view](https://stevenfau.github.io/Wave_Analysis/#/data) shows the same
inventory with **live counts, sizes and periods**, checked hourly against each
collection's ledger. This page does not repeat those numbers, because they
change every hour.

The tables on this page are generated from
[`data/registry/data_dictionary.yaml`](https://github.com/StevenFAU/Wave_Analysis/blob/main/data/registry/data_dictionary.yaml)
at build time (`scripts/mkdocs_data_dictionary.py`). The dashboard reads the
same file, so the two cannot disagree. To change a description, edit the YAML.
The build checks it against the dataset registry, the documentation, and the
project's variable vocabulary (`src/wave_analysis/schemas/variables.py`).

## The six roles

Every collection does one or more of these jobs.

<!-- dictionary:roles -->

## All collections

<!-- dictionary:overview -->

!!! warning "One collection cannot be downloaded again"
    NDBC deletes buoy-camera images after about 72 hours, so the project's
    archive is the only copy of older images. Two hosts collect it, and a daily
    offsite copy goes to object storage
    ([data collection](../operations/data_collection.md),
    [collection plan](../operations/collection_plan.md), P0).

## Collections in detail

A `→ name` after a meaning gives the variable's key in the project's
standardized tables. The NDBC, CDIP and ERA5 adapters map to these keys, and a
test checks this page's keys against their maps.

<!-- dictionary:collections -->

## Tables we built

<!-- dictionary:tables -->

## Same quantity, different names

<!-- dictionary:crosswalk -->

## Where buoy wave numbers are made

NDBC buoys do part of the work on board and NDBC does the rest on shore. The
buoy turns its motion into a spectrum and sends only that. The shore unpacks
the spectrum, checks it, computes the wave numbers and publishes them. The
motion record itself never leaves the buoy.

**On the buoy** (sensors and wave processor):

1. The buoy records its own motion for 20 minutes (40 on one system):
   vertical acceleration, and on directional buoys pitch, roll and heading.
   The sampling rate is 1.50, 2.00, 2.56 or 1.7066 readings a second,
   depending on the system.
2. Pitch, roll and heading become east-west and north-south slopes of the
   sea surface.
3. A Fourier transform turns the motion over time into energy per frequency.
4. The energy is averaged into frequency bands, and the direction values
   (α₁, α₂, r₁, r₂) are computed from the cross-spectra.
5. The result is packed small enough for one satellite message, with summary
   statistics of pitch, roll and heading for quality checks.

**Satellite link.** The message is short. NDBC's processing document gives
message length as the reason: transmitting the raw time series "is not
feasible", so "much of the data analysis is performed onboard" (Earle 2003).

**On shore** (NDBC, Stennis Space Center, Mississippi):

1. The message is decoded and checked: complete, values in range, hull
   behaving.
2. It is unpacked back into a wave-height spectrum.
3. WVHT, DPD, APD, MWD, the swell and wind-wave split, and steepness are
   computed from it (formulas in the [NDBC FAQ](https://www.ndbc.noaa.gov/faq/wavecalc.shtml)).
4. Further automated and manual quality control is applied.

**Public files** (what the project downloads): the standard `.txt` and the
`.spec` summary, five spectral files, as 45-day realtime files and as yearly
archives.

!!! note "Never sent to shore"
    The moment-by-moment motion record. NDBC's current FAQ
    ([wave.shtml](https://www.ndbc.noaa.gov/faq/wave.shtml)) says that the raw
    acceleration or displacement measurements are "not transmitted shore-side."

### How the on-board share grew

NDBC Technical Document 03-01 (Earle 2003) describes each generation of wave
system step by step. As buoy computers got faster, more of the work moved on
board:

| System | First used | The buoy computes and sends | The shore computes |
|---|---|---|---|
| GSBP Wave Data Analyzer | 1979 | 75 numbers describing how the acceleration repeats over time, plus its mean, minimum and maximum | The spectrum itself, all corrections, all wave numbers |
| DACT and VEEP Wave Analyzers | 1984–1991 | The acceleration spectrum (and, on directional systems, cross-spectra) | Noise and hull corrections, conversion to a wave-height spectrum, directions, wave numbers |
| WPM and DWPM wave modules | 1989 onward | Spectra, cross-spectra, hull corrections, the direction values, plus quality-control statistics | Unpacking, quality checks, wave numbers, archiving |

The spectra downloaded for buoy 41002 use the WPM band layout (46 bands,
0.0325–0.485 Hz). TD 03-01 covers systems up to 2003. NDBC's current payloads
are newer and not described in it, but NDBC's FAQ still says the raw motion is
not transmitted. One detail is not fully clear: section 3.3.8 says the WPM
corrects for hull response on board, while its step table lists the final
conversion after transmission. That conversion appears to undo the packing
step.

### Raw vs processed

**Raw** is every reading in time order. A 20-minute record at 1.7066 readings
a second has 2,048 readings each of vertical acceleration, pitch, roll and
heading, about 8,200 numbers. From it you can tell what the buoy was doing at
any second.

**Processed** is a summary of the whole record: energy and direction in 46
frequency bands (46 × 5 = 230 numbers), plus WVHT, DPD, APD and MWD. It says
how much energy there was at each wave period over those 20 minutes, but no
longer *when* anything happened. A real example, buoy 41002, record of
1 Oct 2026 11:50 UTC:

```text
WVHT 0.9 m   DPD 9 s   APD 5.3 s   MWD 51°
0.078 Hz  S 0.342 m²/Hz
0.083 Hz  S 0.488 m²/Hz
…  43 more bands
```

An analogy: the processed record is a song's equalizer display, showing how
loud each pitch was over the whole song. The raw record is the song itself.
The equalizer cannot tell you which note played at 1:23.

## What this means for pairing with a camera

"Pairing" covers two different questions. Only one of them needs raw data.

**How big were the waves around this photo? Processed data is enough, and it
is what the project uses.** Wave height, period and direction are statistics
of 20–27 minutes of sea anyway. Each photo gets the buoy records nearest to
it; the Waimea labels average the records within ±30 min
([ADR 0010](../decisions/0010-label-rule-time-averaged-references.md)). What
limits this is time and place, not rawness: a buoy camera shoots at about :10
past the hour while the wave records cover separate 20-minute windows, and a
shore camera looks at water kilometres from its buoy.

**How was the camera tilted at the instant of this photo? This needs raw
data, and NDBC never sends it.** An NDBC camera rides on the buoy. A 10° roll
tilts the horizon 10° and changes where every pixel points. Using that motion
means knowing the pitch and roll at the shutter's second, either to correct the
image geometry or to combine the camera with a motion sensor. The buoy does
send per-record summaries of pitch and roll for quality checks, but they are
in none of the public files the project collects, and a 20-minute summary
would not give the angle at one second anyway.

Ways around it:

- **Measure the tilt from the picture.** The horizon in each of the six views
  shows how the buoy was tilted when the photo was taken
  ([research question Q-M5](../literature/research_gaps.md)).
- **A published record with the camera's motion.** The Schwendeman & Thomson
  (2017) ship stereo dataset logs the cameras' own motion with an IMU, next to
  drifting-buoy wave spectra ([research datasets](external_research_datasets.md)).
- **CDIP's raw displacement.** CDIP publishes Waverider motion at about 1.28
  samples per second, which gives wave-by-wave statistics such as the single
  highest wave. A Waverider carries no camera, though, so it cannot give a
  camera's tilt.
- **A camera and motion sensor on one clock.** The only route to
  self-collected image and motion pairs (gap G-9 in the
  [collection plan](../operations/collection_plan.md)).

## Sources

- Earle, M. D. (2003). *Nondirectional and Directional Wave Data Analysis
  Procedures*. NDBC Technical Document 03-01, <https://www.ndbc.noaa.gov/wavemeas.pdf>.
  Read in full on 2026-10-01 ([source verification](../literature/source_verification.md)).
- NDBC FAQ: [how spectral wave data are derived](https://www.ndbc.noaa.gov/faq/wave.shtml)
  and [how wave parameters are calculated](https://www.ndbc.noaa.gov/faq/wavecalc.shtml),
  read 2026-10-08.
- CDIP 106 historic file attributes (variable names and units), read from the
  file on 2026-10-08. Image sizes, EXIF fields, spectral bands and reporting
  of each NDBC column were read from the files on disk on the same day.
- The datasheets linked from each collection above.
