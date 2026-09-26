# NDBC test fixtures

Real upstream files from NOAA NDBC (U.S. Government data, public domain),
**trimmed** to keep the repository small. They are used by the offline test
suite so that parsers are tested against genuine formats rather than
hand-written imitations.

Retrieved **2026-09-25/26 (UTC)** from `https://www.ndbc.noaa.gov/`.

## historical/

Annual files from `https://www.ndbc.noaa.gov/data/historical/<dir>/<name>`,
decompressed, trimmed, and re-gzipped (`gzip -n`):

| File | Trimming |
|---|---|
| `41010h2023.txt.gz` | 2 header lines + all rows 2023-03-01 … 03-05 |
| `41010{w,d,i,j,k}2023.txt.gz` | 1 header line + all rows 2023-03-01 … 03-05 |
| `41010h1995/2003/2006.txt.gz`, `41010w2003.txt.gz` | first 40 lines (legacy header eras) |
| `41001a2010.txt.gz` (adcp), `32st1o2010.txt.gz` (ocean) | first 40 lines |

41010's 2023 wave record covers January–April only; March was chosen for dense
half-hourly spectra.

| Original file | SHA-256 of original (as downloaded) |
|---|---|
| `41010h1995.txt.gz` | `0924029c6f8ff0b355e4cb83d85f21559d9b0ad9d1be22cd2b29c6ff1d650dbb` |
| `41010h2003.txt.gz` | `d0a2294e16e40b4685dee7b84669f6821b2bf175115f820185c20601154269e6` |
| `41010h2006.txt.gz` | `e30f84365fd71cec9b5864d9dcb1ed0a607364b76f4722ad895569345f8d94ff` |
| `41010h2023.txt.gz` | `8d1aedcdd151d9d0b8f3b489f486a577dce93655bf7cc57566f66d9036277420` |
| `41010w2003.txt.gz` | `f517a138e7b41ee2c42eac19a5eff5691ea32641b6a9c78df42753b35e12fc6b` |
| `41010w2023.txt.gz` | `bff1f617b37ee25c3e60d42d8e017e3f40ee611902fae25205650162bb167562` |
| `41010d2023.txt.gz` | `da5d8c4c1ecc25c3bb199c32569df16214ad9abe274c3aca51cf2cf49571bf51` |
| `41010i2023.txt.gz` | `fd9fee921a94412360b4d9ea960a16f62056d3b16ccdba2123bb988e16ba2543` |
| `41010j2023.txt.gz` | `b80ed4593516f2c45a3d20dc01886c0adc93e66c2cf2e7330d8ca40251654090` |
| `41010k2023.txt.gz` | `4faee15f254255f7f0a053831785d950108e89617bead9f2f33fa48fd9d98d3b` |
| `41001a2010.txt.gz` | `a40f7d9c7948f32ab8f58d974138b81cc84afd9ac882b05c45d2dfdbf1dfde4e` |
| `32st1o2010.txt.gz` | `c331ee3f1254316eeb9e16a9ede129f8cf77ff7bdc6cb01f447e3964555568be` |

## realtime/

`https://www.ndbc.noaa.gov/data/realtime2/41010.{txt,spec,data_spec,swdir,swdir2,swr1,swr2}`,
first 40 lines each (realtime files list the newest record first).

## index/

| File | Source | Trimming |
|---|---|---|
| `historical_stdmet.html` | `/data/historical/stdmet/` | non-row lines kept; file rows kept only for stations 41009, 41010, 4cONF, eb01 (1970–71) |
| `realtime2.html` | `/data/realtime2/` | rows for 41009, 41010, FPSN7 |
| `activestations.xml` | `/activestations.xml` | stations 41001, 41009, 41010, fpsn7, 32st1 |
| `station_table.txt` | `/data/stations/station_table.txt` | header + 41001, 41009, 41010, fpsn7 |
| `buoycams.json` | `/buoycams.php` | complete (91 cameras) |

To refresh: re-download, apply the same trimming, and update this table.
