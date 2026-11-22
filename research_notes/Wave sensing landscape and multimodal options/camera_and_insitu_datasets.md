# Camera imagery with co-located wave truth, and richer in-situ wave labels: sources not yet held (checked 2026-11-19)

**How to read these notes.** Every finding carries its URL and an evidence code:
**P** = read today (2026-11-19) on the provider's own page or server (THREDDS/OPeNDAP metadata, data-release page, repository page); **S** = seen only in a web-search result summary, record not opened; **R** = taken from this project's own files (`docs/literature/data_source_scan_2026-10.md`, `data/registry/datasets.yaml`, `docs/literature/papers/*.md`), not re-checked; **C** = computed by me from provider metadata. Nothing was downloaded: only catalog listings and file headers (DDS/DAS) were read.

**Two passes, same day.** Pass 1 had five web searches before the shared budget ran out and relied on direct provider fetches (deep on USACE FRF, CDIP, USGS, AODN, IFREMER). Pass 2, with search restored, ran about 40 searches on the thin areas: regional camera programmes, platforms with motion sensors, stereo/drone/flume sets, synthetic data and labelled image sets. Many pass-2 items are **S** only, because several publisher and library pages refuse automated fetches (MDPI, PMC, Springer/Nature, AMS, UCSD Library, Sofar's Notion pages; NCEI and one usgs.gov page timed out). Open an S item by hand before citing or acquiring it. Nothing is filled from memory.

---

## 1. USACE Field Research Facility (Duck, NC): what is openly available

### Takeaway
The FRF THREDDS server openly serves eleven years (2015-10 to 2026) of hourly **surf-zone sea-surface elevation time series** from the dune lidar (7 Hz for 30 min at five virtual gauges, plus the runup line), together with the 8-m array, AWACs, Waveriders and altimeters; a USGS release adds 5 Hz radar water-level series at three points on the pier (2022-2024). Wave-resolving Argus video or pixel timestacks are **not** on the server; the only Duck Argus imagery there is rectified time-exposure and variance images from the BathyDuck project (2015-10 to 2016-08), which the earlier project scan missed.

### Cited Findings
**Server layout and access**
- Top level of `frf/`: `argus_temp` (empty), `geomorphology`, `meteorology`, `oceanography`, `projects`, `sediment`, `survey_temp`. No login. (P) — [FRF THREDDS catalog](https://chldata.erdc.dren.mil/thredds/catalog/frf/catalog.html)
- `argus_temp/` lists no datasets and no subfolders. (P) — [argus_temp](https://chldata.erdc.dren.mil/thredds/catalog/frf/argus_temp/catalog.html)
- The FRF data portal front page did not answer on 2026-11-19 (connection reset twice); THREDDS on `chldata.erdc.dren.mil` was up. (P) — [FRF data portal](https://frfdataportal.erdc.dren.mil/)
- Licence attribute on the lidar files: "These data may be redistributed and used without restriction. Data are intended for scholarly use by the research community, with the express agreement that users will properly acknowledge the USACE Field Research Facility and the supporting investigator(s). Use or reproduction of these data for commercial purposes is prohibited without prior written permission." (P) — [lidarWaveGauge080 2026-05 header](https://chldata.erdc.dren.mil/thredds/dodsC/frf/oceanography/waves/lidarWaveGauge080/2026/FRF-ocean_waves_lidarWaveGauge080_202605.nc.html)

**Lidar wave products (raw elevation time series)**
- `lidarWaveGauge080`, `090`, `100`, `110`, `140`: year folders 2015 to 2026, monthly NetCDF; the first file is 2015-10. (P) — [waves catalog](https://chldata.erdc.dren.mil/thredds/catalog/frf/oceanography/waves/catalog.html); [lidarWaveGauge080/2015](https://chldata.erdc.dren.mil/thredds/catalog/frf/oceanography/waves/lidarWaveGauge080/2015/catalog.html)
- Each file holds `waterLevelTimeSeries[time][tsTime = 12601]` (m, NAVD88), hourly records (741 in 2026-05). `tsTime` runs 0, 0.1429, 0.2857 ... 1800.0 s, so **7 Hz for 30 min each hour**. Also per record: spectrum (48 bands), `waveHs` (0.04-0.5 Hz), `waveHsIG`, `waveTm`, `waveTp`, `waveHsTotal`, min/max level, QC flags, `percentTimeSeriesMissing`. (P) — [gauge080 header](https://chldata.erdc.dren.mil/thredds/dodsC/frf/oceanography/waves/lidarWaveGauge080/2026/FRF-ocean_waves_lidarWaveGauge080_202605.nc.html)
- Gauge 080 sits at xFRF = 80 m, yFRF = 944 m (36.18591 N, 75.75211 W), "40 meters down-line from LiDAR scanner"; instrument "Riegl VZ-1000 or VZ-2000 terrestrial laser scanner" on the "North lidar tower". Gauge 140 is 100 m down-line. (P) — same header; [gauge140 2015-10 header](https://chldata.erdc.dren.mil/thredds/dodsC/frf/oceanography/waves/lidarWaveGauge140/2015/FRF-ocean_waves_lidarWaveGauge140_201510.nc.html)
- The QC text of these files speaks of "2 Hz" completeness thresholds although the stored series is at 7 Hz; treat gaps via `percentTimeSeriesMissing`. (P) — same header
- `lidarWaveRunup` (2015-2026): per hour, the runup line as `elevation`, `xFRF`, `yFRF` time series (12,601 samples) at yFRF 945 m, plus 2 % exceedance `totalWaterLevel`. QC flag 1 = "manually digitized", 2 = "determined using an ML model and automated quality standards were met". (P) — [lidarWaveRunup 2026-05 header](https://chldata.erdc.dren.mil/thredds/dodsC/frf/oceanography/waves/lidarWaveRunup/2026/FRF-ocean_waves_lidarWaveRunup_202605.nc.html)
- `lidarHydrodynamics` (2015-2026): the same statistics on a cross-shore transect of 1,551 points (water level, min/max, spectra, `waveHs`, `waveHsIG`, `waveTm` per point and hour). (P) — [lidarHydrodynamics 2026-05 header](https://chldata.erdc.dren.mil/thredds/dodsC/frf/oceanography/waves/lidarHydrodynamics/2026/FRF-ocean_waves_lidarHydrodynamics_202605.nc.html)
- `projects/WaveShape/lidarWaveScans`: 30 files of single-wave Velodyne lidar scans from the CRAB, September-October 2017 (e.g. 114 frames × 15,626 points). (P) — [WaveShape lidar scans](https://chldata.erdc.dren.mil/thredds/catalog/frf/projects/WaveShape/lidarWaveScans/catalog.html)

**Pressure, acoustic and buoy wave gauges**
- Folders under `oceanography/waves` and the year folders seen: `8m-array` 1990-2012 and 2015-2026 (directional spectrum 62 frequencies × 72 directions, a1/b1); `waverider-17m` 1996-2026; `waverider-26m` 2008-2026; `awac-11m` 2008-2023 and 2025-2026; `awac-6m` 2008-2023; `awac-4.5m` 2016-2024; `adop-3.5m` 2015-2018; `xp100m` 2015-2017; `xp340m` 2021-2026 (341-band spectra); `paros940-200` 2018-2025; `sig940-300` 2020-2026. Others present, years not read: `DWG-BB02/03`, `DWG-OC02/03`, `adop-2m`, `awac-5m`, `awac-8m`, `awac-jpier-11m`, `paros-340x-940y-top`, `paros940-250`, `sig769-300`, `sig940-400/600`, `waverider-17m-1D`, `waverider-20m-1d`, `xp125m/150m/200m`. (P) — [waves catalog](https://chldata.erdc.dren.mil/thredds/catalog/frf/oceanography/waves/catalog.html); [8m-array 2026-05 header](https://chldata.erdc.dren.mil/thredds/dodsC/frf/oceanography/waves/8m-array/2026/FRF-ocean_waves_8m-array_202605.nc.html)
- These standing gauge products hold spectra and bulk parameters, not raw series (checked on `xp340m` and `8m-array`). (P) — [xp340m 2026-05 header](https://chldata.erdc.dren.mil/thredds/dodsC/frf/oceanography/waves/xp340m/2026/FRF-ocean_waves_xp340m_202605.nc.html)
- `oceanography/waveSurfTs` does hold raw series but **only for October 2015**: `adop-3.5m`, `awac-6m`, `awac-11m`, `xp100m`, `xp125m`, `xp150m`, `xp200m`. Each has `surfaceTimeseries[time ≈ 642][4096]` and `pressureTimeseries`, "2 Hz Sea Surface Displacement from NAVD88", 34-min records hourly, surface-corrected with linear theory. (P) — [waveSurfTs awac-11m 2015-10 header](https://chldata.erdc.dren.mil/thredds/dodsC/frf/oceanography/waveSurfTs/awac-11m/2015/FRF-ocean_waveSurfTs_awac-11m_201510.nc.html)
- Altimeters at 15 stations (`Alt769/861/940` at 150-350 m cross-shore); DEMs `clarisDEM`, `duneLidarDEM`, `pierLidarDEM` (2019-2022), `surveyDEM`; `duneLidarTransect` 2015-2026; survey CSVs back to 1974 in `survey_temp`. (P) — [geomorphology](https://chldata.erdc.dren.mil/thredds/catalog/frf/geomorphology/catalog.html)
- `projects/Publications/2025/FRF_DataPaper` holds `8m-array`, `linear-array` (1988-1990), `waverider-17m`, `waverider-17m-1D`, `waverider-20m-1D`, `wind_derived`: a re-issued long record prepared for a data paper. (P) — [FRF_DataPaper folder](https://chldata.erdc.dren.mil/thredds/catalog/frf/projects/Publications/2025/FRF_DataPaper/catalog.html)

**USGS radar on the FRF pier (5 Hz water level)**
- "Oceanographic Radar Time Series Measurements Collected at Duck, North Carolina", DOI 10.5066/P1NLPWRP, published 2026-03-17. Three sites on the pier (yFRF 524 m): x = 600 m from 2022-08-18 (monthly files 2022-08 to 2024-12; zips 456 MB, 1.6 GB, 1.4 GB), x = 300 m and x = 200 m from 2024-01-18 to 2024-04-23 (428 MB, 462 MB). Each gives "5 hertz (Hz) time series data of distance measurements ... and associated water level elevations" (NAVD88) and hourly wave parameters; "Data collection is ongoing". (P) — [USGS data release P1NLPWRP](https://cmgds.marine.usgs.gov/data-releases/datarelease/10.5066-P1NLPWRP/)

**Imagery on the FRF server**
- BathyDuck Argus **timex**: 11 monthly files 2015-10 to 2016-08; the 2015-10 file is `Ip[time = 835][x = 334][y = 1334][rgb = 3]`, a rectified image grid. **Variance** images for 2015-10, -11, -12 and 2016-05. cBathy bathymetry from 2015-08. (P) — [BathyDuck timex](https://chldata.erdc.dren.mil/thredds/catalog/frf/projects/bathyduck/data/argus/timex/catalog.html); [variance](https://chldata.erdc.dren.mil/thredds/catalog/frf/projects/bathyduck/data/argus/variance/catalog.html)
- BathyDuck `timeseries_new`: raw pressure `p` (2.9 M samples) and velocity `u, v, w` for gauges g11, g12, g13, g84 in 2015-10. (P) — [BathyDuck g11 header](https://chldata.erdc.dren.mil/thredds/dodsC/frf/projects/bathyduck/data/timeseries_new/BathyDuck-ocean_timeseries_g11_201510.nc.html)
- `projects/argusNorthhead`: merged rectified products every 30 min for June 2018 (North Head, not Duck). (P) — [argusNorthhead](https://chldata.erdc.dren.mil/thredds/catalog/frf/projects/argusNorthhead/catalog.html)
- `projects/SandyDuck/images`: weekly photo folders from 1997, plus the experiment database. (P) — [SandyDuck images](https://chldata.erdc.dren.mil/thredds/catalog/frf/projects/SandyDuck/images/catalog.html)
- Other project folders present: `CurrituckSound`, `DareCounty`, `Delilah`, `Dunex`, `Wilmington`, `duck94`, `specialProducts` (8-m array infragravity). (P) — [projects](https://chldata.erdc.dren.mil/thredds/catalog/frf/projects/catalog.html)

- A 2015 BathyDuck README on the same server describes the Argus collections of that time: pixel time series "sample and save the 2Hz time variability for 2048 samples (17+ minutes), but only for specific requested pixels", kept on the Oregon State Coastal Imaging Lab FTP site by site/year/camera/day. (S: quoted from a search summary of the README, file not opened) — [CIL collections README](https://chldata.erdc.dren.mil/thredds/fileServer/frf/projects/bathyduck/cBathy/cBathySupportMaterial/CILCollectionsREADME093015.docx)
- That FTP host no longer resolves. (R) — `data/registry/datasets.yaml`, entry `usace_frf`

**Distances**
- Lidar gauge 080 to CDIP 433 / NDBC 44056 (36.19835 N, 75.71659 W in the 2023-2025 deployment file): 3.5 km. (C) — [433p1_d13 header](https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/433p1/433p1_d13.nc.html)
- The project's own table puts the Currituck cameras 17.8 km (Sailfish) and 22.2 km (Hampton Inn) from 44056. (R) — `docs/literature/data_source_scan_2026-10.md` §2

### Inferences
- The FRF lidar gauges are the only open, multi-year record found of **individual waves in the surf zone on the US East Coast**: about 8,700 thirty-minute records a year per gauge, enough to build wave-by-wave height, crest-time and maximum-wave labels.
- They become camera labels only with imagery of the same water. The lidar transect (yFRF ≈ 945 m) and the pier radar (yFRF 524 m) lie inside the view of the FRF Argus tower, so a request to the FRF for Argus pixel timestacks or clips for 2015-2026 would yield the "instrument inside the field of view over long periods" set the project lacks. That imagery is not public (empty `argus_temp`), so this is a request, not a download.
- Usable today without any request: BathyDuck timex + variance images (2015-10 to 2016-08) against the lidar gauges, 8-m array and the October 2015 raw gauge series. Time-exposure images do not resolve single waves, so this supports bulk H_s/T_p and breaker-zone width only.
- The project's Currituck cameras are too far alongshore (about 18-22 km) for the FRF surf-zone gauges to be their truth; the FRF record is a separate site, not a relabel of existing clips.

### Gaps
- Whether the FRF will release Argus video, snapshots or pixel timestacks, and for which years: not stated anywhere I could read. The portal front page was down.
- The earlier search summary said the FRF Oceanographic Observations dataset is described in ERDC/CHL TR-18-11 and that a PANGAEA collection (CC BY 4.0, 1980-2018) exists; I did not open either. (S) — [PANGAEA 902954](https://doi.pangaea.de/10.1594/PANGAEA.902954)
- I found no journal data-descriptor for the 2025 "FRF_DataPaper" folder.
- A pass-2 search for current (2024-2026) access procedures for Duck Argus video or pixel time series found none; the 2015 README is the only description, so 2 Hz pixel stacks existed for chosen pixels but their present location is unknown.
- Year coverage of the gauge folders I did not open (`DWG-*`, `sig769`, `paros-340x`, `awac-5m/8m`) is unread.

---

## 2. USGS CoastCam network and USGS releases with imagery plus wave truth

### Takeaway
USGS has published **2 Hz runup timestacks** (a cross-shore line of pixels sampled for 10-17 min each daylight hour) for six sites, the largest being Madeira Beach (6,329 files, 2017-2022) and Sand Key (1,404 files), both on the calm Gulf coast, and Tres Palmas, Puerto Rico (331 files), which lies about 3.5 km from a CDIP Waverider that archives raw displacement. Pass 2 found that Madeira Beach also has a USGS ADCP and pressure loggers about 1.7 km from the camera (CC0; 2021-03 to 2023-09 released, the record back to 2017 promised), making it the best calm-water pair. Raw video is collected at every site, but no video file was seen in any release.

### Cited Findings
- The CMGDS data-release catalog returns 21 records for the keyword "CoastCam". (P) — [USGS CMGDS data releases](https://cmgds.marine.usgs.gov/data-releases/)
- **Madeira Beach, FL** timestacks, DOI 10.5066/P93VZION: "A total of 6,329 netCDF files"; camera "programmed to collect three-band (RGB) images at 2 Hertz (Hz) for a total of approximately 17 minutes (2048 images), every hour during daylight hours"; 2017-01-21 to 2022-01-25; netCDF with time, pixel intensities and local/world coordinates; monthly zips. Use constraints: "Public domain data from the U.S. Government are freely redistributable with proper metadata and source attribution." The abstract says the camera "collected raw video and produced snapshots and time-averaged image products". (P) — [Madeira timestack metadata](https://cmgds.marine.usgs.gov/catalog/spcmsc/madbeach_c1_timestacks_metadata.html)
- **Sand Key, FL** camera 2 timestacks, DOI 10.5066/P13XR9TY: 1,404 netCDF files, same 2 Hz × 17 min scheme, 2018-05-29 to 2021-11-05, public domain. (P) — [Sand Key timestack metadata](https://cmgds.marine.usgs.gov/catalog/spcmsc/sandkey_c2_timestacks_metadata.html)
- **Madeira Beach in-situ waves** (pass 2): "Oceanographic Time Series Measurements Collected at Madeira Beach, Florida", DOI 10.5066/P9HZKIWB. Site MB2, "1.9 kilometers from the shoreline" in 5.6-5.8 m depth (27.78897 N, 82.81229 W): RDI Sentinel-V ADCPs (velocity, pressure, derived wave parameters; `...Awvs.nc` files up to 1.1 GB) and RBR pressure loggers in burst or continuous mode, 2021-03-10 to 2023-09-26. "ADCP data collection at site MB2 has been ongoing since 2017 and will be added in future releases." Site MB1, 30.9 km offshore: Spotter buoys 2021-08-06 to 2023-10-06, including displacement files (`...Sdisp.nc`, 0.5-1.2 GB). Zips: ADCP 2.6 GB, Spotter 4.9 GB. "These data are marked with a Creative Common CC0 1.0 Universal License." The purpose text links the instruments to "video camera monitoring". (P) — [release page](https://cmgds.marine.usgs.gov/data-releases/datarelease/10.5066-P9HZKIWB/); [ADCP metadata](https://cmgds.marine.usgs.gov/catalog/spcmsc/MadeiraBeach_ADCP_metadata.txt); [pressure metadata](https://cmgds.marine.usgs.gov/catalog/spcmsc/MadeiraBeach_Pressure_metadata.txt); [Spotter metadata](https://cmgds.marine.usgs.gov/catalog/spcmsc/MadeiraBeach_Spotter_metadata.txt)
- The release page and the metadata disagree on the MB1 depth (8.3 m against 21.0 m). (P) — same pages
- Madeira camera footprint centre (27.7971 N, 82.7977 W) to MB2: 1.7 km. The timestack release (to 2022-01-25) and the MB2 release (from 2021-03-10) overlap for about ten months. (C) — [Madeira timestack metadata](https://cmgds.marine.usgs.gov/catalog/spcmsc/madbeach_c1_timestacks_metadata.html)
- **DUNEX, Pea Island NC**, DOI 10.5066/P14EWDST: two cameras on 7 m masts, 2021-09-18 to 2021-10-24; "both cameras recording at 1 Hertz (Hz) for 5 minutes" before each hour for stereo, and "one camera recording at 2 Hz for 17 min" on the hour; 411 timestack files (190 + 221). Only timestacks and coordinates are released. Marked CC0 on usgs.gov. Calibration: DOI 10.5066/P1GDP4HR. (P) — [DUNEX timestacks](https://coastal.er.usgs.gov/data-release/doi-P14EWDST/); [usgs.gov record](https://www.usgs.gov/data/usgs-coastcam-dunex-timestack-imagery-and-coordinate-data)
- USGS's Pea Island page lists an offshore waverider, a bottom "nanopod", surf-zone current and wave sensors and a cross-shore pressure array, but gives no data-release links. The catalog keyword "DUNEX" returns only topography/bathymetry (10.5066/P9DPZZG2) and grain-size releases. (P) — [DUNEX Pea Island experiment](https://www.usgs.gov/centers/whcmsc/science/dunex-pea-island-experiment); [CMGDS catalog](https://cmgds.marine.usgs.gov/data-releases/)
- **Tres Palmas, Rincón PR**, DOI 10.5066/P13QYBXP: 331 netCDF timestacks, 2 Hz for about 10 min each daylight hour, 2019-09-09 to 2020-03-10, 50 MB; "Approved for public release. Distribution is unlimited." Camera footprint about 18.348 N, 67.263 W. (P) — [Tres Palmas timestacks](https://coastal.er.usgs.gov/data-release/doi-P13QYBXP/)
- CDIP 181 (Rincón) at about 18.375 N, 67.281 W has raw displacement in deployments d11 (2019-06-26 to 2019-09-14) and d12 (2019-09-16 to 2020-08-04), covering the whole camera period. Camera-to-buoy distance about 3.5 km. (P for the files, C for the distance) — [CDIP 181 archive](https://thredds.cdip.ucsd.edu/thredds/catalog/cdip/archive/181p1/catalog.html)
- **Waiakāne, Molokaʻi HI**, DOI 10.5066/P14QO72F: 30 timestacks, 2018-08-05 to 2018-08-16, 102.7 MB. **Isla Verde, PR**, DOI 10.5066/P13MJ66F: 15 timestacks, February 2019, 11 MB. (P) — [Molokaʻi](https://coastal.er.usgs.gov/data-release/doi-P14QO72F/); [Isla Verde](https://coastal.er.usgs.gov/data-release/doi-P13MJ66F/)
- **Thermal**: "USGS Thermal Tower Reef Camera Timestack, Papaloloa, Ofu, American Samoa", DOI 10.5066/P19NRBZS: radiometric thermal snapshots every minute, 2020-02-03 to 2020-08-01, on ScienceBase (a reef-temperature study, no wave instrument named). (P) — [ScienceBase item](https://www.sciencebase.gov/catalog/item/667f3becd34e84b99fe5c15e)
- Other catalog records (titles only read): Head of the Meadow 2019-2024 imagery and calibration (10.5066/P138GAKI, 10.5066/P95C0DGM, both 2026); Marconi 2021/2023/2024 (10.5066/P17OF3UP, 10.5066/P13UW5LB); Dorado 2022-2023 calibration (10.5066/P13CYA5A); Isla Verde 2022-2023 calibration (10.5066/P13DOBTM); Sand Key calibration (10.5066/P146YVZF); Madeira imagery and calibration (10.5066/P9SGRTF1, already in the registry). (P) — [CMGDS catalog](https://cmgds.marine.usgs.gov/data-releases/)
- **Marconi Beach, MA** (CACO-02), DOI 10.5066/P17OF3UP: two cameras "every half hour during daylight hours", March-June 2021, February-July 2023, January-September 2024; the release text names snapshots, time-averaged products and IO/EO calibration; CC0. The ScienceBase page text shows no video or timestack file names. (P) — [ScienceBase item](https://www.sciencebase.gov/catalog/item/67a2878dd34e05baae1c8705); [usgs.gov record](https://www.usgs.gov/data/usgs-coastcam-caco-02-marconi-beach-cape-cod-national-seashore-massachusetts-2021-2023-2024)
- The CoastCam programme page names Head of the Meadow and Nuvuk/Point Barrow as decommissioned, and says CoastCams "typically collect images or short videos at regular time intervals (usually every 30–60 minutes)". It links no video archive. (P) — [USGS CoastCams](https://www.usgs.gov/programs/cmhrp/science/usgs-coastcams)
- Nearest buoys for the Florida sites are 26-40 km away (CDIP 214). (R) — `data/registry/datasets.yaml`, entry `usgs_coastcam`
- DUNEX camera footprint about 35.6776 N, 75.4795 W; CDIP 192 (Oregon Inlet) at 35.749 N, 75.331 W with raw displacement in d06 (2020-06-09 to 2022-08-04): about 15.5 km. (P files, C distance) — [CDIP 192 archive](https://thredds.cdip.ucsd.edu/thredds/catalog/cdip/archive/192p1/catalog.html)

### Inferences
- A 2 Hz timestack is wave-resolving along one line: it supports wave period, celerity, breaker position, runup and swash statistics, which hourly stills cannot. It is the closest public substitute for video at USGS sites.
- Tres Palmas + CDIP 181 is the best ready-made pair found in this pass: 331 hours of 2 Hz pixel series with raw buoy displacement 3.5 km offshore, September to March. Rincón faces winter North Atlantic swell, so it likely adds the swell-dominated cases the project lacks; this is my reading of the site, not a statement in the release.
- Madeira Beach supplies thousands of hours of low-energy Gulf conditions (the calm-sea gap) and, for 2021-03-10 to 2022-01-25, an ADCP and pressure logger 1.7 km away in under 6 m of water. If USGS adds the 2017-2021 ADCP record as stated, nearly all 6,329 timestacks get near-field truth. Sand Key still depends on a buoy 26-40 km away.
- Burst or continuous pressure at MB2 is a raw series from which individual waves can be recovered (with depth attenuation limits), so Madeira is also a candidate for wave-by-wave statistics next to 2 Hz pixels.
- Raw video exists at USGS for all these periods ("collected raw video"); a request to the St. Petersburg centre for Madeira or Sand Key clips is plausible but untested.

### Gaps
- Whether the Head of the Meadow, Santa Cruz, Sunset Beach, Tyndall and Alaska releases contain video or timestacks: only titles were read ("Imagery and Calibration Data"); the Head of the Meadow DOI did not resolve to a readable page.
- The MB2 pressure sampling rates are "specified in the netCDF global attributes", which were not read.
- No USGS release of the Pea Island in-situ wave data (waverider, pressure array) was found in the CMGDS catalog. A pass-2 search found only that data.gov lists 17 DUNEX-related records (surface models, orthomosaics, bathymetry, meteorology), none described as video, lidar or wave data. (S) — [data.gov DUNEX listing](https://catalog-beta.data.gov/?q=DUNEX)
- Nuvuk/Barter Island and the "Dream Inn" Santa Cruz camera were not checked individually.

---

## 3. Other long-running coastal camera programmes with archives and nearby instruments

### Takeaway
No regional programme was found that offers open video with an instrument nearby; pass 2 produced mostly negatives (PacIOOS, UK, France, Japan, Korea), each from one or two searches. Confirmed in pass 1: Narrabeen's public portal still offers profiles, not Argus images, but AODN now publishes **raw buoy displacement at Collaroy-Narrabeen**; CoastSnap photos can be downloaded per station with a free account; the 2017-2019 WebCAT cameras recorded 30 fps video in 10-minute files and their page now redirects to WebCOOS; the Deltares public Argus archive address returns 404.

### Cited Findings
- **Narrabeen-Collaroy**: the UNSW portal describes "monthly beach profile data from an ongoing beach survey program, commenced in April 1976" and offers no image or video download. (P) — [narrabeen.wrl.unsw.edu.au](http://narrabeen.wrl.unsw.edu.au/)
- Turner et al. (2016) is a profile and inshore-wave dataset; Argus imaging runs hourly in daylight since July 2004; a Dryad record holds 10,487 shorelines, some from Argus. None is raw imagery. (S) — [Turner et al. 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4828941); [Dryad shorelines](https://datadryad.org/dataset/doi:10.5061/dryad.70rxwdcd4)
- AODN serves delayed-mode **raw displacement** for a Sofar Spotter-V3 at Collaroy-Narrabeen (14 m depth, 33.727 S 151.306 E): 59 files, 2024-04-16 to 2026-10-05, CC BY 4.0 (details in §6). (P) — [AODN Collaroy-Narrabeen raw displacements](https://thredds.aodn.org.au/thredds/catalog/IMOS/COASTAL-WAVE-BUOYS/WAVE-BUOYS/DELAYED/WAVE-RAW-DISPLACEMENTS/COLLAROY-NARRABEEN/catalog.html)
- **CoastSnap**: 37 regional projects in 16 countries are listed. Images are downloaded one station at a time through SPOTTERON ("Choose the option 'Create Spot-Package' and confirm the download"); "The data of the project can be downloaded for free"; an account is needed; no licence is stated. These are phone stills, not video. (P) — [CoastSnap](https://www.coastsnap.com/); [download instructions](https://www.coastsnap.com/data/download-images)
- **WebCAT** (Dusek et al. 2019): seven Surfline-installed cameras at Buxton NC, North Myrtle Beach SC, Folly Beach SC (two), St. Augustine FL, Miami FL and Bradenton FL; minimum 1280 × 720 at 30 fps, stored "in 10-minute increments" on AWS; only Buxton and Miami were surveyed for rectification; no wave instrument or data licence is named. (P, article) — [Dusek et al. 2019](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2019.00353/full)
- `secoora.org/webcat` now redirects (301) to `webcoos.org`. (P) — [secoora.org/webcat](https://secoora.org/webcat/)
- **Deltares Argus** (Egmond, Noordwijk etc.): `argus-public.deltares.nl` and its `/archive/` path return HTTP 404. (P) — [argus-public.deltares.nl](https://argus-public.deltares.nl/)
- **Hawaii / PacIOOS**: a search returned only the 2009-2013 Waikiki and Waimea camera archives the project holds, and the CDIP-distributed buoys; no current PacIOOS beach camera surfaced. (S) — [PacIOOS Waimea beachcam](https://www.pacioos.hawaii.edu/shoreline/beachcam-waimea/)
- **UK**: Channel Coastal Observatory results were buoys (37 Datawell MkIII in a Southampton slide deck), a storm catalogue and aerial photography under the Open Government Licence; no camera or video archive. A Scarborough camera page exists at the North East Coastal Observatory. (S) — [CCO storm catalogue](https://www.coastalmonitoring.org/ccoresources/stormcatalogue/); [NECO Scarborough camera](https://www.northeastcoastalobservatory.org.uk/data/scarborough_camera)
- **France**: a three-camera station at Anglet has about five years of daily images (described in an image-stabilisation paper; release not stated), and a station on the Biarritz lighthouse runs the SIRENA software; no open Biscarrosse or Porsmilin video was found. (S) — [Anglet stabilisation paper](https://oskar-bordeaux.fr/handle/20.500.12278/196757?show=full); [SIAME equipment page](https://liuppa.univ-pau.fr/en/collaborations-and-projects/equipment.html)
- **Korea**: Yoon et al. (2021) trained a CNN on Samcheok Beach CCTV with AWAC wave labels, the same kind of pairing as Kim 2023; no data release is stated, and no AI Hub, KHOA or KIOST CCTV wave dataset surfaced. A Mendeley record holds observed wave heights at 15 east-coast stations (labels only). (S) — [Yoon et al. 2021](https://complete.bioone.org/journals/journal-of-coastal-research/volume-114/issue-sp1/JCR-SI114-057.1/Detection-of-Wave-Parameters-Using-CCTV-Images-Based-on-Deep/10.2112/JCR-SI114-057.1.full); [Mendeley 6cknvrnvh6](https://data.mendeley.com/datasets/6cknvrnvh6)
- **Japan**: NOWPHAS shows real-time wave statistics and JODC documents the archive formats; no coastal camera dataset and nothing on Hazaki pier video was found. (S) — [NOWPHAS](https://www.mlit.go.jp/kowan/nowphas/index_eng.html)
- **Surfline**: no research licence or terms text surfaced; Fox Weather holds "exclusive broadcast rights to over 500 HD Surfline cameras", which indicates commercial licensing. (S) — [TV Technology](https://www.tvtechnology.com/news/fox-weather-taps-surfline-for-weather-coverage)
- **NSW labels**: "Shoreline and Nearshore Wave Data for New South Wales" (v2, 2025-11-24): hourly H_s, T_p, T_m and directions at the 10 m contour for over 8,000 transects, 1979-2024, plus satellite shorelines; CC BY 4.0; 85.6 GB; model-derived, no imagery. (P) — [Zenodo 17693915](https://zenodo.org/records/17693915)
- **Spanish CoastSnap derivatives**: SCLabels (labelled rectified images from five stations) and SCShores (1,721 shorelines, ESSD 2023). (S) — [SCLabels](https://zenodo.org/records/10159978); [SCShores](https://essd.copernicus.org/articles/15/4613/2023/)
- Already scanned by the project and not re-checked: WebCOOS (86 cameras), Western Australia DoT cameras and Waveriders, GLERL webcams, Cam-Era, SOCIB (AWAC 1.5 km from cameras, images need agreement), NSW nearshore buoys, UK CCO buoys, KHOA. (R) — `docs/literature/data_source_scan_2026-10.md`

### Inferences
- The WebCAT sites are probably the "archived" WebCOOS cameras the project already lists (`staugustinecam`, `miami40thcam`), so WebCAT is an earlier period of a known source, reachable through the same WebCOOS request, not a new network.
- Narrabeen now has the truth half of a strong pair (raw displacement in 14 m, a swell-dominated coast) but still no open imagery; the practical image sources there are a CoastSnap station package (stills) or a request to UNSW for Argus products.
- Hillarys (Perth) also has raw displacement on AODN (from 2021-06), and the project's scan lists WA DoT cameras at Trigg Point and Swanbourne near Perth; if the DoT archive request succeeds, that pair gets raw-displacement truth. Distances were not computed.

### Gaps
- Searched in pass 2 with no open archive found (absence in one or two searches, not proof): UK CCO/NNRCMP cameras, Biscarrosse and Porsmilin video, Hazaki pier video, KHOA/KIOST CCTV, current PacIOOS cameras, and European Argus-type archives on 4TU or Zenodo (that search returned only USGS records).
- Still not researched: Spanish networks other than SOCIB and CoastSnap, Gold Coast and NSW council cameras, NOAA CO-OPS and NWS webcams, other IOOS regional associations.
- Surfline's terms of use returned HTTP 403 to an automated fetch, so its restrictions on recording or research use are unquoted. — [Surfline terms](https://www.surfline.com/terms-of-use)
- Which programmes provide video rather than stills: confirmed only for WebCAT/WebCOOS (video), USGS (2 Hz timestacks public, video internal), CoastSnap (stills).

---

## 4. Cameras on platforms that also carry motion or wave sensors

### Takeaway
No open dataset was found that joins a camera, platform motion and wave truth beyond the Schwendeman & Thomson set the project already holds. Open robotics datasets do give real **camera + IMU on one small vessel** (MODD2, MODS, Pohang Canal), but in sheltered water and with no wave measurement. The Ekofisk platform's stereo video is public only as derived whitecap and spectral series, and NDBC has no BuoyCAM archive or video.

### Cited Findings
- NDBC buoys cannot supply motion series to pair with BuoyCAM strips: "the raw acceleration or displacement measurements are not transmitted shore-side". (P) — [NDBC wave FAQ](https://www.ndbc.noaa.gov/faq/wave.shtml)
- "wave displacement time series are not returned from NDBC's wave measurement systems". (P) — [NDBC wave calculation FAQ](https://www.ndbc.noaa.gov/faq/wavecalc.shtml)
- The NDBC BuoyCAM page served only navigation text to a plain fetch (the content loads by script); nothing on video or higher-rate imagery could be read. (P) — [NDBC BuoyCAMs](https://www.ndbc.noaa.gov/buoycams.shtml)
- **MODD2** (Univ. of Ljubljana): 28 sequences, 11,675 stereo frames at 1278 × 958 from an unmanned surface vehicle in the Gulf of Koper, Slovenia, over about 15 months; cameras 0.7 m above the water; all sequences "time-synchronized with the measurements of the on-board sensors" (IMU, GPS, compass); two sequences have abrupt rolling and pitching. Video, IMU and GPS downloads are listed; no licence is stated for MODD2. (P) — [MODD2 page](https://data.vicos.si/borja/viamaro/index.html)
- **Pohang Canal Dataset** (KAIST): stereo, infrared and omnidirectional cameras, three lidars, marine radar, GPS and an AHRS on a vessel through a canal, inner and outer harbour and nearby coastal water, about 7.5 km; CC BY-NC 4.0; free download; a 2023 note reports an AHRS timestamp problem under investigation. (P) — [arXiv 2303.05555](https://arxiv.org/abs/2303.05555); [dataset site](https://sites.google.com/view/pohang-canal-dataset)
- **DUNEX microSWIFT drifters** (no camera): GPS at 4 Hz and IMU at 12 Hz, 3-30 October 2021 at the FRF; 67 missions, 971 drift tracks, 116,307 wave realisations, 3,419 detected breaking events; Dryad DOI 10.5061/dryad.hx3ffbgk0; also on the FRF server as 67 mission files with "Instantaneous sea surface elevation". Licence on the FRF copies: scholarly use with acknowledgement, no commercial use. (P) — [Rainville et al. 2023, ESSD](https://essd.copernicus.org/articles/15/5135/2023/); [FRF Dunex/UW_drifters](https://chldata.erdc.dren.mil/thredds/catalog/frf/projects/Dunex/UW_drifters/catalog.html)
- **MODS** (Bovcon et al. 2021, IEEE T-ITS): about 81,000 stereo images "synchronized with an on-board IMU", over 60,000 annotated objects, from a small unmanned surface vehicle; said to be public at vicos.si/resources. (S) — [arXiv 2105.02359](https://arxiv.org/abs/2105.02359v2)
- **Ekofisk platform, North Sea** (Malila, Thomson, Breivik 2022): UW ResearchWorks hdl 1773/48143, CC BY-NC 3.0 US, one 17.67 MB zip. Per stereo-video acquisition: CSVs of mean sea level, a wave-group mask and whitecap-coverage time series with their spectra; one table of spectral parameters "from stereo video, laser array, and WAMOS radar"; sea state and wind from the platform. The record lists no raw frames and no elevation grids. (P) — [UW ResearchWorks record](https://digital.lib.washington.edu/researchworks/items/adadf2e3-e751-4ede-8014-191333f1a6d1)
- The Ekofisk extreme-crest study used a laser altimeter record from 2003 to 2020 and stereo video from five storms in winter 2019/20; no dataset DOI was found for it. (S) — [UiB record](https://bora.uib.no/bora-xmlui/handle/11250/3144178)
- **Ship bridge camera, Beaufort labels** (Vorkapic et al. 2024, Ocean Eng. 309): "a single stationary camera mounted on the ship bridge" on an ocean-going vessel; the abstract page states no data or code release. (P) — [arXiv 2407.03755](https://arxiv.org/abs/2407.03755)
- **Ship X-band radar with Spotter truth** (not a camera): about 95,000 SeaVision radar images and 62 hours of simultaneous Spotter measurements from four Atlantic and Arctic cruises, on PANGAEA (10.1594/PANGAEA.939620). (P, article text) — [Rezvov et al. 2024](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2024.1363135/full)
- **KOLOMVERSE**: the abstract lists "weather, time, illumination, occlusion, viewpoint, background, wind speed, and visibility" as conditions considered, with no per-image tags named (P) — [arXiv 2206.09885](https://arxiv.org/abs/2206.09885). A search summary says the collection protocol grouped footage by wave height (0, 1-2 m, 3-4 m). (S) — same paper, [PDF](https://arxiv.org/pdf/2206.09885)
- **NDBC BuoyCAM**: a search found no official image archive and no video capability; the FAQ gives only the latest-image link and says cameras typically take photos in daylight. (S) — [NDBC BuoyCAM FAQ](https://ndbc.noaa.gov/faq/buoycamlinks.shtml)
- Yang et al. (2026) worked wholly in a Unity/Crest simulation and name the lack of real synchronised shipborne camera-IMU-wave data as their limitation. (R) — `docs/literature/papers/yang2026.md`
- Saildrone camera footage has no archive; SWIFT v4 drifters carry a camera but publish no images. (R) — `docs/literature/data_source_scan_2026-10.md` §5-6

### Inferences
- The camera-plus-IMU gap stays open for wave work. MODD2 and Pohang can only serve an auxiliary task (horizon and attitude from the image, checked against the IMU), which is the confound Yang et al. model.
- The microSWIFT set gives buoy-frame accelerations and breaking detections in the same surf zone and month as the FRF lidar gauges and tower cameras; with FRF imagery it would label breaking events from two independent instruments.

### Gaps
- Still not researched: research-vessel underway cameras (R2R, Schmidt Ocean, NOAA ships), wave gliders, Sofar platforms with cameras.
- One search for polar or research-vessel stereo on PANGAEA found no wave dataset; the Polarstern stereo system it returned measures sea-ice topography. (S) — [PSITRES paper](https://www.cambridge.org/core/product/38F00E865E5F2BC93520D6116AE1EB53/core-reader)
- Whether MAPFusion (shipborne video with AIS, radar, GNSS and gyrocompass) is public, or has wave labels, is unknown. (S) — [arXiv 2607.03915](https://arxiv.org/pdf/2607.03915)
- The licences of MODD2 and MODS and the Pohang frame rates were not on the pages read.

---

## 5. Open stereo, lidar, drone and laboratory datasets of sea-surface elevation with imagery

### Takeaway
The Guimarães et al. (2020) stereo collection, unreachable in October, is **reachable again** and its Black Sea 2013 folder includes co-located wave-gauge files, which makes it an imagery-plus-point-truth set with the instrument on the same platform. Pass 2 added three smaller items: Ekofisk stereo-derived series (no frames), a 123 GB large-flume video set labelled by breaker type, and a laboratory polarimetric image set with wave-gauge truth. ROXSI and the DUNEX remote-sensing data are not published anywhere I could find.

### Cited Findings
- The IFREMER Sextant record answers: "A Data Set of Sea Surface Stereo Images to Resolve Space-Time Wave Fields", CC BY 4.0, temporal coverage 2011-09-30 to 2018-01-02, DOI 10.12770/af599f42-2770-4d6d-8209-13f40e2c292f, contents "original records, calibration files, configuration files for processing" and example elevation maps. (P) — [Sextant record](https://sextant.ifremer.fr/geonetwork/srv/api/records/af599f42-2770-4d6d-8209-13f40e2c292f)
- The download root lists six campaign folders: `AA_2014`, `AA_2015` (Acqua Alta), `BS_2011`, `BS_2013` (Black Sea), `LJ_2018` (La Jument), `YS_2017` (Yellow Sea). (P) — [data-dataref.ifremer.fr/stereo](https://data-dataref.ifremer.fr/stereo/)
- `BS_2013` holds three acquisitions (2013-09-22 at 10 Hz, 2013-09-25 and 2013-09-30 at 12 Hz, each with `config/`, `input/`, `nc/`), a `BS_2013_Meteo` folder and `BS_2013_WaveGauge` with one 1.5 MB `.mat` wave-gauge file per acquisition. `AA_2014` holds one 12 Hz acquisition plus meteorology. (P) — [BS_2013](https://data-dataref.ifremer.fr/stereo/BS_2013/); [BS_2013_WaveGauge](https://data-dataref.ifremer.fr/stereo/BS_2013/BS_2013_WaveGauge/)
- The WASS data page still links only two software test sets and the Guimarães collection. (P) — [WASS data](https://sites.google.com/unive.it/wass/data)
- FRF single-wave Velodyne lidar scans (30 files, 2017) and the USGS DUNEX stereo pair (1 Hz, 5 min per hour; only timestacks released) are described in §1 and §2. (P) — [WaveShape](https://chldata.erdc.dren.mil/thredds/catalog/frf/projects/WaveShape/lidarWaveScans/catalog.html); [DUNEX release](https://coastal.er.usgs.gov/data-release/doi-P14EWDST/)
- **Large wave flume video** (Robertson et al. 2026, Coastal Eng. 204, 104902): Dryad DOI 10.5061/dryad.3ffbg79w1, published 2026-01-06, 122.96 GB. O.H. Hinsdale flume at Oregon State (about 90 m long, 1/6 scale); a GoPro on the sidewall at 1080 × 1920, 60 fps, videos of 10-15 min; mostly irregular JONSWAP runs named by H_s and T_p (e.g. H030T33 = 0.3 m, 3.3 s); labels are breaking features and breaker type. No wave-gauge files are mentioned; reuse under Dryad's terms. (P) — [Dryad record](https://datadryad.org/dataset/doi:10.5061/dryad.3ffbg79w1)
- **Laboratory polarimetric images with gauges** (Ginio et al., Technion): polarimetric camera images of mechanically generated waves "coupled with surface elevation records by wave gauges linear array"; regular waves of several steepnesses, irregular JONSWAP fields and breaking cases, several camera positions; "available for download on ScienceDB repository" (address not on the page). (P, abstract) — [arXiv 2410.22849](https://arxiv.org/abs/2410.22849)
- **KIOST 2-D flume** (Kim et al. 2021, Sci. Rep.): 57 video sets from side and top cameras with three capacitance gauges at 50 Hz; availability not read. (S) — [PMC8571332](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8571332/)
- **Torrey Pines drone lidar** (UCSD Library): hovering drone lidar and truck lidar "time-stacks", runup statistics and series, and surface-corrected pressure series from a buried sensor transect; no video. The library page refused an automated fetch. (S) — [UCSD collection bb77208588](https://library.ucsd.edu/dc/collection/bb77208588)
- **ROXSI** (rocky shores, Monterey Peninsula): experiments in June-July 2022 and September-November 2023 with ground and drone lidar, georectified drone images and in-water arrays; the "Observational Data" page reads only "This will point to the data sets". (P) — [ROXSI](https://roxsi.ucsd.edu/); [data page](https://roxsi.ucsd.edu/observational-data/)
- **Jangsa Beach, Korea**: drone wave-field videos with echosounder bathymetry (no wave instrument). (S) — [Mendeley ch5fb6f9zd](https://data.mendeley.com/datasets/ch5fb6f9zd)
- Ekofisk stereo-derived series: see §4. (P) — [UW ResearchWorks record](https://digital.lib.washington.edu/researchworks/items/adadf2e3-e751-4ede-8014-191333f1a6d1)
- Record ranges from the data paper: 12 records of about 30 min, H_s 0.30-10.03 m, with wave wires, AWAC and X-band radar at the sites. (R) — `data/registry/datasets.yaml`, entry `guimaraes2020_stereo`

### Inferences
- Guimarães 2020 can now be sized and fetched for the stereo validation work (project item P6); the Black Sea records are the ones with an independent gauge series on the platform.
- Total size is still unknown: the directory listings show no sizes for folders.

### Gaps
- Still open: Vieira et al. low-cost stereo, "BELS", SandyDuck-era imagery beyond photos.
- DUNEX: no video or lidar release surfaced in two searches (no DesignSafe record; data.gov lists records of other kinds). ROXSI: no repository yet.
- The polarimetric set's ScienceDB address, the KIOST flume data statement and the UCSD record could not be read (pages blocked); whether the OSU flume set can be joined to the facility's wave-gauge records is not stated.
- Licence of the two WASS test zips is not stated on the page.

---

## 6. Raw displacement and individual-wave truth

### Takeaway
**CDIP serves raw x, y, z displacement for every deployment of the project's buoys**, in the per-deployment archive files (not the `historic.nc` aggregate the project downloaded) and in a realtime `_xy.nc` file, and it also publishes ready-made **zero-crossing files** with one row per wave. This closes the "no raw elevation series" gap at buoy distance for all existing NC, FL and Hawaii camera archives. NDBC transmits no raw series; AODN publishes raw Spotter displacement for 25 Australian sites.

### Cited Findings
**CDIP archive deployment files** (all checked files contain `xyzZDisplacement`, `xyzXDisplacement`, `xyzYDisplacement`, QC flags, `xyzStartTime`, `xyzSampleRate`, `xyzFilterDelay`)
| Station (name in file title) | Deployment files with xyz | Span of deployments | Notes |
|---|---|---|---|
| 433 Duck FRF 17 m, NC | d01-d13 (13 of 13) | 2013-05-09 to 2025-07-25 | d13: 72,193,535 samples at 1.28 Hz; filter delay 133.3 s |
| 200 Wilmington Harbor, NC | d01-d14 (14 of 14) | 2013-02-06 to 2026-07-20 | DWR-M4 in 2019-11 to 2025-01 |
| 150 Masonboro Inlet, NC | d01-d19 (19 of 19) | 2008-05-07 to 2026-03-21 | |
| 106 Waimea Bay, HI | d01-d28 (28 of 28) | 2001-12-16 to 2026-05-21 | 2009-2013 is d09-d13 |
| 134 Fort Pierce, FL | d01-d26 (26 of 26) | 2006-09-29 to 2026-04-26 | |
| 243 Nags Head, NC | d01-d05 (5 of 5) | 2018-08-26 to 2026-05-20 | |
| 143 Cape Canaveral Nearshore, FL | d01-d36 (36 of 36) | 2006-08-18 to 2026-11-05 | 2023-05 to 2024-09 is d31-d33 |
(P) — [CDIP archive, e.g. 433p1](https://thredds.cdip.ucsd.edu/thredds/catalog/cdip/archive/433p1/catalog.html); [433p1_d13 header](https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/433p1/433p1_d13.nc.html). Spans are first start to last end; there are gaps of days to months between deployments.
- Sample times are "calculated as xyzStartTime + arrayIndex/xyzSampleRate - xyzFilterDelay". (P) — [433p1_d13 header](https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/433p1/433p1_d13.nc.html)
- `433p1_historic.nc` has no xyz variables (0 matches in its DDS). (P) — [433p1_historic header](https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/433p1/433p1_historic.nc.html)
- Licence attribute: "These data may be redistributed and used without restriction." (P) — same d13 header
- 433 d13 carries the note "Wave sensor failure 2023-12-18 to 2024-01-02". (P) — same header

**CDIP realtime displacement files** (`{stn}p1_xy.nc`, current deployment)
- 433: 2025-08-08 to 2026-10-19, resolution 0.39 s (2.56 Hz, DWR-M4), 96.6 M samples. 150: 2026-04-08 to 2026-10-19, 0.39 s. 243: 2026-05-30 to 2026-10-21, 0.39 s. 106: 2026-05-22 to 2026-11-19, 0.78 s. 134: 2026-09-21 to 2026-11-19, 0.78 s. 143 and 200: 2026-11-05 to 2026-11-19, 0.78 s. (P) — [CDIP realtime catalog](https://thredds.cdip.ucsd.edu/thredds/catalog/cdip/realtime/catalog.html); [433p1_xy header](https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/433p1_xy.nc.html)

**CDIP zero-crossing product** (individual waves, already computed)
- `cdip/zcross/archive/` has 196 station folders, including 106, 134, 143, 150, 200, 243 and 433; 433 has one `_zc.nc` per deployment d01-d13. `cdip/zcross/realtime/` has the current deployment (e.g. `433p1_d14_zc.nc`, 762,025 waves). (P) — [zcross archive 433p1](https://thredds.cdip.ucsd.edu/thredds/catalog/cdip/zcross/archive/433p1/catalog.html); [zcross realtime](https://thredds.cdip.ucsd.edu/thredds/catalog/cdip/zcross/realtime/catalog.html)
- Per wave: `zcrossTime` with bounds, `zcrossHeight`, `zcrossPeriod`, `zcrossDirection`, `zcrossZPeak` (trough/crest displacement), `zcrossMaxAccel`, and running statistics over ±900 s or 500 waves (`zcrossHThird`, `zcrossHTenth`, `zcrossTThird`, mean, standard deviation), with QC flags and a one-second index. (P) — [433p1_d14_zc header](https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/zcross/realtime/433p1_d14_zc.nc.html)
- CDIP's Python library reads the series with `StnData.get_xyz(start, end)`. (P) — [CDIPpy xyz example](https://ucsd-sio-cdip.github.io/CDIPpy/examples/cdippy-xyz)

**Other providers**
- NDBC: raw motion "not transmitted shore-side" (see §4). (P) — [NDBC wave FAQ](https://www.ndbc.noaa.gov/faq/wave.shtml)
- AODN / IMOS Coastal Wave Buoys: `DELAYED/WAVE-RAW-DISPLACEMENTS` has 25 site folders (Apollo Bay, Augusta Offshore, Bengello, Brighton, Cape Bridgewater, Ceduna, Collaroy-Narrabeen, Coral Bay, Goodrich Bank, Hillarys, Karumba, Mission Beach, North Kangaroo Island, Ocean Beach, Robe, Shark Bay, Storm Bay, Tantabiddi, Tathra, Torbay West, Wilsons Prom, Wooli and others). Files are about 61 MB per fortnight with `XDIS`, `YDIS`, `ZDIS`, `TIME`; Collaroy-Narrabeen spans 2024-04-16 to 2026-10-05 (no files between 2024-08-06 and 2024-12-05), Hillarys starts 2021-06-17, Wooli 2025-01-23. Licence CC BY 4.0; acknowledgement text required. Spotter rate is 2.5 Hz (1,251,001 samples in a file named for 2026-09-30 to 10-05). (P, rate C) — [AODN raw displacements](https://thredds.aodn.org.au/thredds/catalog/IMOS/COASTAL-WAVE-BUOYS/WAVE-BUOYS/DELAYED/WAVE-RAW-DISPLACEMENTS/catalog.html)
- Sofar Spotter API: the documentation lists payload types `waves`, `full` (spectrum), `track` and `sensorData`, with two 30-minute samples per hourly update; raw displacement is not mentioned in the first 100,000 characters of the documentation export. (P) — [Sofar docs export](https://docs.sofarocean.com/llms-full.txt)
- FRF: 7 Hz lidar elevation 2015-2026, 2 Hz gauge series for 2015-10, 5 Hz pier radar 2022-2024 (§1). DUNEX microSWIFT sea-surface elevation, October 2021 (§4).
- FOWD (Free Ocean Wave Dataset; Häfner, Gemmrich, Jochum, J. Atmos. Oceanic Technol. 38(7), 2021): the processed CDIP catalogue contains "over 4 billion waves"; the code is MIT-licensed and its "best supported source is CDIP buoy data". (P) — [arXiv 2011.12071](https://arxiv.org/abs/2011.12071); [FOWD repository](https://github.com/dionhaefner/FOWD)
- FOWD's processed output is about 1.1 TB from about 270 GB of raw CDIP files. (S) — [FOWD paper PDF](https://arxiv.org/pdf/2011.12071)
- Sofar Spotters save "3D relative displacement time series ... at 2.5Hz to the SD card, regardless of data mode" (FLT files); telemetry carries aggregated data. (S: Sofar's guide is a Notion page that returned no text to a fetch) — [Spotter Data Access guide](https://sofarocean.notion.site/Spotter-Data-Access-c7f51351f82e4b21bc5e9251a84423c3)
- Recovered Spotter displacement does get published by operators: the USGS Madeira Beach release has `Sdisp.nc` displacement files for its MB1 buoy (§2), and AODN's delayed-mode files above are of this kind. (P) — [USGS release P9HZKIWB](https://cmgds.marine.usgs.gov/data-releases/datarelease/10.5066-P9HZKIWB/)

**Overlap with imagery the project holds**
- WebCOOS Currituck (2022-01 on) ↔ 433 d08-d13 then realtime; no displacement file covers 2025-07-25 to 2025-08-08. Masonboro (2023-05 on) ↔ 150 d16-d19 then realtime from 2026-04-08. Oak Island ↔ 200 d10-d14 to 2026-07-20, realtime again from 2026-11-05. Jennette's ↔ 243 d04-d05 to 2026-05-20, realtime from 2026-05-30. Cocoa Beach (2023-05 to 2024-09) ↔ 143 d31-d33. PacIOOS Waimea (2009-02 to 2013-10) ↔ 106 d09-d13. (C, from the spans above and the registry's camera dates) — [CDIP archive](https://thredds.cdip.ucsd.edu/thredds/catalog/cdip/archive/433p1/catalog.html)

### Inferences
- With these files the project can compute, for every existing buoy-labelled clip, H_max, H_1/10, T_z, crest counts, wave-group statistics and both time-domain and spectral H_s (the dual labelling of Yang 2026), and train the five outputs of Kamagata 2026 against an instrument instead of unstated labels.
- The pairing is statistical, not wave-by-wave: buoys are 6-29 km from the cameras, so crest timing at the camera cannot be taken from the buoy. Only the FRF lidar/radar, the Black Sea gauge and the held Carini and Salin sets give phase-resolved truth in view.
- Files are large (three float arrays of tens of millions of samples per deployment); OPeNDAP index subsetting by the time formula avoids full downloads.
- Wherever a CDIP Waverider sits near another camera (e.g. 181 at Rincón, 192 at Oregon Inlet), the same displacement and zero-crossing files exist; only the nine stations named here were checked.

### Gaps
- FOWD's download address and data licence were not found (the journal page returned 403); with CDIP's own zero-crossing files available, FOWD is a convenience, not a need.
- Sofar's own data-access guide could not be read, so "displacement only on the SD card" rests on a search summary plus the API documentation's silence.
- Raw-series policies of UK CCO, Puertos del Estado, MHL (NSW Waveriders) and NOWPHAS were not researched.

---

## 7. Synthetic and simulated imagery with known wave truth

### Takeaway
No public synthetic optical image dataset with wave labels was found in either pass, in line with the project's earlier review, and the only quantified benefit of synthetic pretraining found is for X-band radar, not cameras. What exists openly are simulators: the Crest ocean system used by Yang 2026 (MIT-labelled repository for Unity's built-in renderer) and an FFT-wave Gazebo simulator with vessel hydrodynamics (GPL-3.0).

### Cited Findings
- Yang et al. (2026) rendered a bow camera and vessel attitude in "Unity/Crest", 47 sea states × 100 samples, with virtual buoys for labels. (R) — `docs/literature/papers/yang2026.md`
- Crest: "A class-leading water system implemented in Unity"; repository labelled MIT; "targets the built-in renderer", requires "Unity: 2022.3.62f3 or later"; HDRP/URP versions and "Crest Water 5" are sold on the Asset Store. (P) — [Crest repository](https://github.com/wave-harmonic/crest)
- `asv_wave_sim`: Gazebo plugins with wave algorithms `fft`, `trochoid` and `sinusoid` (FFT driven by wind speed, wind angle and steepness), a visual plugin for the sea surface and a hydrodynamics plugin for floating bodies; main branch targets Gazebo Garden; GPL-3.0. (P) — [asv_wave_sim repository](https://github.com/srmainwaring/asv_wave_sim)
- The project's review states "No public synthetic camera wave dataset with physical labels was found" and notes Airy-wave synthetic clips in Kamagata et al. arXiv:2606.13302. (R) — `docs/literature/landscape_review.md` §5.4
- **Sim-to-real evidence, radar only** (Rezvov et al. 2024): 60,000 synthetic X-band images made from white noise shaped by a Pierson-Moskowitz directional spectrum, with shadowing and tilt modulation, wind speed 3-20 m/s; a network pretrained on them and then trained on about 95,000 real ship-radar images reached H_s RMSE 0.204 m, against 0.48 m for the group's earlier model. The 0.48 m baseline is from a prior study, not a controlled ablation, and the method covers "fully developed sea" conditions only. (P) — [Rezvov et al. 2024](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2024.1363135/full)
- A second radar study trains on spectra from a JONSWAP-based synthetic simulator. (S) — [Ludeno et al. 2024](https://iris.cnr.it/retrieve/4b5eee5a-b5b2-48ea-bab3-0125bbe807b1/jmse-12-01609-v3.pdf)
- A ship study of Beaufort 1-8 from liner images mixed real images with synthetic ones; release not stated. (S) — [TRB record](https://trid.trb.org/View/2685736)
- Kamagata et al. arXiv:2607.11998 give no statement of code or data release on the abstract page; training used "6 annotated training scenes". (P) — [arXiv 2607.11998](https://arxiv.org/abs/2607.11998)

### Inferences
- A simulator, not a dataset, is the realistic route: Crest reproduces Yang's setting; a Gazebo vessel with camera and IMU sensors on FFT waves would give synchronised image, attitude and surface truth. Neither page confirms a prescribed directional spectrum (JONSWAP with spreading) as input, so the project's own spectral code would have to drive the surface.
- The CDIP displacement and FRF lidar series give real elevation records against which a simulated surface's statistics can be checked before any sim-to-real claim.

### Gaps
- Sim-to-real evidence for **optical** wave estimation: two searches found none; the radar result above is the nearest, and it is not a clean ablation.
- Blender/Unreal ocean pipelines and rendering papers with released data: one search found nothing; not exhaustive.
- Whether Crest's wave input is spectrum-based is not stated on the repository front page.

---

## 8. Labelled computer-vision datasets of the sea surface for auxiliary tasks or pretraining

### Takeaway
Candidates exist for every auxiliary task, but few could be confirmed as downloadable: RipVIS (184 rip-current videos; its site has lapsed), MU-SSiD (100,800 sea-state images; access unread), MassMIND (2,900 infrared maritime images), an Oregon State flume set labelled by breaker type (open on Dryad), and NOAA's scaled surf photographs from Oahu's north shore. The usable additions today are the Dryad flume set, the marine segmentation sets, and USGS timestacks and FRF lidar runup lines as supervised targets.

### Cited Findings
- **RipVIS** (Dumitriu et al., CVPR 2025): 184 videos, 212,328 frames; 150 videos with rip currents and 34 without; from drones, mobile phones and fixed beach cameras in the USA, Mexico, Costa Rica, Portugal, Italy, Greece, Romania, Sri Lanka, Australia and New Zealand; most annotated at 5 fps; no licence on the abstract page. (P) — [arXiv 2504.01128](https://arxiv.org/abs/2504.01128)
- The paper's distribution site `ripvis.ai` now shows a registrar parking page (domain expired). (P) — [ripvis.ai](https://ripvis.ai/)
- An earlier image-based rip-current set by the same group has a GitHub repository (Irikos/rip_currents). (S) — [arXiv 2504.01128 PDF](https://arxiv.org/pdf/2504.01128)
- **MU-SSiD** (Umair, Hashmani et al., Symmetry 14(7):1487, 2022): 100,800 visual-range images in four sea states; a 2021 precursor paper said the dataset "is not yet publicly available". The MDPI page refused a fetch, so the access route is unread. (S) — [MDPI article](https://www.mdpi.com/2073-8994/14/7/1487); [2021 precursor](https://aminer.org/pub/6229f1865aee126c0f379408/towards-development-of-visual-range-sea-state-image-dataset-for-deep-learning)
- A separate sea-horizon set by the same authors has 2,673 images at 1920 × 1080. (S) — [DOAJ record](https://doaj.org/article/3aeb9e13333642a08b55fd419195e5d2)
- **MassMIND**: over 2,900 long-wave infrared coastal images with instance segmentation in seven classes (sky, water, obstacle, living obstacle, bridge, self, background); the two repository records give different licences (CC BY-NC-SA and CC BY-NC). (S) — [arXiv 2209.04097](https://arxiv.org/abs/2209.04097v1)
- **Whitecap segmentation**: a 2025 paper segments whitecaps in shipborne images from a north-west Pacific fishery survey; data release not stated. No open annotated whitecap set surfaced. (S) — [Electronics 14(21):4286](https://www.mdpi.com/2079-9292/14/21/4286/htm)
- **Scaled surf photographs, Hawaii** (Caldwell, NOAA NODC): NCEI Accession 0001753, "Scaled photographs of surf over the full range of breaker sizes on the north shore of Oahu and Jaws, Maui", 1998-01 to 2004-05, heights scaled "using surfers as height benchmarks"; Accession 0000165 holds visual surf observations for 1968-1998. The NCEI page timed out. (S) — [NCEI 0001753](https://www.ncei.noaa.gov/archive/accession/0001753)
- **Breaker type in a flume**: the Oregon State set in §5 has labelled breaking features and plunging/spilling classes, 122.96 GB on Dryad. (P) — [Dryad record](https://datadryad.org/dataset/doi:10.5061/dryad.3ffbg79w1)
- MaSTr1325 is CC BY 4.0 according to the MODD2 page; MODD2 itself annotates water edge and obstacles on 11,675 stereo frames. (P) — [MODD2 page](https://data.vicos.si/borja/viamaro/index.html)
- MODD v1: twelve annotated USV videos for sky/shore/sea segmentation. (P) — [MODD page](https://www.vicos.si/resources/modd/)
- FRF `lidarWaveRunup` provides hourly runup-line series with a flag separating manually digitised from ML-derived lines, 2015-2026. (P) — [lidarWaveRunup header](https://chldata.erdc.dren.mil/thredds/dodsC/frf/oceanography/waves/lidarWaveRunup/2026/FRF-ocean_waves_lidarWaveRunup_202605.nc.html)
- Already registered by the project: Stringari et al. 2021 active-breaking crops, Yin et al. 2025 breaker-type clips, KOLOMVERSE, Buscombe optical wave gauging. (R) — `data/registry/datasets.yaml`

### Inferences
- For pretraining on water texture and horizon, the marine-robotics sets (MaSTr1325, MODD2, Pohang, KOLOMVERSE) are available now; none carries a sea-state label.

### Gaps
- Download routes and licences are unconfirmed for MU-SSiD, MassMIND, the whitecap set and NCEI 0001753 (pages blocked or timed out); how MU-SSiD's four sea-state labels were assigned (instrument or observer) is unread.
- The Oahu surf photographs overlap the CDIP 106 record (from 2001-12) for about two and a half years; their count, format and camera positions are unknown.
- Kaggle surf-camera sets and maritime datasets with weather tags were not searched.
- RipVIS's current download location (it may survive on a repository or paper page) is unknown.

---

## 9. Ranking: pairing distance, video, and coverage of swell or calm conditions

### Takeaway
Ranked by what they add for training and testing camera wave models, the top three are (1) CDIP raw displacement and zero-crossing files for the buoys already paired with the project's cameras, (2) the FRF lidar and radar elevation series, conditional on obtaining FRF imagery, and (3) USGS 2 Hz timestacks, led by Madeira Beach (USGS ADCP at 1.7 km, calm water) and Tres Palmas (CDIP buoy at 3.5 km, swell). Only sources with the instrument on the same structure reach the ≤1 km criterion, and none of those has open long-period video.

### Cited Findings
| Rank | Source | Truth ≤1 km of camera? | Video? | Swell or calm cases? | Access and licence | What it newly enables |
|---|---|---|---|---|---|---|
| 1 | CDIP xyz + zero-crossing files for 433, 150, 200, 243, 143, 106, 134 ([archive](https://thredds.cdip.ucsd.edu/thredds/catalog/cdip/archive/433p1/catalog.html), [zcross](https://thredds.cdip.ucsd.edu/thredds/catalog/cdip/zcross/archive/433p1/catalog.html)) | No: same 6-29 km pairs as today's labels (R) | Pairs with held WebCOOS 20-fps clips | 106 Waimea is swell-dominated; all seasons 2001-2026 | Open, no login; "redistributed and used without restriction" (P) | H_max, H_1/10, T_z, wave counts, group statistics, time-domain vs spectral H_s for every existing clip |
| 2 | FRF lidar gauges, runup, hydrodynamics 2015-2026; pier radar 2022-2024 ([FRF](https://chldata.erdc.dren.mil/thredds/catalog/frf/oceanography/waves/catalog.html), [radar](https://cmgds.marine.usgs.gov/data-releases/datarelease/10.5066-P1NLPWRP/)) | Yes for FRF tower cameras (same beach); imagery not public | No public video; timex/variance 2015-10 to 2016-08 only | All conditions over 11 years | Truth open (scholarly use, no commercial use); imagery by request | Wave-by-wave surf-zone truth in view; breaking-wave height; runup; the long-period in-view reference the project lacks |
| 3 | USGS timestacks: Tres Palmas 331, Madeira 6,329, Sand Key 1,404, DUNEX 411, Molokaʻi 30, Isla Verde 15 ([catalog](https://cmgds.marine.usgs.gov/data-releases/)) | No, but close: Madeira 1.7 km to USGS ADCP and pressure at MB2 for 2021-03 to 2022-01, earlier years promised ([release](https://cmgds.marine.usgs.gov/data-releases/datarelease/10.5066-P9HZKIWB/)) (C); Tres Palmas 3.5 km to CDIP 181 (C); DUNEX 15.5 km (C); Sand Key 26-40 km (R) | 2 Hz pixel line, 10-17 min per hour; no video | Madeira and Sand Key: calm Gulf; Tres Palmas: likely winter swell (inference) | Open; public domain or CC0 | Period, celerity and runup from wave-resolving pixels; calm-sea test set; a second swell site |
| 4 | Guimarães et al. stereo, with Black Sea wave gauge ([IFREMER](https://data-dataref.ifremer.fr/stereo/)) | Yes: gauge on the platform | 10-12 Hz stereo image pairs, about 30 min each | H_s 0.30-10.03 m across records (R) | Open, CC BY 4.0 | Space-time surface with an independent point gauge; stereo pipeline validation |
| 5 | AODN raw displacement, 25 sites incl. Collaroy-Narrabeen and Hillarys ([AODN](https://thredds.aodn.org.au/thredds/catalog/IMOS/COASTAL-WAVE-BUOYS/WAVE-BUOYS/DELAYED/WAVE-RAW-DISPLACEMENTS/catalog.html)) | Unknown: no open camera archive paired yet | None open (CoastSnap stills; WA DoT and Argus by request) | Swell-dominated coasts | Open, CC BY 4.0 | Individual-wave truth for an Australian camera if an archive is granted |
| 6 | DUNEX microSWIFT, October 2021 ([ESSD](https://essd.copernicus.org/articles/15/5135/2023/)) | In the FRF surf zone; no camera data attached | No | One autumn month | Dryad; FRF copy scholarly, non-commercial | Buoy-frame IMU and breaking detections beside the lidar gauges |
| 7 | MODD2, Pohang Canal ([MODD2](https://data.vicos.si/borja/viamaro/index.html), [Pohang](https://arxiv.org/abs/2303.05555)) | No wave truth | Yes, vessel video with IMU/AHRS | Sheltered water | MODD2 licence unstated; Pohang CC BY-NC 4.0 | Image-to-attitude auxiliary task on real data |
| 8 | Simulators: Crest, asv_wave_sim ([Crest](https://github.com/wave-harmonic/crest), [asv_wave_sim](https://github.com/srmainwaring/asv_wave_sim)) | Exact truth by construction | Rendered | Any prescribed sea state | MIT label; GPL-3.0 | Reproducing Yang 2026; camera + IMU + surface in simulation |
| 9 | Laboratory sets: Oregon State flume videos ([Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.3ffbg79w1)); Technion polarimetric images with wave gauges ([arXiv](https://arxiv.org/abs/2410.22849)) | Gauges in the basin (polarimetric set); none stated (flume set) | 60 fps video, 10-15 min runs (flume set) | Prescribed JONSWAP and regular waves | Dryad terms, 122.96 GB; ScienceDB address not found | Breaker-type auxiliary labels; controlled image-to-elevation tests |
| 10 | Labelled image sets: RipVIS, MU-SSiD, MassMIND, Oahu scaled surf photographs ([RipVIS](https://arxiv.org/abs/2504.01128), [NCEI](https://www.ncei.noaa.gov/archive/accession/0001753)) | No instrument (human or derived labels) | RipVIS is video; others stills | Oahu set is large winter swell (S) | Access unconfirmed for all four | Pretraining and auxiliary heads only |
| 11 | Ekofisk stereo-derived series ([UW](https://digital.lib.washington.edu/researchworks/items/adadf2e3-e751-4ede-8014-191333f1a6d1)) | Same platform (laser array, radar) | No frames: CSV series only | North Sea storms | Open, CC BY-NC 3.0 US, 17.67 MB | Whitecap-coverage series against spectra; no image training |

### Inferences
- Deciding principle for the order: sources that upgrade labels for imagery already held come first (rank 1 costs no new imagery and no permission); next come sources that add a new kind of truth but need imagery (rank 2) or add wave-resolving imagery with distant truth (rank 3).
- Rank 1 does not meet the ≤1 km test and cannot give crest timing at the camera. If phase-resolved truth is the priority, the FRF imagery request (rank 2) is the single action that would deliver it at scale on the US East Coast.
- For the calm-sea gap, Madeira Beach timestacks are the largest open set (6,329 hours), with an ADCP 1.7 km away for the last ten months; for swell, CDIP 106 displacement with the held Waimea images, then Tres Palmas.
- Order of action that follows from the table: (1) read CDIP xyz and zero-crossing files for the seven paired buoys; (2) fetch the Madeira Beach and Tres Palmas timestacks with the MB2 and CDIP 181 records; (3) ask the FRF for Argus pixel stacks or clips against the lidar gauges; (4) fetch the Guimarães Black Sea records; (5) treat everything from rank 5 down as optional.
- The buoy record covering "only one season" refers to the BuoyCAM set; that cannot be extended backwards (NDBC keeps no image archive and transmits no raw motion), so multi-season individual-wave truth has to come from the CDIP-paired shore cameras.

### Gaps
- Exposure of each pair (does the camera see the water the buoy measures) was not assessed; distances are great-circle only.
- No candidate was confirmed to have both open video and truth within 1 km over long periods.
- Sources unreachable or unreadable today: FRF data portal front page (connection reset), `argus-public.deltares.nl` (404), `ripvis.ai` (expired), Surfline terms, MDPI and the AMS journal page (403 to automated fetch), PMC and the UCSD Library (bot check), Springer/Nature (login redirect), Sofar's Notion guides (no text), NCEI accession page and one usgs.gov page (timeouts), NDBC BuoyCAM page (script-loaded content).
- Items tagged S should be opened by hand before any is cited or acquired: MODS, MU-SSiD, MassMIND, the KIOST flume set, Torrey Pines drone lidar, NCEI 0001753, FOWD's download, Sofar's SD-card guide, the Korean and French camera papers.
- Not covered at all: Kaggle surf-camera sets, NOAA CO-OPS and NWS webcams, Gold Coast and NSW council cameras, research-vessel underway cameras.
