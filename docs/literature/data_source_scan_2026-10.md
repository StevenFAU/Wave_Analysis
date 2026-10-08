# Data Source Scan: New Imagery and Wave-Reference Sources (2026-10-08)

**Search date:** 2026-10-08. **Scope:** sources that are not already in §5 of
the [landscape review](landscape_review.md): public imagery or video with a
wave instrument nearby, imagery that could get one, and new label-only
sources. Evidence levels are those of the landscape review (**Q** queried by
this project's code; **V2** the provider's record or dataset page read; **S**
search result only, not verified). V2 here means the record page was read
through a fetch tool's summary, not downloaded; check the record before
citing a number from it.

## 1. Summary

The largest find is that **WebCOOS has grown from the five Yin et al. (2025)
cameras to 86**. Of these, 41 are within 25 km of a buoy that publishes
directional spectra (35 of them live), and several have years of archived
one-minute stills. Distance is not exposure: the views still have to be
checked. They use the same API, token and downloader the
project already runs, and the same WebCOOS usage guidelines apply.

Three small, openly licensed research datasets fill sensor gaps that no
public camera can: ship-based stereo with ship motion and wave buoys (G-6,
G-9), thermal infrared with lidar wave measurements in the surf zone (G-5),
and stereo reconstructions of extreme waves in sea ice (G-6).

The other new camera networks (Western Australia, Great Lakes, New Zealand)
have archives, but bulk access has to be requested.

| Rank | Source | What it adds | Access | Level |
|---|---|---|---|---|
| 1 | **WebCOOS, cameras beyond the Yin et al. five** (§2) | Beachfront stills and video 6-25 km from CDIP buoys; Florida (Cocoa Beach) and Hawaii (Maui) among them | API token (held); recent 90 days now, older on request | **Q** |
| 2 | **Schwendeman & Thomson (2017) ship stereo** (§3.1) | Stereo surface elevation from a moving ship, with ship motion, SWIFT and Waverider spectra | CC BY 3.0 US; raw frames on request | V2 |
| 3 | **Carini et al. Duck 2016 IR + lidar** (§3.2) | Only public thermal-IR imagery of the surf zone with co-located lidar wave measurements found | CC BY; one 2.65 GB `.mat` | V2 |
| 4 | **Alberello et al. 2017 WASS, S.A. Agulhas II** (§3.3) | Stereo reconstructions of waves above 9 m in the Antarctic marginal ice zone | CC BY 4.0 (AADC record) | V2 |
| 5 | **Western Australia DoT coast cameras + DoT Waveriders** (§4.1) | 11 coastal cameras with an archive; several beside DoT directional buoys | Camera archive: ask; buoys: download | V2 / S |
| 6 | **GLERL Met-ReCON webcams** (§4.2) | Multi-year webcam archives at Great Lakes sites (lake wind sea) | Browsable; bulk on request | V2 |
| 7 | **Cam-Era, New Zealand** (§4.3) | Hourly 10-min averaged images since 1997 | Ask Earth Sciences New Zealand | S / V2 |

## 2. WebCOOS beyond the five Yin et al. cameras (Q)

`wave-analysis webcoos cameras` on 2026-10-08 listed **86 cameras**: 66 `up`,
15 `archived` and 5 `down`; 84 public and 2 protected. For each, the nearest
station publishing directional spectra in the 2026-09-26 NDBC inventory
(`realtime_products` containing `swdir`; CDIP buoys appear under their NDBC
numbers) was found by great-circle distance from the camera position. The
stills inventories below are from `wave-analysis webcoos inventory` on the
same day (metadata only, nothing downloaded). The table lists the
beachfront cameras among the 41 within 25 km, plus the camera nearest FAU.

| Camera | Place | Status | Nearest directional buoy, km | One-minute stills held by WebCOOS | Video archive |
|---|---|---|---|---|---|
| `san_elijo` | San Elijo State Beach, CA | up | **CDIP 262** (46274) Leucadia Nearshore, **5.8** | 66,105 (41 GB), 2026-07-15 → now | yes |
| `indian_river_inlet_1`, `_2` | Indian River Inlet, DE | up | CDIP 263 (44084) Bethany Beach, 7.9-8.2 | no stills product | no |
| `cocoabeach` | Cocoa Beach, FL | archived | **CDIP 143** (41113) Cape Canaveral Nearshore, **8.8** | **216,471 (119 GB), 2023-05-08 → 2024-09-11** | yes |
| `westport_north`, `_south` | Westport, WA | up | CDIP 036 (46211) Grays Harbor, 9.5 | north: 114,355 (33 GB), 2026-05-15 → now | yes |
| `masonboro_inlet` | Wrightsville Beach, NC | up | CDIP 150 (41110) Masonboro Inlet, 10.2 | **802,866 (183 GB), 2023-05-24 → now** | yes |
| `oceanshores_west`, `_east` | Ocean Shores, WA | up | CDIP 036 (46211), 12.7 | west: 178,990 (60 GB), 2026-03-06 → now | yes |
| `kahului_road`, `_harbor` | Kahului, Maui, HI | up | CDIP 187 (51205) Pauwela, 15.1 | road: 92,633 (52 GB), 2026-06-01 → now | yes |
| `currituck_sailfish` | Corolla, NC | up | 44056 Duck FRF, 17.8 | **1,040,526 (643 GB), 2022-01-04 → now** | yes |
| `jennette_north`, `_south` | Nags Head, NC (Yin et al.) | archived / up | CDIP 243 (44086), 18.6-18.8 | collected since 2026-09-26 | yes |
| `oceancity` | Ocean City, MD | up | CDIP 263 (44084), 19.7 | not queried | yes |
| `crescent_city` | Crescent City, CA | up | 46027, 19.8 | 68,382 (27 GB), 2026-07-07 → now | yes |
| `ferrybeach_north` | Saco, ME | up | 44007, 20.6 | 448,383 (90 GB), 2025-02-10 → now | yes |
| `miami40thcam` | Miami Beach, FL | archived | CDIP 265 (41122) Hollywood Beach, 21.1 | no stills product | yes |
| `currituck_hampton_inn` | Corolla, NC (Yin et al.) | up | 44056 Duck FRF, 22.2 | collected since 2026-09-26 | yes |
| `vabeach_hamptonos` | Virginia Beach, VA | up | CDIP 147 (44099) Cape Henry, 23.8 | not queried | yes |
| `staugustinecam` | St. Augustine, FL | archived | CDIP 194 (41117), 24.1 | no stills product | yes |
| `jupiterinlet` | Jupiter Inlet, FL | up | CDIP 134 (41114) Fort Pierce, 70.9 | not queried | yes |

Notes:
- **Distance is not exposure.** Inlet and harbour views (`indian_river_inlet`,
  `masonboro_inlet`, `kahului_harbor`) may look at sheltered water; each site
  needs the view checked and, where a second buoy exists, the buoy-pair
  exposure check of `scripts/buoy_pair_agreement.py`.
- **Florida.** `cocoabeach` is the only Florida camera with a nearshore buoy
  within 10 km, and its stills (2023-05 to 2024-09) are all older than 90 days,
  so they need WebCOOS's agreement. `jupiterinlet`, the camera nearest FAU, has
  no directional buoy within 70 km.
- **Within the 90-day guideline today:** all of `san_elijo` (from 2026-07-15),
  `crescent_city` (from 2026-07-07), and the most recent part of the others.
  Collecting them now loses nothing that later permission would recover.
- Volume at the project's cadence (one still per 30 min of daylight): about
  4-5 GB per camera-year, so roughly 50-60 GB a year for 12 more cameras.
- **Relevant new paper.** Kamagata et al. (arXiv:2606.13302, v2 2026-07-19)
  estimate peak period from video, with labels from expert visual assessment
  and timestacks, not an instrument, and the data are not public (V2). This
  confirms that instrument-labelled public video is still missing.

## 3. Open research datasets that fill sensor gaps

### 3.1 Schwendeman & Thomson (2017), ship-based stereo (V2)

"Data to accompany the article 'Sharp-crested Breaking Surface Waves Observed
from a Ship-Based Stereo Video System'", UW ResearchWorks,
<http://hdl.handle.net/1773/38314>, CC BY 3.0 US. R/V *Thomas G. Thompson*,
North Pacific, December 2014 to January 2015.

- Stereo products: gridded surface elevation and radiance in one-minute bursts,
  a 40 m × 40 m grid at 0.25 m, 5 or 7.5 Hz. Five minutes from 52 videos, and
  full 20-60 min captures from the 10 best.
- Ship motion for each burst (Novatel SPAN-IGM), used for the stereo
  projection.
- SWIFT drifters (2,522 bursts of 512 s: spectra, wave height and period,
  wind, turbulence) and Datawell Waverider spectra (263 records of 30 min).
- Raw frames are not included and are "available upon request from the
  authors".

Why it matters: the only open dataset found with **imagery, platform
attitude and an independent wave reference on a moving platform** (G-9), and
a second stereo validation set beside Guimarães et al. (2020) for P6.

### 3.2 Carini et al., Duck 2016 thermal IR and lidar (V2)

"Dataset to support 'Surf Zone Waves at the Onset of Breaking 1. LIDAR and IR
Data Fusion Method'", UW ResearchWorks, <http://hdl.handle.net/1773/46586>,
CC BY. USACE FRF, Duck, NC, November 2016: processed lidar line scans and
co-located thermal infrared imagery of surf-zone waves, wave and wind
conditions, and bathymetry surveys, in one 2.65 GB MATLAB file (use the file
marked current). The thesis analyses over 4,200 waves.

Why it matters: the only public thermal-IR record of ocean waves with a
co-located wave measurement found (G-5; [thermal IR](../datasets/thermal_ir.md)).
Small and close-range, so it suits method checks, not training.

### 3.3 Alberello et al., WASS on the S.A. Agulhas II (V2)

"Wave acquisition stereo-camera system measurements (WASS) from a voyage of
the S.A. Agulhas II, July 2017", Australian Antarctic Data Centre,
DOI [10.26179/Q9BD-5F74](https://doi.org/10.26179/Q9BD-5F74). Two 2448 × 2048
monochrome cameras at 2 Hz on the icebreaker's port side, 4 July 2017, in the
Antarctic marginal ice zone during an explosive polar cyclone; reconstructed
surface elevation as NetCDF on a 1 m grid. The AADC record gives CC BY 4.0;
the data.gov.au mirror says the licence is not specified. The record names
no ship-motion or wave-instrument data.

Why it matters: stereo reconstructions of very large waves (maximum heights
above 9 m reported) to test the spectral pipeline at the high end (P6).

### 3.4 Smaller finds

| Source | Content | Access | Use | Level |
|---|---|---|---|---|
| Acqua Alta tower, 45-year directional wave record (Scientific Data 2025) | Wave gauges 1979-2023 at the tower where WASS was developed | PANGAEA [10.1594/PANGAEA.974074](https://doi.org/10.1594/PANGAEA.974074), CC BY 4.0, ~117 MB | Reference if CNR-ISMAR ever shares WASS stereo from the tower (not public) | V2 |
| REDIAM (Andalusia) "Beach Camera Image Series for Environmental Monitoring" | Time-stamped, georeferenced images from a fixed beach camera, ongoing | LifeWatch image repository, CC BY 4.0 | Camera only; no instrument named. Check site and nearby Puertos del Estado buoys | V2 |
| SOCIB SIRENA (already in §5.1) | 7.5 fps video in the first 10 min of each daylight hour since 2011; ADCPs (MOBIMS) on THREDDS; derived datasets BWILD and SCLabels | BEAMON viewer; licence still unconfirmed | Ask SOCIB for the video archive and licence | V2 |

## 4. Camera networks with archives that need a request

### 4.1 Western Australia Department of Transport (V2 / S)

- **Cameras** (data.wa.gov.au "Coast Camera (DOT-032)", CC BY 4.0 for the map
  service): Albany, Fremantle Fishing Boat Harbour, Geraldton, Jurien Bay,
  Jurien Bay Harbour, Kalbarri, Lancelin, Mandurah Ocean Marina, Peaceful Bay,
  Swanbourne Beach, Trigg Point. The DoT says they "include an archive of beach
  and ocean conditions"; the archive is reached through the DoT coast-cams web
  pages, which may need a login. Cadence and period are not stated.
- **Wave buoys:** DoT directional Waveriders at Jurien (WMO 56004), Rottnest
  (56005), Cape Naturaliste (56006), Cottesloe (56008), Esperance (56010),
  Albany (56011) and Mandurah (56014). Quality-controlled data up to the end of
  the previous year download from the DoT map; times are AWST (UTC+8). Licence
  statements conflict (CC BY in a 2017 record; DoT conditions of use on the
  site).
- **Possible pairs:** Jurien Bay camera and buoy; Albany; Mandurah;
  Swanbourne and Trigg Point with Cottesloe. Distances not yet computed.
- **Ask DoT:** archive cadence, period, bulk access, and licence for research.

### 4.2 GLERL Met-ReCON webcams, Great Lakes (V2)

Browsable webcam archives at GLERL's real-time meteorological stations, for
example Thunder Bay Island (Lake Huron) 2014-2025 with up to six views, White
Shoal Light 2020-2024, and Muskegon (online since 1998, images every 5 min).
Waves are lake wind sea, short-period and fetch-limited; wave references are
seasonal NDBC/GLOS buoys. A different domain, useful for a cross-domain test
(Q-G2). Bulk requests: oar.glerl.data@noaa.gov.

### 4.3 Cam-Era, New Zealand (S / V2)

Run by NIWA (Earth Sciences New Zealand since 2025-07-01) with regional
councils since 1997: each daylight hour, 600 frames over 10 min, archived as
the first frame and the 10-min average (Raglan, Tairua, Pauanui, Mokau and
others). Averages show breaking patterns rather than individual waves. No bulk
access or licence found; ask the project team.

## 5. Label-only sources found

| Source | Coverage | Access | Use | Level |
|---|---|---|---|---|
| NSW nearshore wave buoys (Scientific Data descriptor) | 20 shallow sites, >7,000 buoy-days, half-hourly | data.nsw.gov.au, CC BY | Labels if an NSW camera archive becomes available | V2 |
| UK Channel Coastal Observatory wave buoys | ~10-12 m depth, south and west England; per-site yearly NetCDF repackaging (Bangor) | coastalmonitoring.org API (key), Open Government Licence | Labels for any UK camera | V2 |
| KHOA buoy API (data.go.kr) | Korean coastal buoys incl. wave height | Free, registration | Labels for KOLOMVERSE-style imagery | V2 |
| Jiang & Su (2026), ESSD 18:6841, gridded H_s from altimetry + WW3 ST6 | Global 0.5°, 1992-2023, open ocean (no altimetry within 75 km of land) | [10.57760/sciencedb.29314](https://doi.org/10.57760/sciencedb.29314) | A second nowcast baseline for offshore NDBC buoy cameras | V2 |
| Saildrone hurricane missions (PMEL ERDDAP) | One-minute USV records incl. waves, 2021-2026 | CC0 for the sensor data | Labels only: camera footage was released as press media, with no archive found | V2 |

## 6. Checked and not available

So that nobody repeats these searches:

- **UNSW Narrabeen Argus:** no public image archive found; NSW buoy data are open.
- **Zandmotor (Sand Engine) Argus:** no public imagery; 4TU holds Nortek X-band
  radar (2014-2016), ADCP and drifter data from the site.
- **FINO1 webcam:** a 2020s forum report says the images were not retained.
- **KHOA beach CCTV:** feeds go to agencies; the public gets derived indices only.
- **AI Hub (Korea):** no wave-image dataset with wave-height labels found.
- **Fusion-Wave** (Beihang, ICAIRC 2025): infrared wave video with annotated
  height and period, no download or licence found.
- **Saildrone cameras:** no image archive; contact aoml.webmaster@noaa.gov.
- **DYNALIT** (France): no camera data in the open portal; the Biscarrosse cameras are run by the University of Bordeaux.
- **USACE FRF Argus:** still only the 2018 North Head merged products on the
  CHL THREDDS server.
- **Coastalwatch** (Australia): commercial surf cameras listed in Research Data
  Australia; no instrument and no open licence.
- **SWIFT drifter images:** v4 drifters carry a camera, but no image archive is published.

## 7. Recommended next steps

1. **Extend the WebCOOS collector** to the near-buoy cameras, within the 90-day
   guideline: `san_elijo`, `masonboro_inlet`, `currituck_sailfish`,
   `westport_north/south`, `oceanshores_east/west`, `kahului_road`,
   `crescent_city`, `ferrybeach_north`, `oceancity`, `vabeach_hamptonos`. Check
   each view for exposure first.
2. **Add the historical stills of `cocoabeach`, `masonboro_inlet`,
   `currituck_sailfish` (from 2022) and `ferrybeach_north` (from 2025-02)** to
   the WebCOOS permission request (P2).
3. **Download the three open validation datasets** (§3.1-3.3) for P6 and the
   G-5/G-9 method checks. Together they are a few GB.
4. **Requests:** Western Australia DoT (camera archive), Schwendeman and
   Thomson (raw stereo frames), SOCIB (SIRENA video and licence), GLERL (bulk
   webcam images), Cam-Era.
5. Add a registry entry (`data/registry/datasets.yaml`) for each source when it
   is acquired.

## Sources

- WebCOOS API (`app.webcoos.org`), queried 2026-10-08 with this project's client.
- UW ResearchWorks: <https://digital.lib.washington.edu/researchworks/items/970ce35d-41c7-4e7d-b5eb-6bc6a5cf053a>;
  <https://digital.lib.washington.edu/researchworks/items/f701cfb2-bba6-447f-936e-df7dba9e9811>
- AADC record via Research Data Australia: <https://researchdata.edu.au/wave-acquisition-stereo-july-2017/1695828>
- PANGAEA Acqua Alta: <https://doi.org/10.1594/PANGAEA.974074>; Scientific Data article <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11802720/>
- REDIAM / LifeWatch: <https://metadatacatalogue.lifewatch.eu/srv/api/records/74b5598f-4e11-4596-b345-aa9784ce9261>
- SOCIB: <https://www.jerico-ri.eu/2021/07/13/virtual-access-to-coastal-ocean-data-enabled-through-the-socib-thredds-data-server/>; <https://portalinvestigacio.uib.es/documentos/67321bfaaea56d4af0482afb?lang=en>
- WA DoT: <https://catalogue.data.wa.gov.au/dataset/coast-camera-dot-032>; <https://wa.gov.au/service/environment/environment-information-services/live-coastal-cameras>; <https://transport.wa.gov.au/imarine/about-wave-data.asp>; <https://oceancurrent.aodn.org.au/waves/info_original.html>
- GLERL: <https://www.glerl.noaa.gov/res/recon/browsewebcam-tbi.php>; <https://glerl.noaa.gov/metdata/mkg/webcaminfo.txt>
- Cam-Era: <https://niwa.co.nz/coasts/cam-era>; <https://waikatocoastaldatabase.org.nz/CoastalDataSet/Details/93>
- NSW buoys: <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10861473/>
- CCO: <https://coastalmonitoring.org/ccoresources/api/>; <https://research.bangor.ac.uk/en/datasets/nc-cco-channel-coastal-observatory-cco-wave-buoy-dataset/>
- KHOA APIs: <https://www.data.go.kr/data/15099707/openapi.do>
- Jiang & Su 2026: <https://essd.copernicus.org/articles/18/6841/2026/>
- Saildrone / PMEL: <https://data.pmel.noaa.gov/pmel/erddap/info/sd1064_hurricane_2023/index.html>; <https://www.pmel.noaa.gov/news-and-media/highlights/when-saildrone-met-hurricane-sam>
- Kamagata et al. 2026b: <https://arxiv.org/abs/2606.13302>
- Zandmotor: <https://data.4tu.nl/articles/_/12689147/1>; FINO1: <https://community.windy.com/topic/40313/does-windy-save-webcam-data>
- KHOA CCTV: <https://www.heraldk.com/article/2026062218000078836>
- Fusion-Wave: <https://research.buaa.edu.cn/en/publications/construction-of-a-wave-image-dataset-for-marine-environment-perce/>
- DYNALIT: <https://hal.archives-ouvertes.fr/hal-03288097>; FRF THREDDS: <https://chlthredds.erdc.dren.mil/thredds/dap4/frf/projects/argusNorthhead/nrthhead_cx_merge_20180605_013000.nc.dsr.html>
- Coastalwatch: <https://researchdata.edu.au/coastalwatch-coastal-condition-monitoring-system/3979232>
- SWIFT v4: <https://www.apl.washington.edu/project/projects/swift/pdfs/SWIFTv4description.pdf>
