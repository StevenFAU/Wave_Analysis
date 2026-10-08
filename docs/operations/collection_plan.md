# Data Collection Plan: Gaps, Priorities and Next Steps

Status: **2026-10-01**, P0 and P1 updated 2026-10-08. This page says which research gaps the data collection
is meant to close, what each data stream contributes, the order of work, and
the specification of a self-collected site. Gap IDs refer to
[research_gaps.md](../literature/research_gaps.md).

## 1. The gaps the collection is aimed at

Three kinds of gap appear in the literature. **Method gaps** are closed by
software and are largely closed already in v0.1. **Data gaps** need public data
collected and paired. **Sensor gaps** need data nobody publishes, so they
need a self-collected site.

| Gap | In one sentence | Kind | Closed when | Status |
|---|---|---|---|---|
| **G-1** | No large, diverse, synchronised real-world image + instrument dataset | Data | ≥ 5 independent sites in ≥ 2 regions, ≥ 1 year each, every label from an instrument | NDBC buoy cameras (79, prospective), Waimea (complete), WebCOOS (collecting); CoastCam not started |
| **G-3** | No cross-station test with instrument labels | Data | A published benchmark with whole stations held out, with baselines | Waiting for ≥ 3 months of buoy-camera archive (about 2026-12-26) and CoastCam |
| **G-10** | How much label error a distant reference adds is assumed, not measured | Data | Measured label-error floor per site | Measured between buoys (2026-10-01); per camera site still open |
| **G-4** | Night, glare, fog and storms are filtered out rather than studied | Data | Results reported per condition with enough samples in each | Buoy cameras collect at night; thermal would help |
| **G-6** | Spectral targets S(f), S(f,θ) almost untried from imagery | Sensor | A pipeline that recovers S(f) from imagery, checked against an instrument | Needs video (WebCOOS pilot) and stereo (Guimarães 2020; self-collected) |
| **G-5** | Thermal IR imagery with a wave instrument essentially absent | Sensor | Radiometric thermal + instrument record across sea states, day and night | Self-collected only |
| **G-9** | No real image + platform-attitude (IMU) dataset | Sensor | Frames with hardware-timestamped attitude + co-located wave reference | Horizon proxy on buoy cameras; self-collected otherwise |
| G-2, G-7, G-8 | Label provenance, inconsistent metrics, no reproducibility | Method | Schemas, evaluation, provenance in code | Done in v0.1; ADR 0010 adds the label rule |

The collection targets **G-1, G-3 and G-10 first** (public data, cheap, needed
for the first paper), then **G-6, G-5 and G-9** (need new sensors).

## 2. What each data stream contributes

● primary contribution · ○ partial

| Stream | Imagery | Reference and distance | G-1 | G-3 | G-10 | G-4 | G-6 | G-5 | G-9 | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| NDBC buoy cameras | 6-view stills, hourly, day and night | Same hull, 0 km | ● | ● | ● | ● | | | ○ horizon | Collecting on two hosts with an offsite copy; 638 camera-hours lost 2026-10-04/05 (see P0) |
| PacIOOS Waimea Bay | 2 shore views, hourly, daytime, 2009–2013 | CDIP 106, 6.3 km, same exposure | ● | ○ one site | ○ | ○ | | | | Paired; labels per ADR 0010 |
| WebCOOS stills | 1 per 30 min, daylight | CDIP 433 / 243, 19–22 km, open coast | ● | ○ | ○ | | | | | Collecting since 2026-07 |
| WebCOOS video | Clips, 5–20 fps | as above | ○ | | | | ● | | | Pilot proposed ([webcoos.md](../datasets/webcoos.md)) |
| USGS CoastCam | Calibrated snapshots, timex, timestacks | CDIP 15–40 km; Dorado Spotter ~2 km | ● | ● 5 sites | ○ | | ○ | | | Not started |
| Guimarães 2020 stereo | Stereo video 10–15 Hz | Wave wires, AWAC, radar on the platform | | | | | ● validation | | | Not started |
| Buoy-pair analysis | none | Buoys 0.5–100 km apart | | | ● | | | | | Done for 2023 |
| Self-collected site | Video, stereo, thermal (phased) | Instrument in view | ● | | ● | ● | ● | ● | ○ / ● on a vessel | Design (§4) |

## 3. Priorities, in order

Each item says why, what, when it is done, and what it depends on.

### P0. Protect the stream that cannot be re-collected (now)

NDBC deletes superseded buoy-camera images after about 72 h, and nobody else
archives them. In the first two weeks the collector host was **down three
times** ([incidents](data_collection.md#incidents)): about 40 h and 10 h
(2026-09-29 to 10-01, both recovered by the 70 h backfill) and about 76 h
(2026-10-04 20:43 to 10-08 00:32 UTC), which **lost 638 camera-hours for
good** (about 8 h of the whole network).

Actions:

1. ✅ Check coverage after every outage:
   `uv run python scripts/buoycam_coverage.py --days 4`.
2. ✅ Offsite copy: Cloudflare R2 since 2026-10-08, synced daily
   ([data collection](data_collection.md#setting-up-r2-one-time-done-by-the-account-owner)).
3. ✅ Second collector host (a laptop) since 2026-10-08: an image is lost only
   if both hosts are down past the backfill window
   ([a second collector host](data_collection.md#a-second-collector-host)).
4. ✅ Backfill fetches the oldest hours first, so an interrupted recovery
   loses nothing it could have saved (2026-10-08).
5. ✅ Alert when no run has completed for 3 h: a GitHub Actions watchdog
   opens an issue ([alerts](data_collection.md#alerts), 2026-10-08).
6. Still open: an **always-on host** (university server or a small cloud VM).
   A laptop helps only while it is awake and logged in, and the desktop
   dual-boots, so time in the other system is collector downtime.

**Done when** two weeks pass without a gap that is not `not_published`, and
`offsite.last_sync` is less than a day old. The two weeks restart on
2026-10-08.

### P1. First benchmark: Waimea Bay (2–3 weeks)

The only historical pairing that needs no collection. It gives the first
numbers and exercises the full pipeline.

1. ✅ Re-run `scripts/waimea_pairs.py` with the ADR 0010 labels (done
   2026-10-01 on the full archive; H_s window mean, T_p and D_p nearest).
2. Build the manifest with splits by **HST day across both cameras**, plus a
   cross-time split by year with an embargo.
3. Report the required baselines before any image model: climatology,
   persistence, the ERA5 nowcast at the site, and ridge regression on image
   statistics (white-water fraction).
4. Report every metric under window-mean and nearest labels (Q-M7).

**Done when** a manifest passes the leakage audit and the baseline table
exists. Depends on P0 only for the host.

### P2. WebCOOS permission and video pilot (send this week)

One email to webcoos@secoora.org covering: (a) historical stills 2021–2026 at
one per 30 min (about 4 GB per camera-year), (b) the video archive inventory
and a 2–4 week pilot of clips synchronised with CDIP records on two cameras,
(c) whether derived datasets may include frames. Pilot details:
[webcoos.md](../datasets/webcoos.md#video-archive-pilot-not-yet-collected).

**Done when** the answer is recorded in `webcoos.md` and the pilot's measured
volume sets the long-term cadence.

### P3. USGS CoastCam adapter (v0.2–0.3, 3–6 weeks)

Five calibrated sites on four coasts make the first genuine leave-site-out
test (G-3) and enable orthorectification. Dorado has a Spotter about 2 km
offshore, the best reference of any public camera. Start with Madeira Beach
(data release with calibration) and Dorado (NIMS S3 snapshots + Spotter).
Check each site's reference for exposure with `scripts/buoy_pair_agreement.py`
where a neighbouring buoy exists.

### P4. Buoy-camera benchmark (from about 2026-12-26)

After 3 months of archive, fix the held-out stations in
`configs/experiments/baseline_hs.yaml` (choose by region, so that no test
station has a training neighbour at the same exposure), then test the
physics features first: whitecap fraction with the buoy's wind (Q-M4) and
6-view horizon tilt (Q-M5).

### P5. Self-collected site (design now, deploy in phases)

Long lead times (permits, procurement, mounting), so design starts now. See §4.

### P6. Stereo validation of the spectral pipeline

Process the Guimarães et al. (2020) stereo records (CC BY 4.0) into spectra
and check that the project's spectral code reproduces the published
statistics. This also builds the stereo skills the self-collected site will
need.

## 4. Self-collected site: what to build, in priority order

The public data cannot close G-5, G-6 or G-9, and every public shore camera
is uncalibrated, low-rate or far from its instrument. A self-collected site
can be designed around those limits. The components below are in priority
order: each one is worth having only if the ones above it are in place.

**Site.** FAU Harbor Branch's campus is on the Indian River Lagoon, which has
no ocean waves, so the camera has to view the open Atlantic from the barrier
island (beach, a jetty at Fort Pierce Inlet, or a tall building). Confirm the
site, the access and the permits first. The 14.6 km to CDIP 134 (WMO 41114)
quoted in the landscape review is from the approximate campus position in
`camera_sites.yaml`. From the ocean beach at the same latitude it is about
9–11 km, and from Fort Pierce Inlet about 13 km, so measure it once the site
is chosen. The coast there is straight and open. Open-coast buoy pairs 10–30 km
apart agree within about 4–10 % ([synchronisation](../methodology/synchronization.md)),
so CDIP 134 is a usable offshore reference, though not a local one.

### 4.1 Calibrated RGB video with disciplined time (essential)

- **Why:** without video there is no wave motion, so no period or direction
  from physics and no temporal model (G-6). Without calibration, pixels cannot
  be turned into metres. Without a trusted clock, labels cannot be matched:
  Waimea lost 638 images to a wrong camera clock.
- **Specification:**
  - at least 1080p at 10 fps or more, so that a 3-s wave gets 30 frames;
  - fixed manual exposure where possible;
  - high-bitrate or lossless recording of the analysis clips;
  - mounted as high as possible (10–30 m), looking along the wave direction;
  - intrinsic calibration (checkerboard), extrinsic calibration from ground
    control points surveyed with RTK GNSS, and a re-check after any
    disturbance;
  - clock disciplined by GPS (PPS or GPS-backed NTP), with the offset logged.
- **Recording:** clips aligned with the reference records (e.g. 20 min in
  every 30 min during daylight), each with an `ImageRecord`: capture window,
  time source and uncertainty, exposure, calibration IDs.
- **Closes:** G-1 (a calibrated site), G-6 (with stereo), G-4 (with
  annotation), and a camera that matches the plain cameras a model would be
  deployed on.

### 4.2 A wave instrument in or near the field of view (essential)

- **Why:** between buoys, a reference within 10 km adds about 6 % mismatch.
  At 10–30 km the median is 15 %, ranging from 4 % on an open coast to 60 %
  across an island. A camera label cannot be better than its reference, and
  only an instrument in view removes the exposure question.
- **Options:** a moored Spotter-class buoy inside the view (also gives a
  visible target and its own GPS position), a bottom pressure sensor or ADCP
  (AWAC) in the surf approach, or both. Log raw displacement if possible, so
  that $H_{1/3}$, $H_{\max}$ and $T_z$ can be computed as well as $H_{m0}$.
- **Closes:** G-10 at this site, and the label-provenance requirements of G-2.

### 4.3 A stereo pair (high value)

- **Why:** stereo measures the sea-surface elevation over an area, so height,
  steepness and spectra come from geometry, not learning (G-6). It also
  provides the ground truth for the mono camera at every pixel, not only at
  the buoy.
- **Specification:**
  - two identical cameras, hardware-triggered (synchronised to well under
    1 ms), 10–15 Hz;
  - a rigid baseline of several metres, mounted high;
  - stereo calibration;
  - processing with WASS (open source).
- **Limits:** covers a few hundred metres from the cameras, needs surface
  texture (flat calm and strong glare are hard), and produces large data
  volumes.
- **Closes:** G-6, and the validation of 4.1 against geometry.

### 4.4 Radiometric thermal (LWIR) (high value for G-4/G-5)

- **Why:** works at night and shows active breaking, the conditions most
  often excluded (G-4, G-5).
- **Specification:**
  - a radiometric camera (calibrated brightness temperature, not
    colour-mapped video), at least 640 × 512;
  - co-aligned with the RGB camera and on the same clock;
  - emissivity and atmospheric-correction settings logged
    (`ThermalMetadata`).
- **Limits:** low resolution, high cost, and no height on its own. Stereo
  matching on thermal images is hard, so use it as a separate, co-registered
  camera.

### 4.5 IMU + GNSS for a moving platform (only for G-9)

- **Why:** G-9 needs images with the camera's attitude measured at each frame.
  A fixed shore rig does not need this (a cheap IMU on the mast still detects
  vibration and slow tilt drift of the calibration).
- **Specification:** an IMU/GNSS unit at 50–100 Hz, timestamped by the same
  clock as the frames (hardware trigger or PPS), on a vessel or buoy with a
  wave reference. A research-vessel cruise with a co-located wave buoy is the
  simplest first step.

### Phasing

| Phase | Components | Gaps | Gate |
|---|---|---|---|
| A | 4.1 with CDIP 134 as reference | G-1, G-4 | Site and permits confirmed |
| B | add 4.2 (instrument in view) and 4.3 (stereo) | G-6, G-10 | Phase A clips recover period and direction against CDIP 134 |
| C | add 4.4 (thermal) | G-5, G-4 | Phase B running reliably |
| D | 4.5 on a vessel campaign | G-9 | Independent of A–C |

## 5. Next steps checklist

| When | Step | Done when |
|---|---|---|
| **This week** | P0.1 coverage check; P0.2 offsite copy; P2 WebCOOS email | No unexplained gaps; `offsite.last_sync` set; email sent |
| 2 weeks | P0.3 always-on host + alert; P1.1–P1.2 Waimea labels and manifest | Manifest passes the audit |
| 1 month | P1.3–P1.4 Waimea baselines; WebCOOS video inventory | First baseline table |
| 1–3 months | P3 CoastCam (Madeira, Dorado); P6 stereo check; P5 site survey and permits | Two CoastCam sites paired |
| ~2026-12-26 | P4 buoy-camera cross-station benchmark | Held-out stations fixed |
| 3–12 months | P5 phases A–C | Phase A clips paired with CDIP 134 |

## 6. Decisions the team has to make

1. **Collector host:** keep the current machine or move to a university or
   cloud server (P0.3).
2. **Offsite store:** Cloudflare R2 (needs a payment method on file) or FAU
   Research Computing storage.
3. **WebCOOS scope:** which cameras, which years, whether to ask about video
   (P2).
4. **Self-collected site:** location on the ocean shore, budget for phases A
   and B, who holds the permits.
5. **Moving platform (G-9):** whether a vessel campaign is in scope, and with
   which wave reference.
