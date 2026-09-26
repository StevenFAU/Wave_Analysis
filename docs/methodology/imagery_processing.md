# Imagery Processing

Code: `qc/imagery.py`, `processing/solar.py`, `schemas/imagery.py`,
`sources/ndbc/buoycam.py`.

## Annotate, don't filter

The reviewed studies filtered their visual data at collection time: Choi et al.
(2020) excluded dark and over-exposed frames by brightness thresholds, and Kim
et al. (2023) collected only 07:00–11:00, converted to grayscale, and removed
fog, rain, and ships. This project keeps all frames and records the conditions,
because storms, spray, darkness, and glare are when wave monitoring matters most
and when vision is least reliable. They are evaluation strata, not noise.

## Per-frame annotations

| Annotation | Definition | Use |
|---|---|---|
| `mean_luma`, `std_luma` | BT.709 luma statistics | exposure, contrast |
| `dark_fraction` | fraction of pixels with luma < 0.08 | night/under-exposure |
| `saturation_fraction` | luma ≥ 0.98 | clipping |
| `glare_fraction` | luma ≥ 0.92 **and** chroma ≤ 0.08 | specular glint / whitened sky (a proxy, see below) |
| `sharpness` | variance of 4-neighbour Laplacian | blur, droplets, fog |
| `entropy_bits` | 256-bin luma entropy | information content |
| solar elevation/azimuth | NOAA solar-position algorithm | illumination category |
| `illumination` | day / low_sun / civil_twilight / nautical_twilight / night | stratum |
| `glint_risk` | sun inside camera HFOV at elevation < 35° | geometric glint prior |
| `dhash` | 64-bit difference hash | near-duplicate / leakage detection |

The glare proxy is validated only qualitatively so far. On NDBC 46026
(2026-09-26 00:10 UTC, late afternoon local) the view facing the sun scores
`glare_fraction` 0.55 (saturation 0.44), while the other five views score
0.00–0.08. All six views of the night image at 41010 have `dark_fraction` ≈ 1.0.
Breaking-wave foam is also bright but keeps texture and some chroma. A labelled
validation set is an open task. Compute annotations **per camera view**
(`buoycam.split_views`), since one strip can mix glare, clear sea, and structure.

## Burned-in text and other identity leaks

NDBC buoycam strips carry a caption (station ID, UTC time, heading labels) in
the bottom rows. A network can read it, and in a cross-station benchmark it
can then identify the site. **Mask the caption band before training** (30 rows,
measured on 47/47 daylight images; `split_views` crops it).
More generally, watermarks, timestamps, logos, and fixed structures (buoy
masts, piers) are shortcuts. Audit saliency maps (Grad-CAM) for attention on them.

## Colour and resolution

Keep RGB. The grayscale conversion of Kim et al. discards chroma, which is
informative about whitecaps, water colour, and illumination. Model-specific
resizing and cropping are experiment parameters, not preprocessing. Store the
original resolution in `ImageRecord`.

## Geometry

For calibrated cameras (CoastCam, Argus-type systems), intrinsics and
extrinsics are stored by ID. Orthorectification to a metric grid is planned for
v0.3. It is the prerequisite for physically scaled features (wavelength,
celerity) and for fixing direction biases like the +17.9° reported by Kamagata
et al. (2026), which they attribute to missing bearing calibration.

## Moving platforms

For ship or buoy cameras, image motion mixes wave motion and platform motion.
Yang et al. (2026) feed IMU attitude as $(\sin, \cos)$ features
(`evaluation.circular.encode_angles`, `models.multimodal.yang2026.encode_attitude`).
For NDBC buoys without a published IMU stream, horizon detection is a candidate
proxy for roll and pitch (open question in `docs/datasets/ndbc.md`).
