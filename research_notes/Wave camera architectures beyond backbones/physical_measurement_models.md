# Measurement-style deep architectures (tracks, depth, 3-D geometry, sea-surface elevation) for wave height, period and direction from a shore camera or a buoy camera

Scope note: current as of 2026-11-18. Items released June–November 2026 are flagged **[NEW 2026-H2]** and were checked against their arXiv abstract pages unless marked otherwise. "Inferences" sections hold the pipeline fit, failure modes and H200 compute reasoning (part f). That material is my reasoning from the cited facts plus standard linear wave theory, not something a source states.

---

## Q1. Which point trackers handle long (minutes), dense tracking on non-rigid, low-texture water/foam, and what accuracy, speed and licence do they have?

### Takeaway
No point tracker has been validated on ocean waves or surf-zone foam. The closest validated work uses learned optical flow (RAFT/SEA-RAFT) on river surfaces, where it roughly matches LSPIV. For a 4–10 min, 20 fps shore clip, the best-fitting open tools are AllTracker (MIT licence, dense flow to every frame, runs at 768×1024 on a 40 GB GPU) and TAPNext++ (Apache 2.0, online, trained on 1024-frame clips, 193 FPS for 256 query points). CoTracker3 is accurate but CC-BY-NC.

### Cited Findings
**CoTracker / CoTracker3 (Meta)**
- Licence: "The majority of CoTracker is licensed under CC-BY-NC". Bundled LocoTrack and TAP-Vid code is Apache 2.0, and Particle Video Revisited code is MIT. — [co-tracker GitHub README](https://github.com/facebookresearch/co-tracker)
- Variants: CoTracker3 **online** is a sliding-window model trained with a 16-frame window and run chunk by chunk, which the README recommends for longer videos because it uses less memory. CoTracker3 **offline** uses a single 60-frame window (window_len=60 at evaluation) and assumes the video length is known. The README states no maximum video length. — [co-tracker README](https://github.com/facebookresearch/co-tracker)
- README benchmark, δ_avg^vis (online / offline): TAP-Vid-Kinetics 68.3 / 67.8; TAP-Vid-DAVIS 76.7 / 76.9; RoboTAP 78.8 / 78.0; RGB-Stacking 82.7 / 85.0. Earlier versions for comparison: CoTracker2 scored 61.8 on Kinetics and 74.6 on DAVIS. — [co-tracker README](https://github.com/facebookresearch/co-tracker)
- Joint tracking capacity: CoTracker2 can track "up to 265*265" points jointly — [co-tracker README](https://github.com/facebookresearch/co-tracker). The original CoTracker paper reports tracking 70k points jointly on a single GPU — [CoTracker arXiv 2307.07635 (via arxiv-vanity)](https://www.arxiv-vanity.com/papers/2307.07635)
- A third-party Hugging Face mirror reports different numbers under a different protocol (256×256 frames, every third video; online variant AJ 70.34), so its figures are not comparable with the official table. — [fastforks/cotracker HF](https://huggingface.co/fastforks/cotracker)

**TAPIR / BootsTAPIR / TAPNext / TAPNext++ (Google DeepMind, `tapnet` repo)**
- Licence: code **and** all released checkpoints are Apache 2.0. TAP-Vid annotations are CC BY 4.0. — [google-deepmind/tapnet README](https://github.com/google-deepmind/tapnet)
- AJ (%) on DAVIS-first / DAVIS-strided / Kinetics-first / RoboTAP-first: TAPIR 58.5 / 63.3 / 50.0 / 59.6; Online TAPIR 56.2 / 58.3 / 51.2 / 59.1; BootsTAPIR 62.4 / 67.4 / 55.8 / 69.2; Online BootsTAPIR 59.7 / 61.2 / 55.1 / 69.1; TAPNext (TrecViT-B, 256²) 65.25 / 68.9 / 57.3 / 64.1; TAPNext++ (256²) 65.6 / – / 53.9 / 61.1; TAPNext++ (512²) 67.0 / 71.2 / – / –. — [tapnet README](https://github.com/google-deepmind/tapnet)
- BootsTAPIR works best at 512×512, and TAPIR can benefit from inputs above 512×512. The live online demo runs at about 17 fps on 480×480 frames on a Quadro RTX 4000. — [tapnet README](https://github.com/google-deepmind/tapnet)
- TAPNext (ICCV 2025, arXiv 2504.05579) frames tracking as causal, online masked-token decoding with no tracking-specific inductive biases, and reports state-of-the-art online tracking at low latency. — [arXiv 2504.05579](https://arxiv.org/abs/2504.05579); [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zholus_TAPNext_Tracking_Any_Point_TAP_as_Next_Token_Prediction_ICCV_2025_paper.html)
- TAPNext++ (arXiv 2604.10582, submitted 12 Apr 2026) found that the original TAPNext fails on long sequences and on re-detection. The fixes are training and data changes: training on 1024-frame sequences with sequence parallelism, and a synthetic Kubric-1024 set of 10,000 videos × 1024 frames. On PointOdyssey, survival rate rose from 13.0% to 67.9%. The authors report 193 FPS for 256 query points and say effective memory extends from about 150 frames (BootsTAPNext) to more than 4,000 frames. These details come from secondary summaries; the README states "40x longer stable tracking". — [alphaXiv 2604.10582](https://www.alphaxiv.org/abs/2604.10582); [arXiv HTML 2604.10582](https://arxiv.org/html/2604.10582); [tapnet README](https://github.com/google-deepmind/tapnet)

**AllTracker (Stanford/TRI/CMU, ICCV 2025)**
- Estimates a dense flow field from a query frame to **every** other frame using a sliding window, at high resolution. It has about 16M parameters and is trained jointly on optical-flow and point-tracking data. It runs at 768×1024 on a 40 GB GPU, a setting where DELTA and DTF run out of memory. — [arXiv 2506.07310](https://arxiv.org/html/2506.07310v2)
- Licence: MIT. README benchmark (δ_avg / AJ / OA): default 76.3 / 63.3 / 90.0; 448×768 78.8 / 65.9 / 90.2; 768×1024 80.6 / 67.2 / 89.7. The README describes it as relating "one frame to hundreds of subsequent frames". — [aharley/alltracker GitHub](https://github.com/aharley/alltracker)

**Other dense / long trackers**
- DOT (CVPR 2024) unifies point tracking and optical flow. It is reported to be on par with or better than CoTracker while running at least two orders of magnitude faster. — [DOT CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Le_Moing_Dense_Optical_Tracking_Connecting_the_Dots_CVPR_2024_paper.pdf)
- TAPTRv3 (arXiv 2411.18671) targets long video and the drift problem in TAPTRv2's RNN-like temporal modelling. — [arXiv 2411.18671](https://arxiv.org/pdf/2411.18671)
- AnthroTAP (CVPR 2026) trains LocoTrack and TAPNext on pseudo-labels from human motion and reports state of the art on TAP-Vid. — [AnthroTAP CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Kim_AnthroTAP_Learning_Point_Tracking_with_Real-World_Motion_CVPR_2026_paper.pdf)
- **[NEW 2026-H2]** TrackEverything (arXiv 2609.30222, submitted 24 Sep 2026; Jain, …, Engel, Fragkiadaki, Harley). It is a 3-D dense tracker that represents a video as persistent 3-D scene tracks in world coordinates. It tracks all visible points in videos longer than 1,000 frames within 40 GB of GPU memory. On TAPVid-3D it beats open-source all-frame dense 3-D trackers by more than 20% APD on short clips and is "competitive" with sparse trackers on long ones. The abstract page does not confirm a code or weights release. — [arXiv 2609.30222](https://arxiv.org/abs/2609.30222)
- **[NEW 2026-H2, abstract not fetched]** TAPVid-MV (arXiv 2609.01899) is a benchmark for 3-D point tracking across multiple views. Its authors report that multi-view trackers do not consistently beat monocular ones and that geometry recovery is the main bottleneck. — [arXiv 2609.01899](https://arxiv.org/abs/2609.01899)

**Behaviour on water (closest evidence: river surface velocimetry)**
- SEA-RAFT agreed strongly with LSPIV for river surface velocity on the Naka and Zengwen rivers under flood conditions. Both methods underestimated velocity in low-speed regions. — [Kyoto Univ. DPRI abstract](https://www.dpri.kyoto-u.ac.jp/hapyo/26/pdf/C204.pdf)
- RAFT generalised well from particle-image training to lab and aerial river footage. A physics-informed alternative struggled on drone data. — [Tlhomole et al., J. Hydrology (Imperial record)](https://spiral.imperial.ac.uk/entities/publication/5ce1094f-d59f-4ea3-95b7-12efb7d79091)
- RivVideoFlow, a multi-frame method built on VideoFlow, outperforms two-frame RAFT on rivers by exploiting temporal continuity. DeepFlow (non-learned) matched PIV only when surface texture was visible. — [search summary of river-velocimetry literature; see Water 18(4):468](https://www.mdpi.com/2073-4441/18/4/468); [Glasgow eprint](https://eprints.gla.ac.uk/309906/2/309906.pdf)
- Classical optical flow on surf-zone **foam** from a single fixed camera at Anglet, France, gave wave-filtered surface currents that agreed reasonably with a current profiler (Rodríguez-Padilla et al., Remote Sensing 2021). — [DOAJ record, Remote Sensing May 2021](https://doaj.org/article/6873e13a10574ea998be06627553991b)
- No paper applying CoTracker, TAPIR, TAPNext or AllTracker to ocean waves, foam or PIV-style wave kinematics turned up. — searches summarised above

### Inferences
- **Fit to the WebCOOS clips.** A 10 min clip at 20 fps is 12,000 frames, about 3× TAPNext++'s stated 4,000-frame effective memory and about 12× TrackEverything's demonstrated length. Any tracker will need chunking with re-seeded queries. Re-seeding is also physically appropriate, because foam is created and destroyed on wave-period timescales (5–20 s, roughly 100–400 frames at 20 fps). Minute-long single tracks of the same foam patch are physically implausible in the surf zone.
- **Ranking for this use.** (1) AllTracker: dense, MIT licence, high resolution, query-to-all-frames flow. This is the most PIV-like option and closest to the river-velocimetry evidence. (2) TAPNext++ / BootsTAPIR (online): Apache 2.0, causal, fast, sparse queries on crests and foam. (3) CoTracker3: probably comparable accuracy, but the CC-BY-NC licence restricts reuse.
- **Physics caveat.** In the surf zone, foam is advected by bores and undertow and does not follow closed linear orbits. Image-plane displacement mixes horizontal advection with vertical surface motion, so "vertical oscillation amplitude" needs camera height and tilt (Q2) plus an assumption about the horizontal/vertical split. Outside the surf zone, on unbroken swell with sun-glint texture, deep-water linear theory gives horizontal ≈ vertical orbital amplitude at the surface. That could, in principle, constrain amplitude from horizontal tracks, which are better resolved in an oblique view.
- **Failure modes to expect.** Trackers drift on specular glints, which move with the sun-surface geometry rather than with the water. Textureless dark swell faces are also a problem. Breaking causes topology changes (foam appears or vanishes), and long-range occlusion behind crests needs re-detection, which is exactly the weakness TAPNext++ was built to fix.
- **H200 compute.** AllTracker at 768×1024 fits in about 40 GB, so it fits on GPUs with 40 GB or more free and possibly at 2688×1520 on the 117 GB GPU (not tested; memory scales with resolution). TAPNext++ at 193 FPS for 256 points means a 12,000-frame clip takes about 1 min per query set, though the source does not name the GPU those FPS were measured on. Temporal subsampling to 5–10 fps keeps 5–20 s waves well resolved.

### Gaps
- No published accuracy for any of these trackers on water, foam or ocean waves. TAP-Vid has no water-specific split, so TAP-Vid AJ does not predict performance here.
- CoTracker3 speed and memory figures (FPS, GB per point per frame) are not in the README. The TAPNext++ 193 FPS hardware and resolution were not confirmed from the primary PDF.
- LocoTrack's and BootsTAPIR's own track-length limits were not found.

---

## Q2. Can modern monocular or feed-forward 3-D models calibrate a fixed shore camera (height above sea, tilt, focal) well enough for metric conversion, and how do they fail on water?

### Takeaway
Several models now output focal length or intrinsics, and some output gravity, roll and pitch, from one image: Depth Pro, MoGe-2, UniDepthV2, GeoCalib, Depth Anything 3, VGGT and MapAnything. None has published accuracy on open-sea or coastal scenes. Camera **height above the sea** is not directly estimated by any of them; it is only implicit in metric depth, which is the weakest output on water. Depth models are known to be fooled by water reflections. The validated coastal practice remains GCP or feature-based calibration with automated re-calibration and stabilisation (UCalib, Morichon et al.). Learned models are best used as initialisers or checkers, applied to the **land** part of the frame.

### Cited Findings
**Single-image calibration and metric depth**
- GeoCalib (ECCV 2024, ETH/Microsoft) estimates focal length, distortion and **gravity direction** (roll and pitch) from one image. It fits a predicted per-pixel perspective field with Levenberg–Marquardt and outputs uncertainties that flag failures. It is designed for scenes without lines, where vanishing-point methods fail. Code and models are at github.com/cvg/GeoCalib. — [arXiv 2409.06704](https://arxiv.org/html/2409.06704v2); [ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5636_ECCV_2024_paper.php). Reported gravity AUC@1° of 83.1% on Stanford2D3D comes from an aggregator only — [Moonlight review](https://www.themoonlight.io/review/geocalib-learning-single-image-calibration-with-geometric-optimization)
- Depth Pro (Apple, ICLR 2025) predicts metric depth in metres and **focal length in pixels** from one image, without intrinsics metadata, and claims state-of-the-art single-image focal estimation. A known focal length can also be passed in. Weight licence: apple-ascl (Apple sample-code licence); one secondary source reads it as research use only. — [arXiv 2410.02073](https://arxiv.org/abs/2410.02073); [apple/ml-depth-pro](https://github.com/apple/ml-depth-pro); [HF model card](https://huggingface.co/apple/DepthPro/blob/main/README.md)
- MoGe-2 (NeurIPS 2025, Microsoft) adds metric scale to MoGe's affine-invariant point maps with a separate MLP head on the CLS token, and outputs camera FOV. Code is MIT (DINOv2 parts Apache 2.0); the weights licence was not confirmed. — [arXiv 2507.02546](https://arxiv.org/html/2507.02546v1); [gitee mirror of repo](https://gitee.com/minttang/MoGe)
- UniDepthV2 predicts camera intrinsics, metric depth and per-pixel rays, plus per-pixel uncertainty, from one image, and optionally accepts known intrinsics. Its outdoor zero-shot tests are ETH3D, Sintel, DDAD and nuScenes, so there is no maritime test. — [arXiv 2502.20110](https://arxiv.org/html/2502.20110v2); [UniDepth GitHub](https://github.com/lpiccinelli-eth/UniDepth)
- Metric3D v2 is a zero-shot metric depth and surface-normal foundation model. Whether it predicts intrinsics or requires them was not confirmed. — [UniDepth/Metric3D search summary; arXiv 2404.15506](https://arxiv.org/pdf/2404.15506v1.pdf)
- Depth Anything 3 (arXiv 2511.10647, 13 Nov 2025) recovers geometry and camera pose from any number of views. The v1 paper reports +35.7% pose accuracy and +23.6% geometric accuracy over VGGT; a secondary review gives 44.3% / 25.1%, so the sources disagree. DA3METRIC-LARGE (0.35B, **Apache 2.0**) gives monocular metric depth via `metric_depth = focal * net_output / 300`, so it needs focal length in pixels. The API returns extrinsics and intrinsics. "-1.1" checkpoints were retrained after a training bug, and DA3-Streaming was released in Nov 2025. — [arXiv 2511.10647](https://arxiv.org/pdf/2511.10647); [DA3METRIC-LARGE card](https://huggingface.co/APaul1/DA3METRIC-LARGE); [PyPI depth-anything-3](https://pypi.org/project/depth-anything-3/); [gitee mirror README](https://gitee.com/ayang2019/Depth-Anything-v3)
- Video Depth Anything (CVPR 2025) produces **affine-invariant**, not metric, depth for arbitrarily long videos, using a spatio-temporal head on DA-V2 and a temporal-gradient consistency loss. It reports AbsRel, δ1 and Temporal Alignment Error. Licence not found. — [arXiv 2501.12375](https://arxiv.org/pdf/2501.12375)

**Feed-forward multi-view 3-D**
- VGGT outputs intrinsic and extrinsic matrices (OpenCV convention) from a camera head. The original VGGT-1B weights are CC-BY-NC-4.0. A separate VGGT-1B-Commercial checkpoint requires an application form, and the code was relicensed in July 2025 to allow commercial use with military use excluded. No focal-error benchmark was found. All of these points come from mirrors and guides, not the primary licence file. — [fastforks/vggt](https://huggingface.co/fastforks/vggt); [LearnOpenCV guide](https://learnopencv.com/vggt-vs-vggt-%CF%89-vggt-omega-a-complete-guide-to-feed-forward-3d-reconstruction/)
- VGGT-Ω (arXiv 2605.15195, 14 May 2026, CVPR 2026 oral) reports state of the art on three static and three dynamic reconstruction benchmarks. A guide updated 3 Sep 2026 says the released materials are noncommercial research only. — [arXiv 2605.15195](https://arxiv.org/abs/2605.15195); [CVPR 2026 oral](https://cvpr.thecvf.com/virtual/2026/oral/40381); [LearnOpenCV](https://learnopencv.com/vggt-vs-vggt-%cf%89-vggt-omega-a-complete-guide-to-feed-forward-3d-reconstruction/)
- π³ (ICLR 2026) is reference-view-free feed-forward geometry. The GitHub repo gives the code as BSD 3-Clause and the Pi3/Pi3X weights as CC BY-NC 4.0, but the Pi3 HF card says BSD-2-Clause, so the sources disagree. — [yyfz/Pi3 GitHub](https://github.com/yyfz/Pi3); [Pi3 HF README](https://huggingface.co/yyfz233/Pi3/blob/main/README.md); [Pi3X HF README](https://huggingface.co/yyfz233/Pi3X/blob/main/README.md)
- MapAnything (3DV 2026) is universal feed-forward **metric** 3-D reconstruction. It accepts optional intrinsics (not together with ray directions) and predicts them. facebook/map-anything is CC-BY-NC 4.0, and facebook/map-anything-apache is Apache 2.0. — [arXiv 2509.13414](https://arxiv.org/abs/2509.13414v3); [MapAnything 3DV 2026 PDF](https://richardt.name/publications/MapAnything-KeethaEtAl-3DV2026.pdf)
- CUT3R (CVPR 2025, Berkeley/DeepMind) is a stateful recurrent model that outputs online **metric-scale** pointmaps for each new frame of a stream. Licence not found. — [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Continuous_3D_Perception_Model_with_Persistent_State_CVPR_2025_paper.pdf)

**Known failures on water**
- End-to-end depth networks are misled by water reflections; for example, DORN gives blurred, misleading depth on water scenes. The only prior symmetry-based method assumed the reflection plane is parallel to the horizon. Lu & Chen (2024) segment water and treat the reflection as a second view. Their WRS benchmark is **synthetic** (3,251 Unreal Engine 4 RGB-D pairs). — [arXiv 2404.07176](https://arxiv.org/html/2404.07176v1)

**Validated coastal-camera calibration practice (non-deep or classical)**
- UCalib (Simarro et al., Remote Sensing 2021) automatically calibrates Argus-type cameras from a basis set of manually calibrated images. It succeeded on more than 90% of images for feature-rich views but only about 40% for the most featureless camera. The authors recommend adding the horizon line to the basis when water is analysed. — [DOAJ, Remote Sensing Jul 2021](https://doaj.org/article/a4fc27e698434beeac46bb6812ab5c92)
- Over five years at Anglet, keypoint-based stabilisation found camera tilt drifts of up to 30 px, and viewing-angle drift could cause rectification errors of up to **400 m at 2.5 km**. The method processed at least 90% of frames automatically. — [scite summary, Morichon et al.](https://scite.ai/authors/denis-morichon-zb9GP5); [HAL, Remote Sensing 12:70](https://hal-univ-pau.archives-ouvertes.fr/hal-02433234/file/remotesensing-12-00070.pdf)
- A self-fed stabilisation method needing a single calibration cut shoreline position error by 40%, with larger gains farther from the camera. — [AMiner, JMSE 2022](https://aminer.org/pub/622a538b5aee126c0fe021ff/shoreline-detection-accuracy-from-video-monitoring-systems)
- Horizon tracking (Schwendeman & Thomson, JTECH 2015) recovers pitch and roll for ship-video rectification and detected the horizon in 92% of frames that contained one. — [JTECH 2015 PDF](https://data-ww3.ifremer.fr/BIB/Schwendeman_Thomson_JTECH2015.pdf)
- Synthetic-video depth inversion accuracy depends on the azimuth between the camera look direction and wave approach, with best contrast when looking along the propagation direction. — [Perugini et al. 2019 postprint](https://iris.univpm.it/bitstream/11566/269388/2/Perugini_etal_2019_CE_postprint.pdf)

**Buoy-mounted cameras**
- The horizon method cannot resolve the camera's vertical heave, which limits it for direct wave height. — [UW thesis](https://digital.lib.washington.edu/researchworks/bitstreams/26f710f4-e096-489e-b923-eeb330cf38e0/download)
- A buoy camera viewing a fixed landmark used PnP pose to obtain wave height at the buoy. It matched magnetic wave sensors in a flume and ran at 40 fps on a Raspberry Pi 4. — [PNNL Tethys record](https://tethys-engineering.pnnl.gov/publications/using-vision-based-wave-height-sensors-estimate-ocean-wave-field)
- On a moored discus buoy, platform motion contaminated the images and made whitecap tracking difficult. — [NTNU/SINTEF poster](https://www.sintef.no/globalassets/project/deepwind2014/presentations/posters/bakhoday_ntnu_field-measurements.pdf)

### Inferences
- **What the learned models can give a fixed shore camera.** Focal length (Depth Pro, MoGe-2 FOV, UniDepthV2, GeoCalib, DA3 or VGGT intrinsics) and roll and pitch (GeoCalib gravity, or the horizon line). Height above mean sea level is the parameter that sets metres per pixel on the water. It can only come from (a) metric depth to land features, where the models are trained and reliable, combined with known geometry, or (b) external survey and GCPs.
- **Geometry check.** The horizon cannot separate camera height from pitch. Horizon dip at height h is about √(2h/R): about 0.14° at 20 m, or about 5 px at a focal length of about 2,000 px. Any pitch error of that size swamps it. So pitch from GeoCalib or the horizon, plus height from a model, has a large unconstrained error. Error grows with range, as the 400 m-at-2.5 km stabilisation result shows.
- **Practical recipe for WebCOOS cameras that pan monthly.** Use per-view calibration from land GCPs or features, in UCalib style. Use the horizon (Hough or learned) for per-frame pitch and roll stabilisation. Use DA3, MapAnything or MoGe-2 on the **land mask only** to propose focal and metric scale as a sanity check against the GCP solution, and flag views where they disagree. Do not trust any model's depth on the sea surface itself: reflections, glint and texturelessness violate the training distribution.
- **Multi-view models give little here.** A fixed camera has no parallax, so VGGT, π³, MapAnything and CUT3R reduce to monocular inference and assume a static scene, which water is not. Month-to-month pans of the same camera could supply multiple views of the static land, which would be a legitimate use for multi-view calibration of a camera rig. That is untested.
- **Buoy stills.** The horizon gives buoy roll and pitch, related to wave slope, but not heave. Metric wave height from a single still would need metric depth to the waves, which is unvalidated on water. A slope-statistics route (Q3, polarimetry and Cox–Munk style) is more physically grounded than depth networks for this camera.
- **Licence summary for any model used in deliverables.** Apache 2.0: DA3METRIC-LARGE, MapAnything-apache, all tapnet checkpoints. MIT: AllTracker, MoGe code. Non-commercial: CoTracker3, VGGT-1B, VGGT-Ω, Pi3X weights, MapAnything research checkpoint. Apple ASCL: Depth Pro.

### Gaps
- No published accuracy for any learned focal, tilt or height estimator on coastal or maritime imagery, and no water-specific failure benchmark for DA-V2/DA3, Depth Pro, MoGe-2 or UniDepth. WRS is synthetic only.
- DUSt3R, MASt3R and MASt3R-SLAM licences and behaviour were not verified in this pass (believed CC BY-NC-SA, unverified). CUT3R and Video Depth Anything licences were not found. The MoGe-2 weights licence was not confirmed.
- GeoCalib focal-error numbers and licence were not extracted from the primary paper.
- DepthCrafter was not researched in this pass.

---

## Q3. What learned sea-surface reconstruction methods exist (stereo or monocular), and what are their errors against in-situ instruments?

### Takeaway
Metric sea-surface elevation still requires **stereo** (WASS/WASSfast, classical or learned stereo matching) or **polarimetric slope sensing**. Learned stereo, such as RAFT-Stereo and the Dec 2025 "neural reconstruction" network, has been shown on synthetic data or reported against unspecified references. Classical ship-borne stereo reaches Hs error ≤0.14 m (about ±10%) against buoys. No validated monocular deep method that outputs a metric elevation field was found. Monocular deep work regresses or classifies Hs and Tp directly, with ADCP, radar or buoy truth.

### Cited Findings
**Stereo pipelines**
- WASS (Bergamasco et al., Computers & Geosciences 2017) is an open-source, OpenCV-based fast dense stereo pipeline with disparity and point-cloud filtering. — [WASS paper PDF](https://www.dais.unive.it/wass/papers/1-s2.0-S0098300417304302-main.pdf)
- WASSfast (Computers & Geosciences 2021) estimates the surface spectrum directly from sparse matched features using the linear dispersion relation and NUFFT. It is more than an order of magnitude faster with accuracy close to state of the art. It has Predict-Update and **CNN** modes, though pip installs only the CNN mode because of TensorFlow version issues. Licence GPL v3; PyPI version 1.5.5. — [PyPI wassfast](https://pypi.org/project/wassfast); [IRIS record](https://iris.unive.it/handle/10278/3734121)
- A public sea-surface stereo dataset exists for resolving space-time wave fields (Scientific Data 2020). — [PMC7228941](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7228941/)
- Ship-borne stereo (Sensors 2026, 26(3):993): Hs error **≤0.14 m**, generally within **±10%** of concurrent buoys. A baseline-to-height ratio of 0.1–0.3 gave the best accuracy, and 10 Hz sampling captured surface evolution better. — [MDPI Sensors 26(3):993](https://www.mdpi.com/1424-8220/26/3/993); [PMC12899735](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12899735/)
- Ship-based stereo (JPO 2017) compared well with buoy statistics for all waves, breaking and non-breaking. — [JPO 47(4)](https://journals.ametsoc.org/view/journals/phoc/47/4/jpo-d-16-0187.1.xml)

**Learned stereo and neural reconstruction**
- "Neural reconstruction of 3D ocean wave hydrodynamics from camera sensing" (arXiv 2512.06024, 4 Dec 2025; Liu, Zhou, Yan, Guo, **Benetazzo**, Li) is a stereo network with an attention-augmented pyramid and physics constraints for 3-D velocity. It reports **millimetre-level** elevation error in the central region and dominant-frequency error **<0.01 Hz**, reconstructing about 2 million points in **1.35 s**. The abstract does not name the reference instrument. — [arXiv 2512.06024](https://arxiv.org/abs/2512.06024)
- RAFT-Stereo for wave-field reconstruction from a moving FPSO (Jebari, Florida Tech dissertation 2025) used Blender-simulated sea states and vessel motion. A related Ocean Engineering 2025 paper is **entirely synthetic**. — [FIT repository](https://repository.fit.edu/etd/1586); [Ocean Eng. 2025](https://www.sciencedirect.com/science/article/abs/pii/S0029801825006717)
- A thermal-stereo wave study fine-tuned RAFT-Stereo on synthetic infrared data. — [chatpaper summary](https://chatpaper.com/paper/52349)
- Physics-informed neural operators for real-time nonlinear wavefield reconstruction exist (arXiv 2508.03315). Abstract not fetched. — [arXiv 2508.03315](https://arxiv.org/pdf/2508.03315)

**Polarimetric slope sensing**
- A polarimetric camera on ocean cruises gave wave-slope variances consistent with Cox–Munk, using a modified Polarization Slope Sensing method (Malinowski, Gilerson et al., Optics Express 2024). — [NOAA repository](https://repository.library.noaa.gov/view/noaa/65975)
- Polarimetric slope and wave height were validated against electric and acoustic gauges in the Oregon State wave tank (AGU 2010). — [ADS abstract](https://ui.adsabs.harvard.edu/abs/2010AGUFMOS51B1315R)
- A Technion study used a low-cost polarimetric camera plus a neural network to infer dense elevation and slope fields, framed as having near-real-time potential. — [Technion CRIS](https://cris.technion.ac.il/en/publications/efficient-machine-learning-method-for-spatio-temporal-water-surfa)
- Integrating slope leaves an unknown constant, so relative heights are more reliable than absolute ones. — [ISPRS 2005](https://www.isprs.org/proceedings/2005/isrse/html/papers/920.pdf)

**Monocular**
- Monocular single-RGB-video fluid-surface reconstruction exists (Research Square preprint) but targets visual realism, not metric elevation. — [Research Square rs-10719630](https://www.researchsquare.com/article/rs-10719630/v1)
- A deep-learning spatiotemporal Hs method with ship-attitude compensation appeared in Ocean Engineering 2026. Not fetched; details unverified. — [ScienceDirect S0029801826003513](https://www.sciencedirect.com/science/article/abs/pii/S0029801826003513)

### Inferences
- The project has no stereo pair at either camera type, so WASS-class methods do not apply directly. They matter as the **ground-truth standard** a monocular method would need to approach (≤0.14 m Hs, about ±10%).
- Two monocular routes look physically plausible. (i) WASSfast's idea, constraining the reconstruction by the linear dispersion relation in (k, ω) space, can be applied monocularly as a **spectral** constraint on pixel-intensity time series. This is the cBathy-style idea already noted, and gives frequency and direction but not amplitude. (ii) Polarimetric or slope-statistics reasoning on buoy stills gives slope variance, not elevation.
- Amplitude (Hs) from monocular video stays the hardest quantity. Every learned monocular approach found regresses it end-to-end (Q4), with modest skill.

### Gaps
- The 2512.06024 reference instrument and dataset (likely a Benetazzo WASS site) could not be confirmed from the abstract.
- No error numbers were obtained for learned stereo (RAFT-Stereo) on real sea data against buoys.
- One search summary conflated the Sci Data 2020 stereo dataset with Datawell or SWIFT comparisons and "mm-level" claims. This was not verified and is excluded above.

---

## Q4. Which wave-specific deep-learning measurement tools (crest tracking, breaking detection, runup, celerity/bathymetry) have published code or weights and validated numbers?

### Takeaway
The mature, validated DL tools are **timestack** tools. Lange et al. 2023 WaveCrestDetection gives surf-zone depth RMSE 0.17 m with weights public. Buckley et al. 2024 runup networks are trained on more than 7,000 timestacks from six sites. Stringari et al. 2021 active-breaking classification reaches about 90% balanced accuracy with code public. Direct video-to-wave-parameter regressors (Kamagata 2026, Kim 2023, the JMSE 2024 shore-camera SWH study) report modest skill, for example Hs Pearson r 0.45 and direction r 0.83 in Kamagata. Classical video breaker-height methods are validated against buoys and wave basins.

### Cited Findings
**Crest detection, celerity and bathymetry**
- WaveCrestDetection (Lange, Fiedler, Merrifield, Guza, Coastal Engineering 2023) is a U-Net with an ImageNet ResNet18 encoder, about 17.8M parameters, PyTorch. It was trained on hand-annotated timestacks from Torrey Pines (two training images, one validation image, 256×256 tiles) and overfits after about 17 epochs. Pretrained weights are linked (Google Drive and `checkpoints/`). Surf-zone depth from crest celerity, with heuristic correction: **RMSE 0.17 m, bias 0.06 m**. From the seaward surf-zone edge to 10 m depth: **RMSE 0.87 m, bias 0.58 m** (112 transects, 8 days). Licence file present but its type was not read. — [AthinaLange/WaveCrestDetection](https://github.com/AthinaLange/WaveCrestDetection); [Coastal Eng. 2023](https://www.sciencedirect.com/science/article/pii/S0378383923000996). Conflict: another summary quotes RMSE "∼0.24 m" for the same paper, possibly uncorrected — [ResearchGate](https://www.researchgate.net/publication/372791204_UAV_video-based_estimates_of_nearshore_bathymetry). The companion UAV bathymetry repo is at [AthinaLange/UAV_Video-based_estimates_of_nearshore_bathymetry](https://github.com/AthinaLange/UAV_Video-based_estimates_of_nearshore_bathymetry)
- cBathy reaches about **0.51 m RMSE** after observing wave speeds for an average of 33 h, and fails to stay robust during storms and near breaking onset. This is the motivation for crest-tracking hybrids. — [WHOI/Elgar PDF](https://www2.whoi.edu/staff/elgar/wp-content/uploads/sites/153/2018/03/135.pdf)
- An FCNN for surf-zone bathymetry at Duck, NC took cBathy celerity and a prior from the Parameter Beach Tool, and ran in about 5 min on one CPU core. A related network was trained on synthetic Celeris imagery and applied to Argus timex and snapshot images, and was biased deep in one case. — [Semantic Scholar PDF](https://pdfs.semanticscholar.org/fe95/7c932344785ddd5d80c771fb48a9e4632f73.pdf); [Remote Sensing 12:3364](http://dx.doi.org/10.3390/rs12203364)
- Webcam wave-celerity depth inversion (Remote Sensing 2025, 17(13):2274) had depth-normalised RMSE generally below 20%. — [doi 10.3390/rs17132274](https://doi.org/10.3390/rs17132274)
- Surf-zone wave tracking (Kim et al., Atmosphere 2020) uses enhancement, an autoencoder and crest tracking. Celerity agreed well with reference values but degraded when crest lines were unclear or breaking was scattered. — [Atmosphere 11(3):304](https://doi.org/10.3390/atmos11030304)

**Breaking**
- Stringari et al. (Scientific Reports 2021) distinguish **active** breaking (visible bubble entrainment) from passive foam. The best model reached about **90% balanced accuracy** on the test set, and the dataset and code are public. — [PMC7878786](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7878786/); [arXiv 2101.11478](https://arxiv.org/pdf/2101.11478)
- Almar et al. (Coastal Engineering 2012) estimate individual breaker height directly from video at breaking onset: **9% mean error, 18% RMS error** against a dense wave-basin dataset. — [HAL record](https://hal-univ-tlse3.archives-ouvertes.fr/hal-00724936); [Almar et al. PDF](https://www.epoc.u-bordeaux.fr/indiv/bonneton/publications/papers/papers-nearshore/almar_etal_CE2011.pdf)
- At the Gold Coast (Gal et al., IEEE TGRS 2014), automated video breaking-height monitoring over three months was comparable to the buoy and to a surf reporter. — [CQU record](https://acquire.cqu.edu.au/articles/journal_contribution/Long-term_automated_monitoring_of_nearshore_wave_height_from_digital_video/13410476)
- The "Wave Pack" system uses a low camera at about 10 m above sea level to measure breaking and re-breaking height and period. — [typeset PDF](https://typeset.io/pdf/a-new-system-for-breakzone-location-and-the-measurement-of-3op6edn2il.pdf)
- Perranporth video surf heights were accurate to at least ±30%. — [WIT Press](https://witpress.com/elibrary/wit-transactions-on-ecology-and-the-environment/126/20629)

**Runup and swash**
- Buckley, Buscombe et al. (Coastal Engineering 2024) trained on more than 7,000 hand-digitised timestacks from six sites: an FCN from scratch and a fine-tuned transformer, with the transformer agreeing best. Both reproduce R2% well, but they under-predict significant swash and over-predict setup. Each takes under 10 s per timestack, against more than 5 min manually. — [USGS publication page](https://www.usgs.gov/publications/wave-runup-and-total-water-level-observations-time-series-imagery-several-sites); [pubs.usgs.gov](https://pubs.usgs.gov/publication/70257858)
- V-BeachNet (Chen et al.) is an FCN swash segmentation trained on 16 images and validated against lidar. — [par.nsf.gov](https://par.nsf.gov/servlets/purl/10513109)
- A lidar runup FCN was trained on 966 linescan time series. — [AMiner](https://www.aminer.org/pub/6419634c90e50fcafd5c0783/automated-extraction-of-a-depth-defined-wave-runup-time-series-from-lidar)
- A UAV DeepUNet waterline method (Chuang, Tsai, Chen, Remote Sensing 18(9):1385, 2026) gives zero-upcrossing wave period. It reports pixel F1/Dice of 75.0%, pixel accuracy of 99.6% and precision of 69.3%. — [MDPI Remote Sensing 18(9):1385](https://www.mdpi.com/2072-4292/18/9/1385)
- A Galveston (NCK 2025) CNN on five-channel timestacks was developed for dissipative beaches. — [NCK BOA 2025](https://www.nck-web.org/boa-2025/wave-runup-extraction-on-dissipative-beaches-new-video-based-methods)

**Direct video-to-wave-parameter regressors**
- **[NEW 2026-H2]** Kamagata et al., arXiv 2607.11998 (13 Jul 2026; 2026 IEEE conference paper). Architecture: V-JEPA ViT-S, a SlowFast dual stream, Farneback optical flow, and a multi-task head with an **Airy dispersion constraint** (λ=0.1). Trained on 6 annotated scenes and tested on held-out sites. Pearson r: **Hs 0.451, Hmax 0.578, Tp 0.643, Tz 0.680, direction 0.832**. Best R² is 0.246. Trained on a DGX A100. The abstract gives no buoy details. — [arXiv 2607.11998](https://arxiv.org/abs/2607.11998)
- **[NEW 2026-H2]** Kamagata et al., arXiv 2606.13302 (v1 11 Jun 2026, v2 19 Jul 2026) estimates Tp from coastal video. ROI comes from temporal pixel variance; models are LtViViT and TinyWaveNet; training is sim-to-real (synthetic, then silver labels, then expert labels); physics-guided output regularisation. The abstract gives no numbers. — [arXiv 2606.13302](https://arxiv.org/abs/2606.13302)
- Kim, Cho, Lee (Ocean Engineering 2023) classify wave height from monoscopic video. Single-frame CNNs were poor at high sea states, so a CNN-LSTM was used, with **ADCP** truth off Korea. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0029801823023867); [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4349127)
- A three-stage monocular approach from offshore infrastructure (Ocean Engineering 2024) uses **X-band radar** truth. A snippet quotes MAPE of 3.01% for Hs and 4.07% for peak frequency, but it is unconfirmed which paper that belongs to. — [ScienceDirect S0029801824026945](https://www.sciencedirect.com/science/article/abs/pii/S0029801824026945)
- Nearshore SWH inversion from instantaneous shore-video frames, plus wind-sea/swell classification (JMSE 2024, 12(11):2003). — [MDPI JMSE 12(11):2003](https://www.mdpi.com/2077-1312/12/11/2003)
- Wave direction from crest features in surf-zone video agreed with directional-spectrum analysis (Korean study, 2009). — [KoreaScience PDF](https://koreascience.kr/article/JAKO200916762902399.pdf)
- LEUCOTEA combines a CNN with optical flow for coastal wave height and tide. Tide was compared to a gauge. — [crossref RS16132275](https://api.crossref.org/works/10.3390%2FRS16132275)

### Inferences
**Pipeline fit for WebCOOS video, with labels from CDIP 20–30 km offshore**
- **Period, the most transferable quantity.** Wave period is conserved through shoaling and refraction, so nearshore Tp from video should match the offshore buoy's Tp better than Hs or direction does, apart from local wind sea and nonlinear harmonics. Routes: pixel-intensity spectra (cBathy-like), crest-passage frequency in timestacks (WaveCrestDetection output), or tracked-foam oscillation frequency (Q1). Celerity does **not** give period in the surf zone, because shallow-water celerity is roughly √(gh) and independent of period. Celerity plus measured frequency gives depth, which is what cBathy and Lange exploit. Celerity yields period only in intermediate or deep water outside the surf zone, and only if depth is known.
- **Direction.** Crest orientation or flow direction in a rectified frame gives the **local, refracted** direction. Comparing it with the offshore CDIP direction needs back-refraction (Snell's law over the bathymetry, which cBathy or a WaveCrestDetection celerity inversion can supply). Kamagata's direction r=0.83 suggests direction is learnable, but the label geometry differs per site.
- **Height.** The physical options are: (a) video breaker height (Almar: 9%/18% in a basin; Gal: comparable to a buoy), then inverted to offshore Hs with a breaker-index or shoaling assumption, which carries large uncertainty; (b) metric vertical displacement from tracks, which needs the Q2 calibration. Learned regressors so far give Hs r of only about 0.45 (Kamagata).
- **Interpretable intermediate features.** Feeding interpretable features (breaking fraction from Stringari, crest count and celerity from WaveCrestDetection, runup statistics from the Buckley models, pixel-spectrum peak frequency) into the existing V-JEPA 2 / DINOv3 probes is a low-compute way to test whether physical measurements add skill over end-to-end embeddings.
- **Compute.** All of these timestack tools are small (U-Net about 18M parameters; runup models under 10 s per timestack). They run trivially on an A100 or H200 and need no large GPU memory.

**Buoy stills**
- Crest, runup and bathymetry tools need video or timestacks and do not apply to the six independent still views per hour. Breaking fraction (whitecap coverage) is the one wave-specific DL output in this list that works on single stills and relates physically to sea state.

### Gaps
- Licences of WaveCrestDetection (Licence.md present, unread), deepwaves (Stringari) and the Buckley runup models were not confirmed. Whether the Buckley weights are public was not confirmed.
- Kamagata 2026 camera, resolution and label instrument were not available from the abstracts.
- No DL method that outputs offshore direction from shore video was found with numbers against a directional buoy, beyond Kamagata's Pearson r.

---

## Q5. Anything new June–November 2026?

### Takeaway
Wave-specific: two Kamagata et al. preprints (June and July 2026) on physics-guided video regression of Tp, Hs and direction. Point tracking: TrackEverything (Sep 2026), a long-horizon dense 3-D tracker, and the TAPVid-MV benchmark (Sep 2026). Other relevant items fall just before the window: TAPNext++ (Apr 2026), VGGT-Ω (May 2026) and DA3 (Nov 2025). Nothing new in the window targets ocean waves with point trackers, depth or 3-D foundation models.

### Cited Findings
- **2606.13302** (submitted 11 Jun 2026, v2 19 Jul 2026): physics-guided spatiotemporal learning for coastal Tp from video (LtViViT, TinyWaveNet, sim-to-real). — [arXiv 2606.13302](https://arxiv.org/abs/2606.13302)
- **2607.11998** (submitted 13 Jul 2026): V-JEPA + SlowFast + Farneback + Airy-constraint multi-task regression of Hs, Hmax, Tp, Tz and θ. Pearson r 0.45–0.83, best R² 0.246, DGX A100. — [arXiv 2607.11998](https://arxiv.org/abs/2607.11998)
- **2609.30222** TrackEverything (submitted 24 Sep 2026): dense 3-D tracking over more than 1,000 frames in 40 GB. — [arXiv 2609.30222](https://arxiv.org/abs/2609.30222)
- **2609.01899** TAPVid-MV (Sep 2026; abstract page not fetched): multi-view 3-D tracking benchmark; geometry recovery is the bottleneck. — [arXiv 2609.01899](https://arxiv.org/abs/2609.01899)
- VGGT-Ω licence status: a LearnOpenCV guide updated 3 Sep 2026 says the materials are noncommercial research only (paper dated 14 May 2026). — [LearnOpenCV](https://learnopencv.com/vggt-vs-vggt-%CF%89-vggt-omega-a-complete-guide-to-feed-forward-3d-reconstruction/); [arXiv 2605.15195](https://arxiv.org/abs/2605.15195)
- Earlier 2026, outside the window but recent: TAPNext++ (12 Apr 2026) — [alphaXiv 2604.10582](https://www.alphaxiv.org/abs/2604.10582); ship-borne stereo Hs ≤0.14 m vs buoys (Sensors 26(3), early 2026) — [MDPI](https://www.mdpi.com/1424-8220/26/3/993); DeepUNet waterline period (Remote Sensing 18(9), 2026) — [MDPI](https://www.mdpi.com/2072-4292/18/9/1385); AnthroTAP (CVPR 2026) — [CVF](https://openaccess.thecvf.com/content/CVPR2026/papers/Kim_AnthroTAP_Learning_Point_Tracking_with_Real-World_Motion_CVPR_2026_paper.pdf)

### Inferences
- The Kamagata 2607.11998 results (Hs r 0.45, Tp r 0.64, direction r 0.83, R² ≤ 0.25 on held-out sites) are the most recent published bar for **monocular coastal video** to wave parameters. Physics constraints such as the Airy dispersion term are already used in that line of work, so adding a dispersion constraint alone would not be novel. Explicit metric geometry and physical measurement features (Q1–Q4) would be.
- The June–November 2026 tracker and 3-D work keeps assuming rigid or articulated scenes. There is still no evidence these models handle free-surface water.

### Gaps
- No CoTracker, Depth Anything or MapAnything update dated June–November 2026 was found. Absence was not exhaustively confirmed, for example on GitHub release pages.
- TAPVid-MV details come from a search snippet only.
- The 2026 Ocean Engineering ship-attitude Hs paper (S0029801826003513) was not fetched, so its date within 2026 and its numbers are unknown.
