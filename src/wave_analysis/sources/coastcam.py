"""USGS CoastCam adapter - planned for v0.2.0 / v0.3.0.

CoastCam data releases provide Argus-style image products (snap, timex,
variance, bright, dark), runup timestacks, and - critically - the intrinsic
(IO) and extrinsic (EO) calibration needed to map pixels to world coordinates.
First target: Madeira Beach, Florida (USGS data release doi:10.5066/P9SGRTF1),
to be paired with nearby NDBC/CDIP wave buoys. See ``docs/datasets/coastcam.md``.
"""

from wave_analysis.sources.base import NotYetImplementedSource


class CoastCamSource(NotYetImplementedSource):
    """Placeholder; see module docstring."""

    dataset_id = "usgs_coastcam"
    doc_path = "docs/datasets/coastcam.md"
    phase = "v0.3.0"
