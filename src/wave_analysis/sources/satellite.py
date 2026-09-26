"""Satellite adapters (Sentinel-1 SAR, Sentinel-2/Landsat/HLS, altimetry) - planned for v0.4.0.

Satellite products provide *context and coarse supervision*, not frame-level
wave labels: SAR Level-2 OCN wave-mode spectra and swell parameters (all-weather),
altimeter along-track Hs (independent, large-scale reference), and multispectral
imagery for surf-zone and turbidity context. See ``docs/datasets/satellite_sar.md``
and ``docs/datasets/multispectral.md``.
"""

from wave_analysis.sources.base import NotYetImplementedSource


class SentinelSource(NotYetImplementedSource):
    """Placeholder; see module docstring."""

    dataset_id = "sentinel"
    doc_path = "docs/datasets/satellite_sar.md"
    phase = "v0.4.0"
