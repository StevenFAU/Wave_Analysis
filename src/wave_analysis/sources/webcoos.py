"""WebCOOS (Webcam Coastal Observation System; SECOORA / NOAA IOOS) adapter - planned.

WebCOOS operates coastal webcams in the U.S. Southeast (e.g. Jennette's Pier
and Currituck, NC; Oak Island, NC) with archived imagery/video and an API. Several
WebCOOS camera identifiers (``jennette_north``, ``oakisland_east``,
``currituck_hampton_inn``) appear in the Yin et al. (2025) wave-breaking video
dataset used as the held-out test set by Kamagata et al. (2026), which makes
WebCOOS a natural bridge between the literature and instrument-labelled data.
Access terms must be confirmed before bulk acquisition. See ``docs/datasets/webcoos.md``.
"""

from wave_analysis.sources.base import NotYetImplementedSource


class WebCOOSSource(NotYetImplementedSource):
    """Placeholder; see module docstring."""

    dataset_id = "webcoos"
    doc_path = "docs/datasets/webcoos.md"
    phase = "v0.2.0"
