"""Wave_Analysis: reproducible multimodal ocean-wave data and modelling framework.

The package is organised by pipeline stage (see ``docs/ARCHITECTURE.md``):

``sources``
    Provider adapters (NDBC, CDIP, ERDDAP, CoastCam, ...) implementing a common
    discover / fetch / parse / normalize / validate interface.
``ingest``
    Byte-preserving downloads, SHA-256 checksums, and acquisition manifests.
``schemas``
    Typed records for stations, sensors, observations, spectra, and imagery.
``physics``
    Linear wave theory, time-domain wave statistics, and parametric spectra.
``processing``
    Spectral analysis, synchronisation, geospatial and solar geometry.
``qc``
    Layered quality control (provider, QARTOD, physical, imagery, sync, ML).
``datasets``
    Leakage-aware ML manifest construction, splits, and provenance records.
``evaluation``
    Regression, circular, classification, agreement, and uncertainty metrics.
``models``
    Statistical baselines and optional reference deep-learning architectures.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("wave-analysis")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
