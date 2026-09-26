"""Common interface for provider adapters.

Every provider (NDBC, CDIP, ERDDAP servers, CoastCam, satellites, ...) is
wrapped in a :class:`DataSource` with the same five stages, mirroring the
raw -> standardized -> validated layers in ``docs/ARCHITECTURE.md``:

``discover``
    What exists upstream? Returns an availability table indexed by
    *station + product + period* (never by provider as a whole).
``fetch``
    Retrieve raw bytes unchanged; record every attempt in a manifest.
``parse``
    Lossless transcription of a raw file into an in-memory table in the
    provider's own names and units (missing-value sentinels -> NaN, nothing else).
``normalize``
    Map to the common observation schema: canonical variable names, CF
    standard names, canonical units, UTC timestamps with explicit time reference.
``validate``
    Run provider-appropriate QC; return per-test flag results (never drop rows).

Keeping ``parse`` and ``normalize`` separate means a unit-conversion or
naming bug can be fixed and re-run without re-parsing or re-downloading, and
the parsed layer can be diffed against the raw file in tests.

Signatures are deliberately permissive (``*args, **kwargs``) because each
provider's natural parameters differ (station lists, bounding boxes, dataset
ids); adapters document their concrete signatures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import pandas as pd

from wave_analysis.ingest.manifest import ManifestEntry
from wave_analysis.qc.flags import QCTestResult


class DataSource(ABC):
    """Abstract provider adapter."""

    #: Registry identifier (``data/registry/datasets.yaml``).
    dataset_id: str

    @abstractmethod
    def discover(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Return an availability table (one row per station/product/period)."""

    @abstractmethod
    def fetch(self, *args: Any, **kwargs: Any) -> Sequence[ManifestEntry]:
        """Download raw files under ``dest_root``; return manifest entries for every attempt."""

    @abstractmethod
    def parse(self, *args: Any, **kwargs: Any) -> Any:
        """Parse one raw file into a provider-native in-memory representation."""

    @abstractmethod
    def normalize(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Convert parsed data to the common observation schema."""

    @abstractmethod
    def validate(self, *args: Any, **kwargs: Any) -> list[QCTestResult]:
        """Run QC and return per-test results."""


class NotYetImplementedSource(DataSource):
    """Placeholder adapter for registry sources planned in later phases.

    Instantiating documents the plan; calling any stage raises
    ``NotImplementedError`` with a pointer to the relevant dataset document.
    """

    doc_path: str = "docs/datasets/README.md"
    phase: str = "v0.2.0"

    def _nyi(self) -> NotImplementedError:
        return NotImplementedError(
            f"{type(self).__name__} is planned for {self.phase}; see {self.doc_path}"
        )

    def discover(self, *args: Any, **kwargs: Any) -> pd.DataFrame:  # noqa: D102
        raise self._nyi()

    def fetch(self, *args: Any, **kwargs: Any) -> Sequence[ManifestEntry]:  # noqa: D102
        raise self._nyi()

    def parse(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D102
        raise self._nyi()

    def normalize(self, *args: Any, **kwargs: Any) -> pd.DataFrame:  # noqa: D102
        raise self._nyi()

    def validate(self, *args: Any, **kwargs: Any) -> list[QCTestResult]:  # noqa: D102
        raise self._nyi()
