"""Generic ERDDAP adapter (IOOS regional associations: SECOORA, PacIOOS, ...) - planned for v0.2.0.

ERDDAP exposes tabledap/griddap services with machine-readable metadata and
CF/ACDD attributes, and supports server-side subsetting by time, space and
variable (e.g. ``{server}/tabledap/{dataset_id}.csv?time,sea_surface_wave_significant_height&time>=...``).
The planned implementation uses ``erddapy`` and reads each dataset's global
attributes into the registry so licence, provider and QC conventions are
captured automatically. See ``docs/datasets/ioos.md``.
"""

from wave_analysis.sources.base import NotYetImplementedSource


class ERDDAPSource(NotYetImplementedSource):
    """Placeholder; see module docstring."""

    dataset_id = "ioos_erddap"
    doc_path = "docs/datasets/ioos.md"
    phase = "v0.2.0"
