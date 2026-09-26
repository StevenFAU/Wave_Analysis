"""Coastal Data Information Program (CDIP, Scripps) adapter - planned for v0.2.0.

Access (verified 2026-09-25): THREDDS/OPeNDAP netCDF, no authentication::

    https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/{stn}p1_rt.nc
    https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/{stn}p1/{stn}p1_historic.nc

Key variables: ``waveTime``, ``waveTimeBounds`` (explicit acquisition window -
preferred over NDBC's rounded timestamps for synchronisation), ``waveHs``,
``waveTp``, ``waveTa``, ``waveDp``, ``waveTz``, ``waveFrequency``,
``waveBandwidth``, ``waveEnergyDensity``, ``waveA1Value``/``waveB1Value``/
``waveA2Value``/``waveB2Value`` (nautical-convention Fourier coefficients),
``waveFlagPrimary``; archive files also carry raw ``xyzZDisplacement``
(Datawell Waverider heave, typically 1.28 Hz), which enables *both* spectral
and zero-crossing labels from the same record (as in Yang et al. 2026).

Implementation plan: ``xarray.open_dataset`` (``netcdf`` extra), map to
:class:`~wave_analysis.processing.spectra.FrequencySpectrum` via
:func:`~wave_analysis.processing.spectra.ndbc_from_fourier`, preserve CDIP flags
as ``source_qc_flag``. See ``docs/datasets/cdip.md``.
"""

from wave_analysis.sources.base import NotYetImplementedSource


class CDIPSource(NotYetImplementedSource):
    """Placeholder; see module docstring."""

    dataset_id = "cdip"
    doc_path = "docs/datasets/cdip.md"
    phase = "v0.2.0"
