"""NOAA National Data Buoy Center (NDBC) adapter.

See ``docs/datasets/ndbc.md`` for the full dataset documentation (provider,
formats, timestamp conventions, QC, known limitations, suitable ML uses).
"""

from wave_analysis.sources.ndbc.inventory import InventoryResult, run_inventory
from wave_analysis.sources.ndbc.parsers import (
    ParsedTable,
    SpectralBlock,
    assemble_spectrum,
    parse_adcp,
    parse_spectral,
    parse_spectral_summary,
    parse_stdmet,
)
from wave_analysis.sources.ndbc.products import PRODUCTS, SPECTRAL_SET, NDBCProduct
from wave_analysis.sources.ndbc.source import NDBCSource

__all__ = [
    "PRODUCTS",
    "SPECTRAL_SET",
    "InventoryResult",
    "NDBCProduct",
    "NDBCSource",
    "ParsedTable",
    "SpectralBlock",
    "assemble_spectrum",
    "parse_adcp",
    "parse_spectral",
    "parse_spectral_summary",
    "parse_stdmet",
    "run_inventory",
]
