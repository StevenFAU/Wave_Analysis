# U.S. IOOS Regional Associations and ERDDAP

Registry id `ioos_erddap` · planned **v0.2.0** · status **candidate**.

## Scope

The U.S. Integrated Ocean Observing System aggregates eleven regional
associations. Relevant for this project are **SECOORA** (Southeast, including
Florida), **PacIOOS** (Pacific Islands, co-located cameras and waves), SCCOOS,
and GCOOS. Most publish through **ERDDAP**, which offers:

- server-side subsetting by time, space, and variable;
- output in netCDF (CF), CSV, JSON, and more;
- dataset-level ACDD metadata (`license`, `creator`, `standard_name`s), which the
  planned adapter will copy into the registry automatically.

## Plan

`sources/erddap.py` will wrap `erddapy`. Discovery searches each server by CF
standard name (`sea_surface_wave_significant_height`, etc.), so new wave datasets
are found without hand-curation.

## Open questions

- Confirm endpoint URLs and which datasets duplicate NDBC/CDIP streams.
  Duplicates must be detected so the same measurement is never counted twice,
  for example across splits.
- Do SECOORA ERDDAP datasets expose QARTOD flags per value (IOOS DMAC
  practice)? If so, preserve them as `source_qc_flag`.
