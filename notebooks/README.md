# Notebooks

Notebooks are for **exploration and figures only**. Rules:

1. No transformation that a result depends on may live only in a notebook.
   Move it into `src/wave_analysis` with tests (repository specification §18).
2. Notebooks read standardized data and manifests. They do not download raw
   data themselves.
3. Name as `YYYY-MM-DD_<initials>_<topic>.ipynb` under `exploratory/`.
4. Clear outputs before committing (large outputs bloat history), unless the
   notebook is a documented figure source.
5. The first cell records the Git commit and the manifest versions used.
