"""Data exports for the public dashboard (GitHub Pages).

Two kinds of output, both plain JSON:

* the **catalog** (:mod:`.catalog`): built from files in this repository
  (dataset and camera-site registries, bibliography, verification log). CI
  builds it on every push to ``main``.
* the **live status** (:mod:`.status`, :mod:`.seastate`): built on the
  collector host from the raw archive and ledgers, and published hourly to the
  ``dashboard-data`` branch by ``scripts/publish_dashboard.sh``.

The web front end lives in ``dashboard/`` and reads only these files. See
ADR 0009 and ``docs/operations/dashboard.md``.
"""
