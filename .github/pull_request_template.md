## Summary

<!-- What does this change and why? Link issues: Closes #... / Refs Q-G1 -->

## Type
- [ ] Bug fix
- [ ] New data source / parser
- [ ] Methodology / QC / evaluation
- [ ] Model / experiment
- [ ] Documentation

## Checklist
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest` pass
- [ ] New or changed behaviour is tested (parsers on **real trimmed fixtures** with a provenance README)
- [ ] No data are silently dropped; QC returns flags (ADR 0001)
- [ ] New variables are registered in `schemas/variables.py` with CF standard names
- [ ] Scientific claims cite `docs/literature/bibliography.bib` or provider docs; verification level stated where relevant
- [ ] Numbers quoted in docs are reproducible by a script in `scripts/`
- [ ] Docs updated (datasheet / methodology / ADR if a design decision changed)
- [ ] `CHANGELOG.md` updated under *Unreleased*
- [ ] For results: manifest version, split protocol, leakage audit, baselines, and stratified metrics reported (`docs/REPRODUCIBILITY.md`)
